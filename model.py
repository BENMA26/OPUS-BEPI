import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from EGAT import EGAT,AE
from torch.nn.utils.rnn import pad_sequence,pack_sequence,pack_padded_sequence,pad_packed_sequence

import torch
import torch.nn as nn
import math

class LightweightSelfAttention(nn.Module):
    def __init__(self, embed_dim=21):
        super().__init__()
        self.scale = math.sqrt(embed_dim)

    def forward(self, x):
        # x: [batch_size, N, 21, 6]
        B, N, D, L = x.shape  # L = 6 (seq_len for attention), D = 21

        # Step 1: reshape to [B*N, L, D] → treat each (batch, position) as independent
        x = x.permute(0, 1, 3, 2)      # [B, N, 6, 21]
        x = x.reshape(B * N, L, D)     # [B*N, 6, 21]

        # Step 2: self-attention over L=6 (Q=K=V=x, no projection)
        Q = K = V = x                  # [B*N, 6, 21]
        attn_scores = torch.bmm(Q, K.transpose(-2, -1)) / self.scale  # [B*N, 6, 6]
        attn_weights = torch.softmax(attn_scores, dim=-1)            # [B*N, 6, 6]
        attended = torch.bmm(attn_weights, V)                        # [B*N, 6, 21]

        # Step 3: aggregate over L=6 → [B*N, 21]
        out = attended.mean(dim=1)     # [B*N, 21]

        # Step 4: reshape back to [B, N, 21]
        out = out.view(B, N, D)        # [batch_size, N, 21]

        return out

class ParametricSelfAttention(nn.Module):
    def __init__(self, embed_dim=21, attn_dim=32, use_output_proj=True):
        """
        embed_dim: 输入特征维度（你的默认是 21）
        attn_dim:  Q/K/V 投影后的维度（attention hidden size）
        use_output_proj: 是否使用输出投影层
        """
        super().__init__()
        self.scale = math.sqrt(attn_dim)

        # 可学习的 Q/K/V 投影层
        self.W_q = nn.Linear(embed_dim, attn_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, attn_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, attn_dim, bias=False)

        # 可选：输出投影（将 attn_dim 变回 embed_dim）
        self.use_output_proj = use_output_proj
        if use_output_proj:
            self.W_o = nn.Linear(attn_dim, embed_dim, bias=False)

    def forward(self, x):
        # x: [B, N, 21, 6]
        B, N, D, L = x.shape  # D = embed_dim, L = 6

        # Step 1: reshape → [B*N, L, D]
        x = x.permute(0, 1, 3, 2)   # [B, N, 6, 21]
        x = x.reshape(B * N, L, D)  # [B*N, 6, 21]

        # Step 2: Q/K/V 投影，得到 [B*N, 6, attn_dim]
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Step 3: self-attention
        attn_scores = torch.bmm(Q, K.transpose(-2, -1)) / self.scale  # [B*N, 6, 6]
        attn_weights = torch.softmax(attn_scores, dim=-1)            # [B*N, 6, 6]
        attended = torch.bmm(attn_weights, V)                        # [B*N, 6, attn_dim]

        # Step 4: 聚合 over L=6 → [B*N, attn_dim]
        out = attended.mean(dim=1)  # [B*N, attn_dim]

        # Step 5: 可选输出投影 → [B*N, D]
        if self.use_output_proj:
            out = self.W_o(out)    # [B*N, 21]

        # Step 6: reshape 回原格式 [B, N, D]
        out = out.view(B, N, -1)

        return out

class GraphBepi(pl.LightningModule):
    def __init__(
        self, 
        feat_dim=2560, hidden_dim=256, 
        exfeat_dim=13, edge_dim=51, 
        augment_eps=0.05, dropout=0.2, 
        lr=1e-6, metrics=None, result_path=None
    ):
        super().__init__()
        self.metrics=metrics
        self.path=result_path
        # loss function
        self.loss_fn=nn.BCELoss()
        # Hyperparameters
        self.exfeat_dim=exfeat_dim
        self.augment_eps = augment_eps
        self.lr = lr
        self.cls = 1
        bias=False
        self.W_v = nn.Linear(feat_dim, hidden_dim, bias=bias)
        self.W_u1 = AE(exfeat_dim,hidden_dim,hidden_dim, bias=bias)
        self.edge_linear=nn.Sequential(
            nn.Linear(edge_dim,hidden_dim//4, bias=True),
            nn.ELU(),
        )
        self.gat=EGAT(2*hidden_dim,hidden_dim,hidden_dim//4,dropout)
        self.lstm1 = nn.LSTM(hidden_dim,hidden_dim//2,3,batch_first=True,bidirectional=True,dropout=dropout)
        self.lstm2 = nn.LSTM(hidden_dim,hidden_dim//2,3,batch_first=True,bidirectional=True,dropout=dropout)
        # output
        self.mlp=nn.Sequential(
            nn.Linear(4*hidden_dim,hidden_dim,bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim,1,bias=True),
            nn.Sigmoid()
        )
        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, V, edge):
        h=[]
        V = pad_sequence(V, batch_first=True, padding_value=0).float()
        mask=V.sum(-1)!=0
        if self.training and self.augment_eps > 0:
            aug=torch.randn_like(V)
            aug[~mask]=0
            V = V+self.augment_eps * aug
        mask=mask.sum(1)
        feats,exfeats=self.W_v(V[:,:,:-self.exfeat_dim]),self.W_u1(V[:,:,-self.exfeat_dim:])
        x_gcns=[]
        for i in range(len(V)):
            E=self.edge_linear(edge[i]).permute(2,0,1)
            x1,x2=feats[i,:mask[i]],exfeats[i,:mask[i]]
            x_gcn=torch.cat([x1,x2],-1)
            x_gcn,E=self.gat(x_gcn,E)
            x_gcns.append(x_gcn)
        feats=pack_padded_sequence(feats,mask.cpu(),True,False)
        exfeats=pack_padded_sequence(exfeats,mask.cpu(),True,False)
        feats=pad_packed_sequence(self.lstm1(feats)[0],True)[0]
        exfeats=pad_packed_sequence(self.lstm2(exfeats)[0],True)[0]
        x_attns=torch.cat([feats,exfeats],-1)
        
        x_attns=[x_attns[i,:mask[i]] for i in range(len(x_attns))]
        h=[torch.cat([x_attn,x_gcn],-1) for x_attn,x_gcn in zip(x_attns,x_gcns)]
        h=torch.cat(h,0)
        return self.mlp(h)
    
    def training_step(self, batch, batch_idx): 
        feat, edge, y = batch
        pred = self(feat, edge).squeeze(-1)
        loss=self.loss_fn(pred,y.float())
        self.log('train_loss', loss.cpu().item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if self.metrics is not None:
            #result=self.metrics.calc_prc(pred.detach().clone(),y.detach().clone())
            result = self.metrics(pred.detach().clone(), y.detach().clone())
            self.log('train_auc', result['AUROC'], on_epoch=True, prog_bar=True, logger=True)
            self.log('train_prc', result['AUPRC'], on_epoch=True, prog_bar=True, logger=True)
            self.log('train_mcc', result['MCC'], on_epoch=True, prog_bar=True, logger=True)
            self.log('traom_f1', result['F1'], on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        feat, edge, y = batch
        pred = self(feat, edge).squeeze(-1)
        # 手动保存当前 batch 输出
        self.validation_step_outputs.append((pred.detach(), y.detach()))
        return pred, y

    def on_validation_epoch_start(self):
        # 初始化输出缓存
        self.validation_step_outputs = []

    def on_validation_epoch_end(self):
        # 聚合保存的 batch 输出
        outputs = self.validation_step_outputs
        if len(outputs) == 0:
            return

        pred, y = zip(*outputs)
        pred = torch.cat(pred, 0)
        y = torch.cat(y, 0)
        loss = self.loss_fn(pred, y.float())

        self.log('val_loss', loss.cpu().item(), on_epoch=True, prog_bar=True, logger=True)
        if self.metrics is not None:
            result = self.metrics(pred.detach().clone(), y.detach().clone())
            self.log('val_AUROC', result['AUROC'], on_epoch=True, prog_bar=True, logger=True)
            self.log('val_AUPRC', result['AUPRC'], on_epoch=True, prog_bar=True, logger=True)
            self.log('val_mcc', result['MCC'], on_epoch=True, prog_bar=True, logger=True)
            self.log('val_f1', result['F1'], on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        feat, edge, y = batch
        pred = self(feat, edge).squeeze(-1)
        # 缓存用于 epoch-end 聚合（与 validation 一致）
        self.test_step_outputs.append((pred.detach(), y.detach()))
        # 可选：也可以返回字典，但这里我们依赖缓存
        return {"pred": pred, "y": y}

    def on_test_epoch_start(self):
        """初始化测试输出缓存"""
        self.test_step_outputs = []

    def on_test_epoch_end(self):
        """聚合所有测试批次的结果"""
        if len(self.test_step_outputs) == 0:
            return

        # 解包并拼接
        preds, ys = zip(*self.test_step_outputs)
        pred = torch.cat(preds, dim=0)
        y = torch.cat(ys, dim=0)

        # 计算 loss
        loss = self.loss_fn(pred, y.float())
        self.log("test_loss", loss.cpu().item(), on_epoch=True, prog_bar=True, logger=True)

        # 保存预测结果
        if self.path:
            os.makedirs(self.path, exist_ok=True)
            torch.save({'pred': pred.cpu(), 'gt': y.cpu()}, f'{self.path}/result.pkl')

        # 计算并记录指标
        if self.metrics is not None:
            result = self.metrics(pred.detach().clone(), y.detach().clone())
            self.log('test_AUROC', result['AUROC'], on_epoch=True, prog_bar=True, logger=True)
            self.log('test_AUPRC', result['AUPRC'], on_epoch=True, prog_bar=True, logger=True)
            self.log('test_recall', result['RECALL'], on_epoch=True, prog_bar=True, logger=True)
            self.log('test_precision', result['PRECISION'], on_epoch=True, prog_bar=True, logger=True)
            self.log('test_f1', result['F1'], on_epoch=True, prog_bar=True, logger=True)
            self.log('test_mcc', result['MCC'], on_epoch=True, prog_bar=True, logger=True)
            self.log('test_bacc', result['BACC'], on_epoch=True, prog_bar=True, logger=True)
            self.log('test_threshold', result['threshold'], on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), betas=(0.9, 0.99), lr=self.lr, weight_decay=1e-5, eps=1e-5)

class GraphBepi_att(pl.LightningModule):
    def __init__(
        self, 
        feat_dim=2581, hidden_dim=256, 
        exfeat_dim=13, edge_dim=51, 
        augment_eps=0.05, dropout=0.2, 
        lr=1e-6, metrics=None, result_path=None
    ):
        super().__init__()
        self.metrics=metrics
        self.path=result_path
        # loss function
        self.loss_fn=nn.BCELoss()
        # Hyperparameters
        self.exfeat_dim=exfeat_dim
        self.augment_eps = augment_eps
        self.lr = lr
        self.cls = 1
        bias=False
        self.W_v = nn.Linear(feat_dim, hidden_dim, bias=bias)
        self.W_u1 = AE(exfeat_dim,hidden_dim,hidden_dim, bias=bias)
        self.edge_linear=nn.Sequential(
            nn.Linear(edge_dim,hidden_dim//4, bias=True),
            nn.ELU(),
        )
        self.gat=EGAT(2*hidden_dim,hidden_dim,hidden_dim//4,dropout)
        self.lstm1 = nn.LSTM(hidden_dim,hidden_dim//2,3,batch_first=True,bidirectional=True,dropout=dropout)
        self.lstm2 = nn.LSTM(hidden_dim,hidden_dim//2,3,batch_first=True,bidirectional=True,dropout=dropout)
        # output
        self.mlp=nn.Sequential(
            nn.Linear(4*hidden_dim,hidden_dim,bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim,1,bias=True),
            nn.Sigmoid()
        )
        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.attn = ParametricSelfAttention()

    def forward(self, fold_token, V, edge):
        h=[]
        #print(fold_token[0].shape)
        #print(V[0].shape)
        fold_token = pad_sequence(fold_token, batch_first=True, padding_value=0).float()
        fold_token = self.attn(fold_token)
        V = pad_sequence(V, batch_first=True, padding_value=0).float()
        mask=V.sum(-1)!=0
        V = torch.cat([V,fold_token],dim=-1)
        #V = torch.
        mask=V.sum(-1)!=0
        if self.training and self.augment_eps > 0:
            aug=torch.randn_like(V)
            aug[~mask]=0
            V = V+self.augment_eps * aug
        mask=mask.sum(1)
        feats,exfeats=self.W_v(V[:,:,:-self.exfeat_dim]),self.W_u1(V[:,:,-self.exfeat_dim:])
        x_gcns=[]
        for i in range(len(V)):
            E=self.edge_linear(edge[i]).permute(2,0,1)
            x1,x2=feats[i,:mask[i]],exfeats[i,:mask[i]]
            x_gcn=torch.cat([x1,x2],-1)
            x_gcn,E=self.gat(x_gcn,E)
            x_gcns.append(x_gcn)
        feats=pack_padded_sequence(feats,mask.cpu(),True,False)
        exfeats=pack_padded_sequence(exfeats,mask.cpu(),True,False)
        feats=pad_packed_sequence(self.lstm1(feats)[0],True)[0]
        exfeats=pad_packed_sequence(self.lstm2(exfeats)[0],True)[0]
        x_attns=torch.cat([feats,exfeats],-1)
        
        x_attns=[x_attns[i,:mask[i]] for i in range(len(x_attns))]
        h=[torch.cat([x_attn,x_gcn],-1) for x_attn,x_gcn in zip(x_attns,x_gcns)]
        h=torch.cat(h,0)
        return self.mlp(h)
    
    def training_step(self, batch, batch_idx): 
        fold_token, feat, edge, y = batch
        pred = self(fold_token, feat, edge).squeeze(-1)
        loss=self.loss_fn(pred,y.float())
        self.log('train_loss', loss.cpu().item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if self.metrics is not None:
            #result=self.metrics.calc_prc(pred.detach().clone(),y.detach().clone())
            result = self.metrics(pred.detach().clone(), y.detach().clone())
            self.log('train_auc', result['AUROC'], on_epoch=True, prog_bar=True, logger=True)
            self.log('train_prc', result['AUPRC'], on_epoch=True, prog_bar=True, logger=True)
            self.log('train_mcc', result['MCC'], on_epoch=True, prog_bar=True, logger=True)
            self.log('traom_f1', result['F1'], on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        fold_token, feat, edge, y = batch
        pred = self(fold_token, feat, edge).squeeze(-1)
        # 手动保存当前 batch 输出
        self.validation_step_outputs.append((pred.detach(), y.detach()))
        return pred, y

    def on_validation_epoch_start(self):
        # 初始化输出缓存
        self.validation_step_outputs = []

    def on_validation_epoch_end(self):
        # 聚合保存的 batch 输出
        outputs = self.validation_step_outputs
        if len(outputs) == 0:
            return

        pred, y = zip(*outputs)
        pred = torch.cat(pred, 0)
        y = torch.cat(y, 0)
        loss = self.loss_fn(pred, y.float())

        self.log('val_loss', loss.cpu().item(), on_epoch=True, prog_bar=True, logger=True)
        if self.metrics is not None:
            result = self.metrics(pred.detach().clone(), y.detach().clone())
            self.log('val_AUROC', result['AUROC'], on_epoch=True, prog_bar=True, logger=True)
            self.log('val_AUPRC', result['AUPRC'], on_epoch=True, prog_bar=True, logger=True)
            self.log('val_mcc', result['MCC'], on_epoch=True, prog_bar=True, logger=True)
            self.log('val_f1', result['F1'], on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        fold_token, feat, edge, y = batch
        pred = self(old_token, feat, edge).squeeze(-1)
        return pred,y
    
    def on_test_epoch_end(self):
        # 兼容新版 Lightning：outputs 不再自动传入
        outputs = getattr(self.trainer, "model_outputs", None)
        if outputs is not None and len(outputs) > 0:
            outputs = outputs[0]  # DDP 情况下取 rank 0 输出
        else:
            return

        pred, y = [], []
        for i, j in outputs:
            pred.append(i)
            y.append(j)
        pred = torch.cat(pred, 0)
        y = torch.cat(y, 0)

        loss = self.loss_fn(pred, y.float())

        # 保存结果
        if self.path:
            os.makedirs(self.path, exist_ok=True)
            torch.save({'pred': pred.cpu(), 'gt': y.cpu()}, f'{self.path}/result.pkl')

        # 计算指标
        if self.metrics is not None:
            result = self.metrics(pred.detach().clone(), y.detach().clone())
            self.log('test_loss', loss.cpu().item(), on_epoch=True, prog_bar=True, logger=True)
            self.log('test_AUROC', result['AUROC'], on_epoch=True, prog_bar=True, logger=True)
            self.log('test_AUPRC', result['AUPRC'], on_epoch=True, prog_bar=True, logger=True)
            self.log('test_recall', result['RECALL'], on_epoch=True, prog_bar=True, logger=True)
            self.log('test_precision', result['PRECISION'], on_epoch=True, prog_bar=True, logger=True)
            self.log('test_f1', result['F1'], on_epoch=True, prog_bar=True, logger=True)
            self.log('test_mcc', result['MCC'], on_epoch=True, prog_bar=True, logger=True)
            self.log('test_bacc', result['BACC'], on_epoch=True, prog_bar=True, logger=True)
            self.log('test_threshold', result['threshold'], on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), betas=(0.9, 0.99), lr=self.lr, weight_decay=1e-5, eps=1e-5)