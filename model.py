import os
import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from EGAT import EGAT, AE
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class LightweightSelfAttention(nn.Module):
    def __init__(self, embed_dim=21):
        super().__init__()
        self.scale = math.sqrt(embed_dim)

    def forward(self, x):
        B, N, D, L = x.shape
        x = x.permute(0, 1, 3, 2).reshape(B * N, L, D)
        attn = torch.softmax(torch.bmm(x, x.transpose(-2, -1)) / self.scale, dim=-1)
        return torch.bmm(attn, x).mean(dim=1).view(B, N, D)


class ParametricSelfAttention(nn.Module):
    def __init__(self, embed_dim=21, attn_dim=32, use_output_proj=True):
        super().__init__()
        self.scale = math.sqrt(attn_dim)
        self.W_q = nn.Linear(embed_dim, attn_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, attn_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, attn_dim, bias=False)
        self.use_output_proj = use_output_proj
        if use_output_proj:
            self.W_o = nn.Linear(attn_dim, embed_dim, bias=False)

    def forward(self, x):
        B, N, D, L = x.shape
        x = x.permute(0, 1, 3, 2).reshape(B * N, L, D)
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)
        attn = torch.softmax(torch.bmm(Q, K.transpose(-2, -1)) / self.scale, dim=-1)
        out = torch.bmm(attn, V).mean(dim=1)
        if self.use_output_proj:
            out = self.W_o(out)
        return out.view(B, N, -1)


class BaseLightningModel(pl.LightningModule):
    """Shared Lightning training/validation/test logic for all GraphBepi variants."""

    def _forward_batch(self, batch):
        raise NotImplementedError

    # ── Training ───────────────────────────────────────────────────────────────
    def training_step(self, batch, batch_idx):
        pred, y = self._forward_batch(batch)
        loss = self.loss_fn(pred, y.float())
        self.log('train_loss', loss.cpu().item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if self.metrics is not None:
            result = self.metrics(pred.detach().clone(), y.detach().clone())
            self.log('train_auc', result['AUROC'], on_epoch=True, prog_bar=True, logger=True)
            self.log('train_prc', result['AUPRC'], on_epoch=True, prog_bar=True, logger=True)
            self.log('train_mcc', result['MCC'], on_epoch=True, prog_bar=True, logger=True)
            self.log('train_f1', result['F1'], on_epoch=True, prog_bar=True, logger=True)
        return loss

    # ── Validation ─────────────────────────────────────────────────────────────
    def on_validation_epoch_start(self):
        self.validation_step_outputs = []

    def validation_step(self, batch, batch_idx):
        pred, y = self._forward_batch(batch)
        self.validation_step_outputs.append((pred.detach(), y.detach()))
        return pred, y

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return
        pred = torch.cat([o[0] for o in self.validation_step_outputs])
        y = torch.cat([o[1] for o in self.validation_step_outputs])
        self.log('val_loss', self.loss_fn(pred, y.float()).cpu().item(),
                 on_epoch=True, prog_bar=True, logger=True)
        if self.metrics is not None:
            result = self.metrics(pred.detach().clone(), y.detach().clone())
            self.log('val_AUROC', result['AUROC'], on_epoch=True, prog_bar=True, logger=True)
            self.log('val_AUPRC', result['AUPRC'], on_epoch=True, prog_bar=True, logger=True)
            self.log('val_mcc', result['MCC'], on_epoch=True, prog_bar=True, logger=True)
            self.log('val_f1', result['F1'], on_epoch=True, prog_bar=True, logger=True)

    # ── Test ───────────────────────────────────────────────────────────────────
    def on_test_epoch_start(self):
        self.test_step_outputs = []

    def test_step(self, batch, batch_idx):
        pred, y = self._forward_batch(batch)
        self.test_step_outputs.append((pred.detach(), y.detach()))
        return {"pred": pred, "y": y}

    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            return
        pred = torch.cat([o[0] for o in self.test_step_outputs])
        y = torch.cat([o[1] for o in self.test_step_outputs])
        self.log("test_loss", self.loss_fn(pred, y.float()).cpu().item(),
                 on_epoch=True, prog_bar=True, logger=True)
        if self.path:
            os.makedirs(self.path, exist_ok=True)

            # Calculate optimal threshold
            threshold = self.metrics.calc_thresh(pred.detach().clone(), y.detach().clone()) if self.metrics else 0.5

            # Convert predictions to binary using threshold
            pred_binary = (pred.cpu() > threshold).long()
            y_cpu = y.cpu().long()

            # Calculate TP, TN, FP, FN for each position
            tp = ((pred_binary == 1) & (y_cpu == 1)).long()
            tn = ((pred_binary == 0) & (y_cpu == 0)).long()
            fp = ((pred_binary == 1) & (y_cpu == 0)).long()
            fn = ((pred_binary == 0) & (y_cpu == 1)).long()

            # Save results with classification details
            torch.save({
                'pred': pred.cpu(),
                'gt': y.cpu(),
                'threshold': threshold,
                'tp': tp,
                'tn': tn,
                'fp': fp,
                'fn': fn
            }, f'{self.path}/result.pkl')
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
        return torch.optim.Adam(self.parameters(), betas=(0.9, 0.99),
                                lr=self.lr, weight_decay=1e-5, eps=1e-5)


def _build_graph_model(feat_dim, hidden_dim, exfeat_dim, edge_dim, dropout):
    """Construct shared sub-modules used by both GraphBepi variants."""
    bias = False
    W_v = nn.Linear(feat_dim, hidden_dim, bias=bias)
    W_u1 = AE(exfeat_dim, hidden_dim, hidden_dim, bias=bias)
    edge_linear = nn.Sequential(nn.Linear(edge_dim, hidden_dim // 4, bias=True), nn.ELU())
    gat = EGAT(2 * hidden_dim, hidden_dim, hidden_dim // 4, dropout)
    lstm1 = nn.LSTM(hidden_dim, hidden_dim // 2, 3, batch_first=True, bidirectional=True, dropout=dropout)
    lstm2 = nn.LSTM(hidden_dim, hidden_dim // 2, 3, batch_first=True, bidirectional=True, dropout=dropout)
    mlp = nn.Sequential(
        nn.Linear(4 * hidden_dim, hidden_dim, bias=True), nn.ReLU(),
        nn.Linear(hidden_dim, 1, bias=True), nn.Sigmoid()
    )
    return W_v, W_u1, edge_linear, gat, lstm1, lstm2, mlp


class GraphBepi(BaseLightningModel):
    def __init__(self, feat_dim=2560, hidden_dim=256, exfeat_dim=13, edge_dim=51,
                 augment_eps=0.05, dropout=0.2, lr=1e-6, metrics=None, result_path=None,
                 loss_fn=None):
        super().__init__()
        self.metrics = metrics
        self.path = result_path
        self.loss_fn = loss_fn if loss_fn is not None else nn.BCELoss()
        self.exfeat_dim = exfeat_dim
        self.augment_eps = augment_eps
        self.lr = lr
        self.cls = 1
        (self.W_v, self.W_u1, self.edge_linear,
         self.gat, self.lstm1, self.lstm2, self.mlp) = _build_graph_model(
            feat_dim, hidden_dim, exfeat_dim, edge_dim, dropout)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, V, edge):
        V = pad_sequence(V, batch_first=True, padding_value=0).float()
        mask = V.sum(-1) != 0
        if self.training and self.augment_eps > 0:
            aug = torch.randn_like(V)
            aug[~mask] = 0
            V = V + self.augment_eps * aug
        mask = mask.sum(1)
        feats = self.W_v(V[:, :, :-self.exfeat_dim])
        exfeats = self.W_u1(V[:, :, -self.exfeat_dim:])
        x_gcns = []
        for i in range(len(V)):
            E = self.edge_linear(edge[i]).permute(2, 0, 1)
            x_gcn, _ = self.gat(torch.cat([feats[i, :mask[i]], exfeats[i, :mask[i]]], -1), E)
            x_gcns.append(x_gcn)
        feats = pad_packed_sequence(
            self.lstm1(pack_padded_sequence(feats, mask.cpu(), True, False))[0], True)[0]
        exfeats = pad_packed_sequence(
            self.lstm2(pack_padded_sequence(exfeats, mask.cpu(), True, False))[0], True)[0]
        x_attns = torch.cat([feats, exfeats], -1)
        h = torch.cat([torch.cat([x_attns[i, :mask[i]], x_gcns[i]], -1) for i in range(len(x_gcns))], 0)
        return self.mlp(h)

    def _forward_batch(self, batch):
        feat, edge, y = batch
        return self(feat, edge).squeeze(-1), y


class GraphBepi_att(BaseLightningModel):
    def __init__(self, feat_dim=2581, hidden_dim=256, exfeat_dim=13, edge_dim=51,
                 augment_eps=0.05, dropout=0.2, lr=1e-6, metrics=None, result_path=None,
                 loss_fn=None):
        super().__init__()
        self.metrics = metrics
        self.path = result_path
        self.loss_fn = loss_fn if loss_fn is not None else nn.BCELoss()
        self.exfeat_dim = exfeat_dim
        self.augment_eps = augment_eps
        self.lr = lr
        self.cls = 1
        (self.W_v, self.W_u1, self.edge_linear,
         self.gat, self.lstm1, self.lstm2, self.mlp) = _build_graph_model(
            feat_dim, hidden_dim, exfeat_dim, edge_dim, dropout)
        self.attn = ParametricSelfAttention()
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, fold_token, V, edge):
        fold_token = self.attn(pad_sequence(fold_token, batch_first=True, padding_value=0).float())
        V = pad_sequence(V, batch_first=True, padding_value=0).float()
        V = torch.cat([V, fold_token], dim=-1)
        mask = V.sum(-1) != 0
        if self.training and self.augment_eps > 0:
            aug = torch.randn_like(V)
            aug[~mask] = 0
            V = V + self.augment_eps * aug
        mask = mask.sum(1)
        feats = self.W_v(V[:, :, :-self.exfeat_dim])
        exfeats = self.W_u1(V[:, :, -self.exfeat_dim:])
        x_gcns = []
        for i in range(len(V)):
            E = self.edge_linear(edge[i]).permute(2, 0, 1)
            x_gcn, _ = self.gat(torch.cat([feats[i, :mask[i]], exfeats[i, :mask[i]]], -1), E)
            x_gcns.append(x_gcn)
        feats = pad_packed_sequence(
            self.lstm1(pack_padded_sequence(feats, mask.cpu(), True, False))[0], True)[0]
        exfeats = pad_packed_sequence(
            self.lstm2(pack_padded_sequence(exfeats, mask.cpu(), True, False))[0], True)[0]
        x_attns = torch.cat([feats, exfeats], -1)
        h = torch.cat([torch.cat([x_attns[i, :mask[i]], x_gcns[i]], -1) for i in range(len(x_gcns))], 0)
        return self.mlp(h)

    def _forward_batch(self, batch):
        fold_token, feat, edge, y = batch
        return self(fold_token, feat, edge).squeeze(-1), y
