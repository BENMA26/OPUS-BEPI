"""
DPO fine-tuning of GraphBepi with spatial-coherence preference.

Pipeline
--------
1. Load a pre-trained GraphBepi checkpoint as the *frozen* reference model.
2. Initialise the *trainable* policy model from the same checkpoint.
3. For every batch:
     a. One forward pass through both models (ref is no-grad).
     b. Construct y_l (spatially incoherent labels) on-the-fly per protein.
     c. Compute DPO loss + task loss (BCE on GT).
     d. Back-prop through the policy model only.

Usage example
-------------
python train_dpo.py \\
    --ref_ckpt ./model/BCE_633_GraphBepi/model_-1.ckpt \\
    --dataset  BCE_633 --tag GraphBepi_dpo \\
    --gpu 0 --fold -1 --epochs 50 \\
    --beta 0.1 --lambda_task 1.0
"""
import os
import copy
import warnings
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from tool import METRICS
from model import GraphBepi
from dataset import PDB_DPO, collate_fn_dpo
from dpo import compute_dpo_loss
from train_utils import seed_everything, build_arg_parser
warnings.simplefilter('ignore')


# ── DPO Lightning module ──────────────────────────────────────────────────────

class GraphBepiDPO(pl.LightningModule):
    """
    Wraps a policy GraphBepi and a frozen reference GraphBepi.
    Optimises:  L = L_DPO + λ · L_task
    """

    def __init__(self, policy: GraphBepi, ref: GraphBepi,
                 beta: float = 0.1, lambda_task: float = 1.0,
                 metrics=None, result_path: str = None):
        super().__init__()
        self.policy = policy
        self.ref = ref
        self.beta = beta
        self.lambda_task = lambda_task
        self.metrics = metrics
        self.path = result_path
        self.loss_fn = torch.nn.BCELoss()

        # Freeze reference model permanently
        for p in self.ref.parameters():
            p.requires_grad_(False)

    # ── Forward helpers ───────────────────────────────────────────────────────

    def _predict(self, model, feats, edges):
        """Run GraphBepi forward; returns (N_total,) probabilities."""
        return model(feats, edges).squeeze(-1)

    # ── Training ──────────────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        feats, edges, labels, coords_list, rsa_list = batch

        # Policy forward (with grad)
        pred_policy = self._predict(self.policy, feats, edges)

        # Reference forward (no grad)
        with torch.no_grad():
            pred_ref = self._predict(self.ref, feats, edges)

        # Split labels per protein for per-protein DPO
        lengths = [len(c) for c in coords_list]
        labels_list = list(labels.split(lengths))

        # DPO loss (spatial preference)
        l_dpo = compute_dpo_loss(
            pred_policy, pred_ref,
            labels_list, coords_list, rsa_list,
            beta=self.beta,
        )

        # Task loss (keep predictive accuracy)
        l_task = self.loss_fn(pred_policy, labels.float())

        loss = l_dpo + self.lambda_task * l_task

        self.log('train_loss',      loss.item(),   on_epoch=True, prog_bar=True)
        self.log('train_dpo_loss',  l_dpo.item(),  on_epoch=True, prog_bar=True)
        self.log('train_task_loss', l_task.item(), on_epoch=True, prog_bar=True)
        return loss

    # ── Validation ────────────────────────────────────────────────────────────

    def on_validation_epoch_start(self):
        self.val_outputs = []

    def validation_step(self, batch, batch_idx):
        feats, edges, labels, coords_list, rsa_list = batch
        pred = self._predict(self.policy, feats, edges)
        self.val_outputs.append((pred.detach(), labels.detach()))
        return pred, labels

    def on_validation_epoch_end(self):
        if not self.val_outputs:
            return
        pred = torch.cat([o[0] for o in self.val_outputs])
        y    = torch.cat([o[1] for o in self.val_outputs])
        self.log('val_loss',  self.loss_fn(pred, y.float()).item(), on_epoch=True, prog_bar=True)
        if self.metrics is not None:
            result = self.metrics(pred, y)
            self.log('val_AUPRC', result['AUPRC'], on_epoch=True, prog_bar=True)
            self.log('val_AUROC', result['AUROC'], on_epoch=True, prog_bar=True)

    # ── Test ──────────────────────────────────────────────────────────────────

    def on_test_epoch_start(self):
        self.test_outputs = []

    def test_step(self, batch, batch_idx):
        feats, edges, labels, coords_list, rsa_list = batch
        pred = self._predict(self.policy, feats, edges)
        self.test_outputs.append((pred.detach(), labels.detach()))

    def on_test_epoch_end(self):
        if not self.test_outputs:
            return
        pred = torch.cat([o[0] for o in self.test_outputs])
        y    = torch.cat([o[1] for o in self.test_outputs])
        self.log('test_loss', self.loss_fn(pred, y.float()).item(), on_epoch=True, prog_bar=True)
        if self.path:
            os.makedirs(self.path, exist_ok=True)
            torch.save({'pred': pred.cpu(), 'gt': y.cpu()}, f'{self.path}/result.pkl')
        if self.metrics is not None:
            result = self.metrics(pred, y)
            for k in ['AUROC', 'AUPRC', 'RECALL', 'PRECISION', 'F1', 'MCC', 'BACC']:
                self.log(f'test_{k}', result[k], on_epoch=True, prog_bar=True)
            self.log('test_threshold', result['threshold'], on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        # Only optimise policy parameters
        return torch.optim.Adam(self.policy.parameters(),
                                lr=self.policy.lr, betas=(0.9, 0.99),
                                weight_decay=1e-5, eps=1e-5)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = build_arg_parser()
    parser.add_argument('--ref_ckpt',     type=str, required=True,
                        help='path to pre-trained GraphBepi checkpoint (reference model).')
    parser.add_argument('--beta',         type=float, default=0.1,
                        help='DPO KL penalty β.')
    parser.add_argument('--lambda_task',  type=float, default=1.0,
                        help='weight for task (BCE) loss.')
    parser.add_argument('--feat_dim',     type=int, default=2560,
                        help='input feature dim matching the checkpoint.')
    args = parser.parse_args()

    seed_everything(args.seed)
    device   = 'cpu' if args.gpu == -1 else f'cuda:{args.gpu}'
    root     = f'./data/{args.dataset}'
    log_name = f'{args.dataset}_{args.tag}'

    # ── Datasets ──────────────────────────────────────────────────────────────
    trainset = PDB_DPO(mode='train', fold=args.fold, root=root)
    valset   = PDB_DPO(mode='val',   fold=args.fold, root=root)
    testset  = PDB_DPO(mode='test',  fold=args.fold, root=root)

    train_loader = DataLoader(trainset, batch_size=args.batch, shuffle=True,
                              collate_fn=collate_fn_dpo, drop_last=True)
    val_loader   = DataLoader(valset, batch_size=args.batch, shuffle=False,
                              collate_fn=collate_fn_dpo)
    test_loader  = DataLoader(testset, batch_size=args.batch, shuffle=False,
                              collate_fn=collate_fn_dpo)
    if args.fold == -1:
        val_loader = test_loader

    # ── Models ────────────────────────────────────────────────────────────────
    model_kwargs = dict(
        feat_dim=args.feat_dim, hidden_dim=args.hidden,
        exfeat_dim=13, edge_dim=51,
        augment_eps=0.05, dropout=0.2, lr=args.lr,
    )
    # Reference model: load checkpoint, freeze
    ref_model = GraphBepi(**model_kwargs)
    ref_model.load_state_dict(
        torch.load(args.ref_ckpt, map_location='cpu')['state_dict']
    )

    # Policy model: initialised from same checkpoint
    policy_model = GraphBepi(**model_kwargs, metrics=METRICS(device),
                             result_path=f'./model/{log_name}')
    policy_model.load_state_dict(
        torch.load(args.ref_ckpt, map_location='cpu')['state_dict']
    )

    dpo_model = GraphBepiDPO(
        policy=policy_model, ref=ref_model,
        beta=args.beta, lambda_task=args.lambda_task,
        metrics=METRICS(device), result_path=f'./model/{log_name}',
    )

    # ── Training ──────────────────────────────────────────────────────────────
    accelerator = 'gpu' if args.gpu != -1 else 'cpu'
    mc = ModelCheckpoint(
        f'./model/{log_name}/', f'model_dpo_{args.fold}',
        'val_AUPRC', mode='max', save_weights_only=True,
    )
    logger = TensorBoardLogger(args.logger, name=f'{log_name}_dpo_{args.fold}')

    trainer = pl.Trainer(
        accelerator=accelerator, max_epochs=args.epochs,
        callbacks=[mc], logger=logger, check_val_every_n_epoch=1,
    )

    ckpt_path = f'./model/{log_name}/model_dpo_{args.fold}.ckpt'
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    trainer.fit(dpo_model, train_loader, val_loader)

    # Reload best checkpoint for test
    best_state = torch.load(ckpt_path)['state_dict']
    # state_dict keys are prefixed with "policy." inside GraphBepiDPO
    policy_state = {k.removeprefix('policy.'): v
                    for k, v in best_state.items() if k.startswith('policy.')}
    policy_model.load_state_dict(policy_state)

    trainer = pl.Trainer(accelerator=accelerator, logger=logger)
    trainer.test(dpo_model, test_loader)
    os.rename(f'./model/{log_name}/result.pkl',
              f'./model/{log_name}/result_dpo_{args.fold}.pkl')
