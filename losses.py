"""
Custom loss functions for B-cell epitope prediction.

All losses accept:
    pred : Tensor, shape (N,), sigmoid output in [0, 1]
    y    : Tensor, shape (N,), binary labels {0, 1}

Usage (via build_loss_fn in train_utils.py):
    --loss bce          BCELoss (baseline)
    --loss afl          AsymmetricFocalLoss
    --loss bce_dice     BCEDiceLoss (BCE + Dice regularizer)
    --loss smooth_bce   LabelSmoothingBCELoss (positive-side smoothing)
    --loss pu           PULoss (non-negative Positive-Unlabeled learning)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1. Asymmetric Focal Loss (AFL)
# ─────────────────────────────────────────────────────────────────────────────
class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Focal Loss (Ridnik et al., ICCV 2021).

    For positives : L+ = -(1 - p)^gamma_pos * log(p)
    For negatives : L- = -(p_m)^gamma_neg * log(1 - p_m)
                    where p_m = max(p - clip, 0)  (probability shift)

    gamma_pos = 0  -> standard CE for positives (no down-weighting of easy positives)
    gamma_neg = 2  -> strong down-weighting of easy negatives
    clip      = 0.05 -> discard very low-confidence negatives
    """

    def __init__(self, gamma_pos: float = 0.0, gamma_neg: float = 2.0, clip: float = 0.05):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip

    def forward(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pred = pred.clamp(1e-7, 1.0 - 1e-7)

        # Positive loss
        pos_loss = -(1.0 - pred) ** self.gamma_pos * torch.log(pred)

        # Negative loss with probability shift
        pred_neg = (pred - self.clip).clamp(min=0.0)
        neg_loss = -(pred_neg ** self.gamma_neg) * torch.log(1.0 - pred_neg + 1e-7)

        loss = y * pos_loss + (1.0 - y) * neg_loss
        return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# 2. BCE + Dice Loss
# ─────────────────────────────────────────────────────────────────────────────
class BCEDiceLoss(nn.Module):
    """
    Weighted combination: alpha * BCE + (1 - alpha) * Dice.

    Dice component:
        Dice = 1 - (2 * sum(p * y) + smooth) / (sum(p) + sum(y) + smooth)

    alpha = 1.0 -> pure BCE (same as baseline)
    alpha = 0.5 -> equal weighting (default)
    alpha = 0.0 -> pure Dice

    Dice is computed per-sample (per protein chain) then averaged,
    so it measures patch-level overlap rather than residue-level accuracy.
    """

    def __init__(self, alpha: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth
        self.bce = nn.BCELoss()

    def forward(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(pred, y)

        intersection = (pred * y).sum()
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (
            pred.sum() + y.sum() + self.smooth
        )

        return self.alpha * bce_loss + (1.0 - self.alpha) * dice_loss


# ─────────────────────────────────────────────────────────────────────────────
# 3. Label-Smoothing BCE (positive-side only)
# ─────────────────────────────────────────────────────────────────────────────
class LabelSmoothingBCELoss(nn.Module):
    """
    BCE with one-sided label smoothing applied only to positive labels.

        y_smooth_i = y_i * (1 - eps)   for positives
        y_smooth_i = y_i               for negatives (unchanged)

    Motivation: epitope annotations from PDB complexes are incomplete
    (only residues within 4-5 Å of the antibody are marked positive).
    Smoothing prevents the model from being over-confident on noisy labels
    while leaving the negative class supervision unchanged.

    eps = 0.05 -> mild smoothing (default)
    """

    def __init__(self, eps: float = 0.05):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_smooth = y * (1.0 - self.eps)
        pred = pred.clamp(1e-7, 1.0 - 1e-7)
        loss = -(y_smooth * torch.log(pred) + (1.0 - y) * torch.log(1.0 - pred))
        return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Non-negative PU Loss (nnPU)
# ─────────────────────────────────────────────────────────────────────────────
class PULoss(nn.Module):
    """
    Non-negative Positive-Unlabeled (nnPU) loss (Kiryo et al., NeurIPS 2017).

    Treats y=0 surface residues as *unlabeled* rather than confirmed negatives,
    because a residue not observed in any antibody complex is not proven to be
    a non-epitope.

    R_pu = prior * R_p+ + max(0, R_u- - prior * R_p-)

    where:
        R_p+  = E[l(f(x), +1) | x ~ P(x|y=1)]   positive risk on labeled positives
        R_p-  = E[l(f(x), -1) | x ~ P(x|y=1)]   negative risk on labeled positives
        R_u-  = E[l(f(x), -1) | x ~ P(x|y=0)]   negative risk on unlabeled samples
        prior = P(y=1)  (fraction of epitope residues, ~0.15 by default)

    The max(0, ...) ensures the unlabeled risk never goes below zero,
    preventing the model from collapsing to always predicting positive.
    """

    def __init__(self, prior: float = 0.15):
        super().__init__()
        if not (0.0 < prior < 1.0):
            raise ValueError(f"prior must be in (0, 1), got {prior}")
        self.prior = prior

    def _bce(self, pred: torch.Tensor, target: float) -> torch.Tensor:
        pred = pred.clamp(1e-7, 1.0 - 1e-7)
        if target == 1.0:
            return -torch.log(pred)
        else:
            return -torch.log(1.0 - pred)

    def forward(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pos_mask = y == 1
        unl_mask = y == 0

        if pos_mask.sum() == 0:
            # No positive samples in batch — fall back to standard BCE
            return F.binary_cross_entropy(pred, y)

        pos_pred = pred[pos_mask]
        unl_pred = pred[unl_mask] if unl_mask.sum() > 0 else pred.new_zeros(1)

        r_p_pos = self._bce(pos_pred, 1.0).mean()   # positive risk on positives
        r_p_neg = self._bce(pos_pred, 0.0).mean()   # negative risk on positives
        r_u_neg = self._bce(unl_pred, 0.0).mean()   # negative risk on unlabeled

        unlabeled_risk = r_u_neg - self.prior * r_p_neg

        # Non-negative correction
        if unlabeled_risk < 0:
            pu_loss = -unlabeled_risk
        else:
            pu_loss = self.prior * r_p_pos + unlabeled_risk

        return pu_loss
