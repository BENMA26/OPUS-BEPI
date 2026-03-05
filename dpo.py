"""
DPO (Direct Preference Optimization) for epitope spatial coherence.

Key insight:
  GraphBepi is an encoder-only classifier: the forward pass produces per-residue
  probabilities p_i independently of labels. Thus the sequence log-probability
  factors as:

      log π(y | x) = Σ_i [ y_i·log p_i + (1-y_i)·log(1-p_i) ]  (sum over N residues)

  This lets us compute log π(y_w|x) and log π(y_l|x) from a SINGLE forward
  pass, without needing two separate forward passes as in autoregressive DPO.

DPO loss (Bradley–Terry preference model):

  L_DPO = -E[ log σ( β · [(log π_θ(y_w|x) - log π_ref(y_w|x))
                          - (log π_θ(y_l|x) - log π_ref(y_l|x))] ) ]

  where:
    π_θ   = policy model (trainable)
    π_ref = reference model (frozen, pre-trained GraphBepi)
    y_w   = spatially coherent labels (ground truth)
    y_l   = spatially incoherent labels (scrambled; same # of positives)
    β     = KL penalty weight (controls deviation from reference)

Total training objective:
  L = L_DPO + λ · L_task
where L_task = BCE(p_θ, y_w) anchors the model to the original task.
"""
import torch
import torch.nn.functional as F
from spatial_utils import make_incoherent_labels


# ── Log-probability helper ────────────────────────────────────────────────────

def sequence_log_prob(pred: torch.Tensor,
                      labels: torch.Tensor,
                      normalize: bool = True) -> torch.Tensor:
    """
    Compute log π(y | x) for one protein chain.

    Args:
        pred:      (N,) predicted probabilities in (0, 1).
        labels:    (N,) binary labels.
        normalize: if True, divide by N (makes β comparable across lengths).

    Returns:
        Scalar log-probability.
    """
    pred = pred.clamp(1e-7, 1 - 1e-7)
    log_p = labels.float() * torch.log(pred) + (1 - labels.float()) * torch.log(1 - pred)
    return log_p.mean() if normalize else log_p.sum()


# ── DPO loss ──────────────────────────────────────────────────────────────────

def dpo_loss_single(log_pi_w: torch.Tensor,
                    log_pi_ref_w: torch.Tensor,
                    log_pi_l: torch.Tensor,
                    log_pi_ref_l: torch.Tensor,
                    beta: float = 0.1) -> torch.Tensor:
    """
    DPO loss for a single (x, y_w, y_l) triplet.

    reward_margin = (log π_θ(y_w|x) - log π_ref(y_w|x))
                  - (log π_θ(y_l|x) - log π_ref(y_l|x))

    L = -log σ(β · reward_margin)
    """
    reward_w = log_pi_w - log_pi_ref_w
    reward_l = log_pi_l - log_pi_ref_l
    return -F.logsigmoid(beta * (reward_w - reward_l))


def compute_dpo_loss(pred_policy: torch.Tensor,
                     pred_ref: torch.Tensor,
                     labels_list: list,
                     coords_list: list,
                     rsa_list: list,
                     beta: float = 0.1) -> torch.Tensor:
    """
    Compute mean DPO loss across a batch of proteins.

    Args:
        pred_policy:  (N_total,) policy predictions (requires_grad=True).
        pred_ref:     (N_total,) reference predictions (no_grad).
        labels_list:  list of (N_i,) ground-truth tensors (y_w per protein).
        coords_list:  list of (N_i, 3) Cα coordinate tensors.
        rsa_list:     list of (N_i,) RSA masks.
        beta:         DPO KL penalty.

    Returns:
        Scalar DPO loss.
    """
    lengths = [len(y) for y in labels_list]
    pred_p_list = pred_policy.split(lengths)
    pred_r_list = pred_ref.split(lengths)

    losses = []
    for pred_p, pred_r, y_w, coords, rsa in zip(
        pred_p_list, pred_r_list, labels_list, coords_list, rsa_list
    ):
        # Construct spatially incoherent y_l on-the-fly
        y_l = make_incoherent_labels(y_w, rsa, coords)

        log_p_w = sequence_log_prob(pred_p, y_w)
        log_p_l = sequence_log_prob(pred_p, y_l)
        with torch.no_grad():
            log_r_w = sequence_log_prob(pred_r, y_w)
            log_r_l = sequence_log_prob(pred_r, y_l)

        losses.append(dpo_loss_single(log_p_w, log_r_w, log_p_l, log_r_l, beta))

    return torch.stack(losses).mean()
