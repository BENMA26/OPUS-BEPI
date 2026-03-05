"""
Spatial coherence utilities for epitope DPO training.

Epitopes tend to form contiguous surface patches. These utilities:
  1. Measure how spatially coherent a predicted/labeled epitope set is.
  2. Construct incoherent "losing" label sequences (y_l) by scattering
     epitope labels across surface-exposed residues.
"""
import torch
import torch.nn.functional as F


# ── Spatial adjacency ─────────────────────────────────────────────────────────

def spatial_adjacency(coords: torch.Tensor, threshold: float = 8.0) -> torch.Tensor:
    """
    Return a (N, N) boolean matrix: True if Cα distance < threshold (Å).

    Args:
        coords:    (N, 3) Cα coordinates in Å.
        threshold: distance cutoff, default 8 Å (typical neighbour cutoff).
    """
    dists = torch.cdist(coords.float(), coords.float())   # (N, N)
    adj = dists < threshold
    adj.fill_diagonal_(False)                              # exclude self-loop
    return adj                                             # bool (N, N)


def spatial_coherence(labels: torch.Tensor,
                      coords: torch.Tensor,
                      threshold: float = 8.0) -> torch.Tensor:
    """
    Fraction of epitope–epitope pairs that are spatially adjacent.

    coherence = |{(i,j): i≠j, label_i=1, label_j=1, dist(i,j)<threshold}|
                / max(|epi_set| * (|epi_set|-1), 1)

    Returns a scalar in [0, 1]; higher = more spatially clustered.
    """
    epi_mask = labels.bool()
    n_epi = epi_mask.sum().item()
    if n_epi < 2:
        return torch.tensor(1.0)  # trivially coherent

    adj = spatial_adjacency(coords, threshold)             # (N, N)
    # Count adjacent epitope pairs
    epi_adj = adj[epi_mask][:, epi_mask]                  # (n_epi, n_epi)
    n_adj_pairs = epi_adj.sum().float()
    n_total_pairs = n_epi * (n_epi - 1)
    return n_adj_pairs / n_total_pairs


# ── Preference pair construction ──────────────────────────────────────────────

def make_incoherent_labels(labels: torch.Tensor,
                           rsa: torch.Tensor,
                           coords: torch.Tensor,
                           rsa_threshold: float = 0.15,
                           n_attempts: int = 5) -> torch.Tensor:
    """
    Construct y_l: same number of positives as y_w (GT), but placed on
    surface residues in a spatially dispersed manner.

    Strategy: randomly sample n_epi residues from surface-exposed residues.
    Among n_attempts random samples, return the one with lowest coherence.

    Args:
        labels:        (N,) ground-truth binary labels (y_w).
        rsa:           (N,) relative solvent accessibility mask (bool or float).
        coords:        (N, 3) Cα coordinates.
        rsa_threshold: residues with rsa > threshold are considered surface.
        n_attempts:    number of random draws; pick the least coherent one.

    Returns:
        y_l: (N,) binary tensor, same dtype as labels.
    """
    n_epi = int(labels.sum().item())
    if n_epi == 0:
        return labels.clone()

    # Surface-exposed candidate residues
    surface = (rsa.float() > rsa_threshold).nonzero(as_tuple=True)[0]
    if len(surface) < n_epi:
        surface = torch.arange(len(labels), device=labels.device)

    best_y_l, best_coherence = None, float('inf')
    for _ in range(n_attempts):
        perm = surface[torch.randperm(len(surface), device=labels.device)[:n_epi]]
        y_l = torch.zeros_like(labels)
        y_l[perm] = 1
        coh = spatial_coherence(y_l, coords).item()
        if coh < best_coherence:
            best_coherence = coh
            best_y_l = y_l

    return best_y_l
