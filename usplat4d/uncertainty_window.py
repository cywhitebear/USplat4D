# usplat4d/uncertainty_window.py
import torch
from .state import TemporalState


def update_uncertainty_window(
    state: TemporalState,
    u_mean: torch.Tensor,
    seen_any: torch.Tensor,
    window_size: int,
):
    """
    Store per-timestep mean uncertainty, and maintain a sliding window
    of the last `window_size` timesteps for significant-period stats.
    """
    # keep full history for debugging / saving
    state.temporal_uncertainty.append(u_mean.cpu())

    # sliding window (CPU to save GPU memory)
    state.uncertainty_window.append(u_mean.cpu())
    state.visibility_window.append(seen_any.cpu())

    if len(state.uncertainty_window) > window_size:
        state.uncertainty_window.pop(0)
        state.visibility_window.pop(0)


def compute_significant_period_stats(
    state: TemporalState,
    T_min: int,
):
    """
    Implements 'Thresholding by significant period' (USplat4D, page 5).
    Returns:
      avg_unc      : (N,) tensor, inf for invalid Gaussians
      valid_mask   : (N,) bool, True where significant period >= T_min
      has_valid    : bool, whether any valid Gaussians exist
      W            : int, current window length
    """
    W = len(state.uncertainty_window)
    if W < T_min:
        return None, None, False, W

    # shape: (W, N)
    u_win = torch.stack(state.uncertainty_window, dim=0)        # float
    v_win = torch.stack(state.visibility_window, dim=0).float() # 0/1

    # number of visible frames in the window
    sig_counts = v_win.sum(dim=0)                               # (N,)
    valid_mask = sig_counts >= T_min                            # (N,) bool

    if not valid_mask.any():
        return None, valid_mask, False, W

    weighted_sum = (u_win * v_win).sum(dim=0)                   # (N,)
    avg_unc = torch.full_like(weighted_sum, float("inf"))       # (N,)
    avg_unc[valid_mask] = weighted_sum[valid_mask] / sig_counts[valid_mask]

    return avg_unc, valid_mask, True, W
