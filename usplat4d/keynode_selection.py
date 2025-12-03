# usplat4d/keynode_selection.py
import torch
from .state import TemporalState
from .uncertainty_window import compute_significant_period_stats


def select_key_nodes_from_window(
    t: int,
    params: dict,
    state: TemporalState,
    T_min: int = 5,
    quantile: float = 0.02,
    voxel_ratio: float = 0.05,
):
    """
    Implements:
      - Thresholding by significant period (USplat4D, page 5)
      - Selecting lowest-uncertainty 2% among valid Gaussians
      - Spatial sparsification by voxel (keep 1 per voxel)

    Side effects:
      - Appends (N,) bool mask to state.key_gaussians
      - Prints a one-line summary for debugging
    """
    avg_unc, valid_mask, has_valid, W = compute_significant_period_stats(
        state, T_min=T_min
    )

    if W < T_min:
        print(f"[key-selection] t={t}, window={W} < T_min={T_min}, skip key-node selection")
        return

    if not has_valid:
        print(f"[key-selection] t={t}, no Gaussians with significant period >= {T_min}")
        return

    # average uncertainty is well-defined only where valid_mask == True
    tau = torch.quantile(avg_unc[valid_mask], quantile)
    base_key_mask = (avg_unc <= tau) & valid_mask              # (N,) bool

    pts = params["means3D"].detach()
    N = pts.shape[0]

    # set voxel size once, based on scene radius
    if state.voxel_size is None:
        state.voxel_size = state.scene_radius * voxel_ratio
    voxel_size = state.voxel_size

    idx_key = torch.nonzero(base_key_mask, as_tuple=False).view(-1)
    if idx_key.numel() == 0:
        key_mask = base_key_mask
        state.key_gaussians.append(key_mask.cpu())
        print(
            f"[key-selection] t={t}, "
            f"valid={valid_mask.sum().item()}, "
            f"base_selected={base_key_mask.sum().item()}, "
            f"voxel_selected=0, "
            f"tau={tau.item():.6f}"
        )
        return

    pts_key = pts[idx_key].cpu()              # (M,3)
    avg_unc_key = avg_unc[idx_key].cpu()      # (M,)

    coords = torch.floor(pts_key / voxel_size).to(torch.int64)  # (M,3)
    coords_np = coords.numpy()
    avg_unc_np = avg_unc_key.numpy()
    idx_key_np = idx_key.cpu().numpy()

    # choose one Gaussian per voxel: lowest uncertainty in that voxel
    voxel_best = {}  # (vx,vy,vz) -> (best_unc, best_idx)
    for v, u_val, g_idx in zip(coords_np, avg_unc_np, idx_key_np):
        key_tuple = (int(v[0]), int(v[1]), int(v[2]))
        if key_tuple not in voxel_best or u_val < voxel_best[key_tuple][0]:
            voxel_best[key_tuple] = (u_val, g_idx)

    key_mask = torch.zeros(N, dtype=torch.bool)
    for _, (_, g_idx) in voxel_best.items():
        key_mask[g_idx] = True

    state.key_gaussians.append(key_mask.cpu())

    print(
        f"[key-selection] t={t}, "
        f"valid={valid_mask.sum().item()}, "
        f"base_selected={base_key_mask.sum().item()}, "
        f"voxel_selected={key_mask.sum().item()}, "
        f"tau={tau.item():.6f}"
    )
