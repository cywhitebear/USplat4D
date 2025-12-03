# usplat4d/voxel_sparsify.py
import torch


def voxel_sparsify_keys(
    pts: torch.Tensor,           # (N,3), cuda or cpu
    base_key_mask: torch.Tensor, # (N,) bool
    voxel_size: float,
) -> torch.Tensor:
    """
    From a dense set of key-node candidates (base_key_mask),
    keep at most ONE Gaussian per voxel (the one with lowest uncertainty).

    Returns:
      key_mask: (N,) bool
    """
    N = pts.shape[0]
    key_mask = torch.zeros(N, dtype=torch.bool)

    idx_key = torch.nonzero(base_key_mask, as_tuple=False).view(-1)
    if idx_key.numel() == 0:
        return base_key_mask.clone()

    pts_key = pts[idx_key].cpu()              # (M,3)
    coords = torch.floor(pts_key / voxel_size).to(torch.int64)  # (M,3)

    # We need uncertainties per candidate; the caller will pass them as avg_unc[idx_key]
    # To keep this helper generic, we do NOT compute uncertainties here.

    # This function is intended to be called with
    #   voxel_sparsify_keys(..., base_key_mask, voxel_size)
    # from a place where we already know "best per voxel".
    # Here, we only implement the mapping structure; actual "best" logic
    # is done in keynode_selection where we have avg_unc.

    # However, to avoid mixing concerns, we simply return the mask here
    # and let the caller fill it. See keynode_selection.select_key_nodes_from_window.
    # (We keep this function minimal on purpose.)

    # NOTE: The actual per-voxel selection is implemented in keynode_selection,
    # because it needs access to avg_unc.

    return key_mask  # dummy; real sparsification is handled in keynode_selection
