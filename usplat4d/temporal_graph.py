# usplat4d/temporal_graph.py
"""
Temporal graph construction for USplat4D (§4.2(b)).

Builds uncertainty-aware adjacency between key nodes and non-key nodes,
encoding spatial affinity and motion similarity.
"""

import torch
import numpy as np
from typing import Tuple, Dict, List, Optional
from .state import TemporalState
from .uncertainty import (
    transform_to_anisotropic,
    compute_mahalanobis_distance,
    invert_covariance_safe,
)


def build_temporal_graph(
    params: dict,
    state: TemporalState,
    output_params: list,
    t: int,
    num_knn: int = 5,
    camera_rotation: Optional[torch.Tensor] = None,
    is_monocular: Optional[bool] = None,
) -> None:
    """
    Construct temporal graph encoding edges between key and non-key nodes.
    
    Implements §4.2(b) edge construction:
    - Key-key edges: UA-kNN on uncertainty-weighted Mahalanobis distance
    - Non-key edges: assignment to closest key node across entire sequence
    
    Args:
        params: Gaussian parameters dict with 'means3D', 'unnorm_rotations', etc.
        state: TemporalState with key_gaussians, uncertainty_window
        output_params: List of saved params per timestep (for historical positions)
        t: Current timestep
        num_knn: Number of neighbors for key-node kNN
        camera_rotation: [3, 3] camera-to-world rotation for anisotropic uncertainty (Eq. 6)
                        If None, uses identity (isotropic)
        is_monocular: If True, uses (1,1,5) for depth scaling; if False, uses (1,1,1)
                     If None, auto-detects from number of cameras in training
    
    Side effects:
        Appends to state.temporal_graph (list of dicts per timestep):
        - 'key_key_edges': (M_k, k) int tensor of neighbor indices
        - 'key_key_weights': (M_k, k) float tensor of edge weights
        - 'non_key_edges': (M_n, k) int tensor of k-nearest key neighbors for DQB
        - 'non_key_weights': (M_n, k) float tensor normalized per row for DQB blending
        - 'non_key_assignments': (M_n,) int tensor of closest key-node (backward compat)
        - 'key_transforms': (M_k, 4, 4) SE(3) transforms (identity at t=0)
    
    Raises:
        ValueError: If no key nodes exist or inconsistent state
    """
    
    if not hasattr(state, 'temporal_graph'):
        state.temporal_graph = []
    
    means3D = params['means3D'].detach()  # (N, 3)
    N = means3D.shape[0]
    
    # Retrieve key mask for current timestep
    if len(state.key_gaussians) == 0:
        print(f"[temporal_graph] t={t}, no key nodes selected yet, skip")
        return
    
    key_mask = state.key_gaussians[-1]  # (N,) bool
    if isinstance(key_mask, torch.Tensor) and key_mask.device.type != 'cuda':
        key_mask = key_mask.cuda()
    
    key_indices = torch.where(key_mask)[0]  # (M_k,)
    non_key_indices = torch.where(~key_mask)[0]  # (M_n,)
    
    M_k = key_indices.shape[0]
    M_n = non_key_indices.shape[0]
    
    if M_k == 0:
        print(f"[temporal_graph] t={t}, no key nodes, skip")
        return
    
    # ============================================================
    # 1. Build key-key edges (UA-kNN, Eq. 7)
    # ============================================================
    
    # Auto-detect monocular if not specified (stored in state or inferred)
    if is_monocular is None:
        is_monocular = getattr(state, 'is_monocular', True)  # Default to monocular
    
    # Use identity camera rotation if not provided
    if camera_rotation is None:
        camera_rotation = torch.eye(3, device='cuda', dtype=torch.float32)
    
    key_key_edges, key_key_weights = _build_key_key_edges(
        means3D=means3D,
        uncertainty_window=state.uncertainty_window,
        key_indices=key_indices,
        num_knn=num_knn,
        output_params=output_params,
        camera_rotation=camera_rotation,
        is_monocular=is_monocular,
    )
    
    # ============================================================
    # 2. Build non-key edges (Eq. 8, 10 - k-NN for DQB)
    # ============================================================
    
    non_key_edges, non_key_weights, non_key_assignments = _assign_nonkey_to_key(
        means3D=means3D,
        uncertainty_window=state.uncertainty_window,
        non_key_indices=non_key_indices,
        key_indices=key_indices,
        output_params=output_params,
        num_knn=num_knn,  # Use same k as key-key edges for consistency
    )
    
    # ============================================================
    # 3. Initialize SE(3) transforms (identity for key nodes)
    # ============================================================
    
    # Make transforms optimizable parameters (keep on GPU)
    key_transforms = torch.eye(4, device='cuda', dtype=torch.float32).unsqueeze(0).repeat(M_k, 1, 1)
    key_transforms = torch.nn.Parameter(key_transforms.clone(), requires_grad=True)
    
    # ============================================================
    # Store in state
    # ============================================================
    
    graph_dict = {
        'timestep': t,
        'key_indices': key_indices,  # Keep on GPU for optimization
        'non_key_indices': non_key_indices,
        'key_key_edges': key_key_edges,
        'key_key_weights': key_key_weights,
        'non_key_edges': non_key_edges,  # (M_n, k) k-NN edges for DQB
        'non_key_weights': non_key_weights,  # (M_n, k) normalized weights for DQB
        'non_key_assignments': non_key_assignments,  # (M_n,) closest key (backward compat)
        'key_transforms': key_transforms,  # Optimizable parameter
    }
    
    state.temporal_graph.append(graph_dict)
    
    print(
        f"[temporal_graph] t={t}, "
        f"key_nodes={M_k}, non_key_nodes={M_n}, "
        f"key_edges={key_key_edges.shape[0]}×{key_key_edges.shape[1]}, "
        f"non-key_edges={non_key_edges.shape[0]}×{non_key_edges.shape[1]} (k-NN for DQB)"
    )


def _build_key_key_edges(
    means3D: torch.Tensor,
    uncertainty_window: List[torch.Tensor],
    key_indices: torch.Tensor,
    num_knn: int = 5,
    output_params: list = None,
    camera_rotation: Optional[torch.Tensor] = None,
    is_monocular: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build key-key edges using Uncertainty-Aware kNN (Eq. 7).
    
    For each key node i, find k nearest key neighbors at its most reliable frame.
    Distance is Mahalanobis-weighted by anisotropic uncertainty (Eq. 6).
    
    Args:
        means3D: (N, 3) Gaussian positions (current frame)
        uncertainty_window: list of (N,) uncertainty tensors per timestep
        key_indices: (M_k,) indices of key nodes
        num_knn: Number of neighbors (k in paper)
        output_params: list of saved params per timestep (for historical positions)
        camera_rotation: [3, 3] camera-to-world rotation for Eq. 6
        is_monocular: If True, uses (1,1,5) depth scaling; else (1,1,1)
    
    Returns:
        key_key_edges: (M_k, k) indices of neighbors (local to key_indices)
        key_key_weights: (M_k, k) normalized edge weights
    """
    
    M_k = key_indices.shape[0]
    num_knn = min(num_knn, M_k - 1)  # Cannot have more neighbors than key nodes - 1
    
    if num_knn <= 0:
        print("[_build_key_key_edges] Not enough key nodes for kNN")
        return torch.zeros((M_k, 1), dtype=torch.long, device='cuda'), \
               torch.ones((M_k, 1), dtype=torch.float32, device='cuda')
    
    # ---- Find most reliable frame t_hat for each key node ----
    # t_hat = arg min_t { u_i,t }
    
    t_hat = torch.zeros(M_k, dtype=torch.long, device='cuda')
    
    if len(uncertainty_window) > 0:
        unc_stack = torch.stack(uncertainty_window, dim=1)  # (N, T_window) on CPU
        for i, g_idx in enumerate(key_indices):
            g_idx_cpu = g_idx.item() if g_idx.is_cuda else g_idx
            if g_idx_cpu < unc_stack.shape[0]:
                t_hat[i] = torch.argmin(unc_stack[g_idx_cpu])
            else:
                t_hat[i] = 0
    
    # ---- Get positions at most reliable frames ----
    # Extract from output_params if available (contains historical positions)
    # output_params[0] = t=0 (all params), output_params[1:] = t=1+ (means3D only)
    
    if output_params and len(output_params) >= 2:
        # Use last available timestep positions from output_params
        # (Could use t_hat per node, but simplified for now)
        last_idx = len(output_params) - 1
        pos_at_t = torch.tensor(output_params[last_idx]['means3D']).cuda()  # (N, 3)
        pts_key = pos_at_t[key_indices]  # (M_k, 3)
    else:
        # Fallback to current frame
        pts_key = means3D[key_indices]  # (M_k, 3)
    
    # ---- Compute pairwise Mahalanobis distances (Eq. 7) ----
    # Transform scalar uncertainty to anisotropic covariance (Eq. 6)
    
    if len(uncertainty_window) > 0 and camera_rotation is not None:
        unc_stack = torch.stack(uncertainty_window, dim=1)  # (N, T_window) on CPU
        # Mean uncertainty for each key node
        unc_key = unc_stack[key_indices.cpu()].mean(dim=1).cuda()  # (M_k,)
        
        # Transform to anisotropic covariance matrices (Eq. 6)
        cov_key = transform_to_anisotropic(
            uncertainty=unc_key,
            positions=pts_key,
            camera_rotation=camera_rotation,
            is_monocular=is_monocular,
        )  # (M_k, 3, 3)
        
        # Invert covariance for Mahalanobis distance
        cov_inv_key = invert_covariance_safe(cov_key)  # (M_k, 3, 3)
        
        # Compute Mahalanobis distances: d²(i,j) = (x_i - x_j)^T U_i^{-1} (x_i - x_j)
        dists = compute_mahalanobis_distance(pts_key, pts_key, cov_inv_key)  # (M_k, M_k)
        
    else:
        # Fallback: simple L2 distance if no uncertainty available
        dists = torch.cdist(pts_key, pts_key, p=2) ** 2  # (M_k, M_k), squared for consistency
    
    # Mask out self-distances
    dists.fill_diagonal_(float('inf'))
    
    # kNN: select k smallest neighbors per node
    neighbor_dists, neighbor_local_indices = torch.topk(dists, k=num_knn, dim=1, largest=False)
    
    # Convert distances to weights (exponential decay)
    neighbor_weights = torch.exp(-neighbor_dists.clamp(min=1e-8))
    
    # Normalize weights per node
    neighbor_weights = neighbor_weights / (neighbor_weights.sum(dim=1, keepdim=True) + 1e-8)
    
    return neighbor_local_indices.long(), neighbor_weights.float()


def _assign_nonkey_to_key(
    means3D: torch.Tensor,
    uncertainty_window: List[torch.Tensor],
    non_key_indices: torch.Tensor,
    key_indices: torch.Tensor,
    output_params: list = None,
    num_knn: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Assign each non-key node to k-nearest key nodes for DQB interpolation (Eq. 8, 10).
    
    For each non-key node i, find k closest key nodes across entire sequence:
        Neighbors = kNN_{l in Vk} sum_{t=0}^{T-1} ||p_i,t - p_l,t||_{U_weighted}
    
    Args:
        means3D: (N, 3) Gaussian positions (current frame)
        uncertainty_window: list of (N,) uncertainty tensors per timestep
        non_key_indices: (M_n,) indices of non-key nodes
        key_indices: (M_k,) indices of key nodes
        output_params: list of saved params per timestep (for historical positions)
        num_knn: Number of nearest key neighbors for DQB blending
    
    Returns:
        edges: (M_n, k) local indices into key_indices of k-nearest neighbors
        weights: (M_n, k) normalized weights for DQB blending (sum to 1 per row)
        assignments: (M_n,) local index of closest key node (for backward compatibility)
    """
    
    M_n = non_key_indices.shape[0]
    M_k = key_indices.shape[0]
    
    if M_n == 0 or M_k == 0:
        empty_edges = torch.zeros((M_n, num_knn), dtype=torch.long, device='cuda')
        empty_weights = torch.zeros((M_n, num_knn), dtype=torch.float32, device='cuda')
        empty_assignments = torch.zeros((M_n,), dtype=torch.long, device='cuda')
        return empty_edges, empty_weights, empty_assignments
    
    # Clamp k to available key nodes
    k = min(num_knn, M_k)
    
    # ---- Temporal accumulation of distances (Eq. 8) ----
    # sum_{t=0}^{T-1} ||p_i,t - p_l,t||
    
    if output_params and len(output_params) >= 2:
        # Full temporal accumulation across all saved timesteps
        # output_params[0] = t=0, output_params[1:] = t=1+
        total_dists = None
        
        for idx, p in enumerate(output_params[1:], start=1):  # Skip t=0 (has all params)
            pos_t = torch.tensor(p['means3D']).cuda()  # (N, 3)
            pts_nonkey_t = pos_t[non_key_indices]  # (M_n, 3)
            pts_key_t = pos_t[key_indices]  # (M_k, 3)
            
            dists_t = torch.cdist(pts_nonkey_t, pts_key_t, p=2)  # (M_n, M_k)
            
            # Uncertainty weighting (if available in window)
            # Note: uncertainty_window may be shorter than output_params
            window_idx = idx - 1  # Adjust for window indexing
            if window_idx < len(uncertainty_window):
                unc_t = uncertainty_window[window_idx].cuda()  # (N,)
                unc_nonkey = unc_t[non_key_indices.cpu()].cuda().unsqueeze(1)  # (M_n, 1)
                unc_key = unc_t[key_indices.cpu()].cuda().unsqueeze(0)  # (1, M_k)
                unc_pair = unc_nonkey + unc_key  # (M_n, M_k)
                dists_t = dists_t * torch.sqrt(1.0 + unc_pair.clamp(min=0))
            
            if total_dists is None:
                total_dists = dists_t
            else:
                total_dists = total_dists + dists_t
        
        # Average distance across timesteps
        if total_dists is not None:
            dists = total_dists / (len(output_params) - 1)
        else:
            # Fallback if no historical data
            pts_nonkey = means3D[non_key_indices]
            pts_key = means3D[key_indices]
            dists = torch.cdist(pts_nonkey, pts_key, p=2)
    else:
        # Use current positions as fallback
        pts_nonkey = means3D[non_key_indices]  # (M_n, 3)
        pts_key = means3D[key_indices]  # (M_k, 3)
        dists = torch.cdist(pts_nonkey, pts_key, p=2)  # (M_n, M_k)
    
    # Find k-nearest key nodes per non-key node
    topk_dists, topk_indices = torch.topk(dists, k, dim=1, largest=False, sorted=True)  # (M_n, k)
    
    # Convert distances to weights (exponential decay, normalized per row)
    weights = torch.exp(-topk_dists.clamp(min=1e-8))  # (M_n, k)
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)  # Normalize to sum=1
    
    # Pad to num_knn if k < num_knn
    if k < num_knn:
        pad_size = num_knn - k
        edges = torch.cat([topk_indices, topk_indices[:, :1].repeat(1, pad_size)], dim=1)
        weights = torch.cat([weights, torch.zeros(M_n, pad_size, device='cuda')], dim=1)
    else:
        edges = topk_indices
    
    # Closest key node (for backward compatibility)
    assignments = topk_indices[:, 0]  # (M_n,) - first neighbor
    
    return edges.long(), weights.float(), assignments.long()


def get_temporal_graph_at_t(state: TemporalState, t: int) -> Dict:
    """
    Retrieve temporal graph dict for timestep t.
    
    Args:
        state: TemporalState
        t: Timestep index
    
    Returns:
        Dict with keys: key_indices, non_key_indices, key_key_edges,
                        key_key_weights, non_key_assignments, non_key_weights,
                        key_transforms
        Returns None if no graph at this timestep
    """
    
    if not hasattr(state, 'temporal_graph') or len(state.temporal_graph) == 0:
        return None
    
    for graph_dict in state.temporal_graph:
        if graph_dict['timestep'] == t:
            return graph_dict
    
    return None
