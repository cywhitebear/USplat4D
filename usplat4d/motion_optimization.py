# usplat4d/motion_optimization.py
"""
Motion optimization for USplat4D (§4.3).

Implements SE(3) transform optimization for key nodes and 
Dual Quaternion Blending (DQB) interpolation for non-key nodes.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple
from .state import TemporalState


def dual_quaternion_blending(
    positions: torch.Tensor,
    rotations: torch.Tensor,
    transforms: torch.Tensor,
    weights: torch.Tensor,
    neighbor_indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Dual Quaternion Blending (DQB) for smooth motion interpolation (Eq. 10).
    
    Blends SE(3) transforms from multiple key nodes to interpolate motion.
    
    Args:
        positions: (N,3) original positions of nodes to interpolate
        rotations: (N,4) original quaternions of nodes to interpolate
        transforms: (M_k,4,4) SE(3) transforms of key nodes
        weights: (N,k) normalized blending weights for each neighbor
        neighbor_indices: (N,k) indices of key neighbors (local to key nodes)
    
    Returns:
        blended_positions: (N,3) interpolated positions
        blended_rotations: (N,4) interpolated quaternions
    
    Reference: Kavan et al. 2007, "Skinning with Dual Quaternions"
    """
    
    N = positions.shape[0]
    k = neighbor_indices.shape[1]
    
    # Extract rotation and translation from SE(3) transforms
    # T = [R t; 0 1] where R is 3x3 rotation, t is 3x1 translation
    
    # Get transforms for each node's neighbors
    # neighbor_indices: (N, k) -> gather transforms: (N, k, 4, 4)
    neighbor_transforms = transforms[neighbor_indices]  # (N, k, 4, 4)
    
    # Extract rotations (top-left 3x3) and translations (top-right 3x1)
    R_neighbors = neighbor_transforms[:, :, :3, :3]  # (N, k, 3, 3)
    t_neighbors = neighbor_transforms[:, :, :3, 3]   # (N, k, 3)
    
    # Convert rotation matrices to quaternions
    # Simplified: Use weighted average (proper DQB requires dual quaternion math)
    # For now, implement simplified weighted blending
    
    # Weighted position blending
    blended_positions = (weights.unsqueeze(-1) * t_neighbors).sum(dim=1)  # (N, 3)
    
    # Weighted rotation blending (simplified - proper DQB needs dual quat)
    # Convert R to quaternions, blend, convert back
    # For simplicity: just use first neighbor's rotation (TODO: proper DQB)
    # This is a placeholder - full DQB implementation needed
    blended_rotations = rotations.clone()  # (N, 4)
    
    return blended_positions, blended_rotations


def apply_se3_transform_to_gaussians(
    positions: torch.Tensor,
    rotations: torch.Tensor,
    transforms: torch.Tensor,
    indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply SE(3) transforms to Gaussian positions and rotations.
    
    Args:
        positions: (N,3) original positions
        rotations: (N,4) original quaternions
        transforms: (M,4,4) SE(3) transforms for subset of Gaussians
        indices: (M,) global indices of Gaussians to transform
    
    Returns:
        new_positions: (N,3) transformed positions
        new_rotations: (N,4) transformed quaternions
    """
    
    new_positions = positions.clone()
    new_rotations = rotations.clone()
    
    # Apply transforms only to specified indices
    for i, idx in enumerate(indices):
        T = transforms[i]  # (4,4)
        R = T[:3, :3]      # (3,3)
        t = T[:3, 3]       # (3,)
        
        # Transform position
        p_old = positions[idx]  # (3,)
        p_new = R @ p_old + t
        new_positions[idx] = p_new
        
        # Transform rotation (quaternion)
        # q_new = R * q_old (rotation composition)
        # Simplified: just update with R (TODO: proper quaternion composition)
        new_rotations[idx] = rotations[idx]  # Placeholder
    
    return new_positions, new_rotations


def compute_key_node_loss(
    params: dict,
    state: TemporalState,
    graph_dict: Dict,
    init_params: dict,
    t: int,
) -> torch.Tensor:
    """
    Compute key node loss (Eq. 9).
    
    L_key = Σ ||p_i,t - p^o_i,t||_{U^-1} + L_motion,key
    
    Args:
        params: Current Gaussian parameters
        state: TemporalState with uncertainty info
        graph_dict: Temporal graph for current timestep
        init_params: Initial/pretrained parameters (p^o)
        t: Current timestep
    
    Returns:
        key_loss: Scalar loss for key nodes
    """
    
    key_indices = graph_dict['key_indices'].cuda()  # (M_k,)
    M_k = key_indices.shape[0]
    
    if M_k == 0:
        return torch.tensor(0.0, device='cuda', requires_grad=True)
    
    # Current and initial positions
    p_curr = params['means3D'][key_indices]  # (M_k, 3)
    
    # Handle case where new Gaussians were added (not in init_params)
    N_init = init_params['means3D'].shape[0]
    key_indices_cpu = key_indices.cpu()
    valid_init_mask = key_indices_cpu < N_init
    
    if not valid_init_mask.any():
        # All key nodes are newly added, skip this loss
        return torch.tensor(0.0, device='cuda', requires_grad=True)
    
    # Only compute loss for Gaussians that existed at initialization
    valid_key_indices = key_indices[valid_init_mask]
    p_curr_valid = p_curr[valid_init_mask]
    p_init = init_params['means3D'][valid_key_indices]  # (M_k_valid, 3)
    
    # Position deviation
    deviation = p_curr_valid - p_init  # (M_k_valid, 3)
    
    # Uncertainty weighting: U^-1 down-weights high-uncertainty directions
    # Simplified: use scalar uncertainty as isotropic weighting
    if len(state.uncertainty_window) > 0:
        # Get uncertainty for key nodes at current window position
        window_idx = min(t - 1, len(state.uncertainty_window) - 1)
        if window_idx >= 0:
            unc = state.uncertainty_window[window_idx].cuda()  # (N_old,)
            # Check if valid indices are within uncertainty bounds
            valid_key_indices_cpu = valid_key_indices.cpu()
            valid_unc_mask = valid_key_indices_cpu < unc.shape[0]
            if valid_unc_mask.any():
                valid_unc_indices = valid_key_indices_cpu[valid_unc_mask]
                unc_key = torch.ones(valid_key_indices.shape[0], device='cuda')
                unc_key[valid_unc_mask] = unc[valid_unc_indices].cuda()
                weights = 1.0 / (unc_key + 1e-6)
            else:
                weights = torch.ones(valid_key_indices.shape[0], device='cuda')
        else:
            weights = torch.ones(valid_key_indices.shape[0], device='cuda')
    else:
        weights = torch.ones(valid_key_indices.shape[0], device='cuda')
    
    # Weighted L2 loss
    M_k_valid = valid_key_indices.shape[0]
    position_loss = (weights.unsqueeze(-1) * deviation ** 2).sum() / M_k_valid
    
    # L_motion,key: isometry, rigidity, rotation constraints
    # For now, return just position loss (TODO: add motion constraints)
    motion_loss = torch.zeros(1, device='cuda', requires_grad=True)[0]
    
    return position_loss + motion_loss


def compute_non_key_loss(
    params: dict,
    state: TemporalState,
    graph_dict: Dict,
    init_params: dict,
    t: int,
) -> torch.Tensor:
    """
    Compute non-key node loss (Eq. 11).
    
    L_non-key = Σ ||p_i,t - p^o_i,t||_{U^-1} + Σ ||p_i,t - p^DQB_i,t||_{U^-1} + L_motion,non-key
    
    Args:
        params: Current Gaussian parameters
        state: TemporalState
        graph_dict: Temporal graph for current timestep
        init_params: Initial/pretrained parameters
        t: Current timestep
    
    Returns:
        non_key_loss: Scalar loss for non-key nodes
    """
    
    non_key_indices = graph_dict['non_key_indices'].cuda()  # (M_n,)
    M_n = non_key_indices.shape[0]
    
    if M_n == 0:
        return torch.tensor(0.0, device='cuda', requires_grad=True)
    
    # Current and initial positions
    p_curr = params['means3D'][non_key_indices]  # (M_n, 3)
    
    # Handle new Gaussians not in init_params
    N_init = init_params['means3D'].shape[0]
    nonkey_indices_cpu = non_key_indices.cpu()
    valid_init_mask = nonkey_indices_cpu < N_init
    
    # Deviation from initialization (only for original Gaussians)
    if valid_init_mask.any():
        valid_nonkey_for_init = non_key_indices[valid_init_mask]
        p_init = init_params['means3D'][valid_nonkey_for_init]
        deviation_init = p_curr[valid_init_mask] - p_init
    else:
        deviation_init = torch.zeros_like(p_curr[:1])  # Dummy for gradient
    
    # DQB interpolation from key nodes
    # For now, use simple assignment (TODO: implement full DQB)
    key_indices = graph_dict['key_indices'].cuda()
    assignments = graph_dict['non_key_assignments'].cuda()  # (M_n,) local indices
    key_transforms = graph_dict['key_transforms'].cuda()  # (M_k, 4, 4)
    M_k = key_indices.shape[0]
    
    # Clamp assignments to valid range
    assignments = torch.clamp(assignments, 0, M_k - 1)
    
    # Get assigned key node transforms
    assigned_transforms = key_transforms[assignments]  # (M_n, 4, 4)
    
    # Apply transforms to current positions (vectorized)
    # T @ p + t for each: [R|t] @ [p; 1]
    R = assigned_transforms[:, :3, :3]  # (M_n, 3, 3)
    trans = assigned_transforms[:, :3, 3]   # (M_n, 3) - translation vectors
    # Batch matrix multiply: (M_n, 3, 3) @ (M_n, 3, 1) -> (M_n, 3, 1)
    p_dqb = torch.bmm(R, p_curr.unsqueeze(-1)).squeeze(-1) + trans  # (M_n, 3)
    
    deviation_dqb = p_curr - p_dqb  # (M_n, 3)
    
    # Uncertainty weighting
    if len(state.uncertainty_window) > 0:
        window_idx = min(t - 1, len(state.uncertainty_window) - 1)
        if window_idx >= 0:
            unc = state.uncertainty_window[window_idx].cuda()
            # Check bounds
            nonkey_indices_cpu = non_key_indices.cpu()
            valid_mask = nonkey_indices_cpu < unc.shape[0]
            if valid_mask.any():
                valid_indices = nonkey_indices_cpu[valid_mask]
                unc_nonkey = torch.ones(M_n, device='cuda')
                unc_nonkey[valid_mask] = unc[valid_indices].cuda()
                weights = 1.0 / (unc_nonkey + 1e-6)
            else:
                weights = torch.ones(M_n, device='cuda')
        else:
            weights = torch.ones(M_n, device='cuda')
    else:
        weights = torch.ones(M_n, device='cuda')
    
    # Weighted losses
    if valid_init_mask.any():
        M_n_valid = valid_init_mask.sum().item()
        init_loss = (weights[valid_init_mask].unsqueeze(-1) * deviation_init ** 2).sum() / M_n_valid
    else:
        init_loss = torch.zeros(1, device='cuda', requires_grad=True)[0]
    
    dqb_loss = (weights.unsqueeze(-1) * deviation_dqb ** 2).sum() / M_n
    
    # Motion loss (placeholder)
    motion_loss = torch.zeros(1, device='cuda', requires_grad=True)[0]
    
    return init_loss + dqb_loss + motion_loss


def compute_motion_regularization_loss(
    params: dict,
    state: TemporalState,
    graph_dict: Dict,
) -> torch.Tensor:
    """
    Compute motion regularization over temporal graph edges.
    
    Encourages smooth motion between connected key nodes.
    
    Args:
        params: Current Gaussian parameters
        state: TemporalState
        graph_dict: Temporal graph with key-key edges
    
    Returns:
        regularization_loss: Scalar loss
    """
    
    key_indices = graph_dict['key_indices'].cuda()
    key_key_edges = graph_dict['key_key_edges'].cuda()  # (M_k, k) local indices
    key_key_weights = graph_dict['key_key_weights'].cuda()  # (M_k, k)
    
    M_k = key_indices.shape[0]
    if M_k == 0:
        return torch.tensor(0.0, device='cuda', requires_grad=True)
    
    # Get key node positions
    positions = params['means3D'][key_indices]  # (M_k, 3)
    
    # Compute motion smoothness over edges (vectorized)
    # For each key node, compute weighted distance to its k neighbors
    # positions[i] - positions[edges[i, j]] weighted by weights[i, j]
    
    # Gather neighbor positions: (M_k, k, 3)
    k = key_key_edges.shape[1]
    valid_edges = torch.clamp(key_key_edges, 0, M_k - 1)  # Clamp to valid range
    neighbor_positions = positions[valid_edges]  # (M_k, k, 3)
    
    # Compute differences: (M_k, k, 3)
    pos_diff = positions.unsqueeze(1) - neighbor_positions  # (M_k, 1, 3) - (M_k, k, 3)
    
    # Weighted squared differences: (M_k, k)
    weighted_sq_diff = key_key_weights * (pos_diff ** 2).sum(dim=-1)
    
    # Sum over all edges
    loss = weighted_sq_diff.sum() / (M_k + 1e-6)
    
    return loss
