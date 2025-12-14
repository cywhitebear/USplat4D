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
        return torch.tensor(0.0, device='cuda')
    
    # Current and initial positions
    p_curr = params['means3D'][key_indices]  # (M_k, 3)
    p_init = init_params['means3D'][key_indices]  # (M_k, 3)
    
    # Position deviation
    deviation = p_curr - p_init  # (M_k, 3)
    
    # Uncertainty weighting: U^-1 down-weights high-uncertainty directions
    # Simplified: use scalar uncertainty as isotropic weighting
    if len(state.uncertainty_window) > 0:
        # Get uncertainty for key nodes at current window position
        window_idx = min(t - 1, len(state.uncertainty_window) - 1)
        if window_idx >= 0:
            unc = state.uncertainty_window[window_idx].cuda()  # (N,)
            unc_key = unc[key_indices.cpu()].cuda()  # (M_k,)
            # Inverse uncertainty weighting (lower unc = higher weight)
            weights = 1.0 / (unc_key + 1e-6)  # (M_k,)
        else:
            weights = torch.ones(M_k, device='cuda')
    else:
        weights = torch.ones(M_k, device='cuda')
    
    # Weighted L2 loss
    position_loss = (weights.unsqueeze(-1) * deviation ** 2).sum() / M_k
    
    # L_motion,key: isometry, rigidity, rotation constraints
    # For now, return just position loss (TODO: add motion constraints)
    motion_loss = torch.tensor(0.0, device='cuda')
    
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
        return torch.tensor(0.0, device='cuda')
    
    # Current and initial positions
    p_curr = params['means3D'][non_key_indices]  # (M_n, 3)
    p_init = init_params['means3D'][non_key_indices]  # (M_n, 3)
    
    # Deviation from initialization
    deviation_init = p_curr - p_init  # (M_n, 3)
    
    # DQB interpolation from key nodes
    # For now, use simple assignment (TODO: implement full DQB)
    key_indices = graph_dict['key_indices'].cuda()
    assignments = graph_dict['non_key_assignments'].cuda()  # (M_n,) local indices
    key_transforms = graph_dict['key_transforms'].cuda()  # (M_k, 4, 4)
    
    # Get assigned key node transforms
    assigned_transforms = key_transforms[assignments]  # (M_n, 4, 4)
    
    # Apply transforms to get DQB positions
    p_dqb = torch.zeros_like(p_curr)
    for i in range(M_n):
        T = assigned_transforms[i]
        p_old = p_init[i]
        p_dqb[i] = T[:3, :3] @ p_old + T[:3, 3]
    
    deviation_dqb = p_curr - p_dqb  # (M_n, 3)
    
    # Uncertainty weighting
    if len(state.uncertainty_window) > 0:
        window_idx = min(t - 1, len(state.uncertainty_window) - 1)
        if window_idx >= 0:
            unc = state.uncertainty_window[window_idx].cuda()
            unc_nonkey = unc[non_key_indices.cpu()].cuda()
            weights = 1.0 / (unc_nonkey + 1e-6)
        else:
            weights = torch.ones(M_n, device='cuda')
    else:
        weights = torch.ones(M_n, device='cuda')
    
    # Weighted losses
    init_loss = (weights.unsqueeze(-1) * deviation_init ** 2).sum() / M_n
    dqb_loss = (weights.unsqueeze(-1) * deviation_dqb ** 2).sum() / M_n
    
    # Motion loss (placeholder)
    motion_loss = torch.tensor(0.0, device='cuda')
    
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
        return torch.tensor(0.0, device='cuda')
    
    # Get key node positions
    positions = params['means3D'][key_indices]  # (M_k, 3)
    
    # Compute motion smoothness over edges
    loss = torch.tensor(0.0, device='cuda')
    
    for i in range(M_k):
        p_i = positions[i]  # (3,)
        neighbors = key_key_edges[i]  # (k,) local indices
        weights = key_key_weights[i]  # (k,)
        
        for j_local, w in zip(neighbors, weights):
            if j_local < M_k:  # Valid neighbor
                p_j = positions[j_local]
                # Penalize position difference
                loss += w * ((p_i - p_j) ** 2).sum()
    
    return loss / (M_k + 1e-6)
