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


def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to quaternion (w, x, y, z).
    
    Args:
        R: (..., 3, 3) rotation matrices
    
    Returns:
        q: (..., 4) quaternions with w as first component
    """
    batch_shape = R.shape[:-2]
    R_flat = R.reshape(-1, 3, 3)
    N = R_flat.shape[0]
    
    q = torch.zeros(N, 4, device=R.device, dtype=R.dtype)
    
    trace = R_flat[:, 0, 0] + R_flat[:, 1, 1] + R_flat[:, 2, 2]
    
    # Case 1: trace > 0
    mask1 = trace > 0
    s = torch.sqrt(trace[mask1] + 1.0) * 2
    q[mask1, 0] = 0.25 * s  # w
    q[mask1, 1] = (R_flat[mask1, 2, 1] - R_flat[mask1, 1, 2]) / s  # x
    q[mask1, 2] = (R_flat[mask1, 0, 2] - R_flat[mask1, 2, 0]) / s  # y
    q[mask1, 3] = (R_flat[mask1, 1, 0] - R_flat[mask1, 0, 1]) / s  # z
    
    # Case 2: R[0,0] is largest diagonal
    mask2 = (~mask1) & (R_flat[:, 0, 0] > R_flat[:, 1, 1]) & (R_flat[:, 0, 0] > R_flat[:, 2, 2])
    s = torch.sqrt(1.0 + R_flat[mask2, 0, 0] - R_flat[mask2, 1, 1] - R_flat[mask2, 2, 2]) * 2
    q[mask2, 0] = (R_flat[mask2, 2, 1] - R_flat[mask2, 1, 2]) / s
    q[mask2, 1] = 0.25 * s
    q[mask2, 2] = (R_flat[mask2, 0, 1] + R_flat[mask2, 1, 0]) / s
    q[mask2, 3] = (R_flat[mask2, 0, 2] + R_flat[mask2, 2, 0]) / s
    
    # Case 3: R[1,1] is largest diagonal
    mask3 = (~mask1) & (~mask2) & (R_flat[:, 1, 1] > R_flat[:, 2, 2])
    s = torch.sqrt(1.0 + R_flat[mask3, 1, 1] - R_flat[mask3, 0, 0] - R_flat[mask3, 2, 2]) * 2
    q[mask3, 0] = (R_flat[mask3, 0, 2] - R_flat[mask3, 2, 0]) / s
    q[mask3, 1] = (R_flat[mask3, 0, 1] + R_flat[mask3, 1, 0]) / s
    q[mask3, 2] = 0.25 * s
    q[mask3, 3] = (R_flat[mask3, 1, 2] + R_flat[mask3, 2, 1]) / s
    
    # Case 4: R[2,2] is largest diagonal
    mask4 = (~mask1) & (~mask2) & (~mask3)
    s = torch.sqrt(1.0 + R_flat[mask4, 2, 2] - R_flat[mask4, 0, 0] - R_flat[mask4, 1, 1]) * 2
    q[mask4, 0] = (R_flat[mask4, 1, 0] - R_flat[mask4, 0, 1]) / s
    q[mask4, 1] = (R_flat[mask4, 0, 2] + R_flat[mask4, 2, 0]) / s
    q[mask4, 2] = (R_flat[mask4, 1, 2] + R_flat[mask4, 2, 1]) / s
    q[mask4, 3] = 0.25 * s
    
    # Normalize and reshape
    q = F.normalize(q, dim=-1)
    return q.reshape(*batch_shape, 4)


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion (w, x, y, z) to rotation matrix.
    
    Args:
        q: (..., 4) quaternions with w as first component
    
    Returns:
        R: (..., 3, 3) rotation matrices
    """
    batch_shape = q.shape[:-1]
    q = q.reshape(-1, 4)
    q = F.normalize(q, dim=-1)
    
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    R = torch.stack([
        torch.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w], dim=-1),
        torch.stack([2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w], dim=-1),
        torch.stack([2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y], dim=-1),
    ], dim=-2)
    
    return R.reshape(*batch_shape, 3, 3)


def se3_to_dual_quaternion(T: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert SE(3) transform to dual quaternion representation.
    
    Dual quaternion: q̂ = qᵣ + ε qₜ
    where qᵣ is rotation quaternion, qₜ = 0.5 * [0, t] * qᵣ
    
    Args:
        T: (..., 4, 4) SE(3) transforms
    
    Returns:
        q_real: (..., 4) real part (rotation quaternion)
        q_dual: (..., 4) dual part (translation quaternion)
    """
    R = T[..., :3, :3]  # Rotation
    t = T[..., :3, 3]   # Translation
    
    # Rotation to quaternion
    q_real = rotation_matrix_to_quaternion(R)
    
    # Translation quaternion: qₜ = 0.5 * [0, t] * qᵣ
    # where [0, t] represents quaternion (0, tx, ty, tz)
    t_quat = torch.cat([torch.zeros_like(t[..., :1]), t], dim=-1)  # (0, tx, ty, tz)
    
    # Quaternion multiplication: t_quat * q_real
    w_t, x_t, y_t, z_t = t_quat[..., 0], t_quat[..., 1], t_quat[..., 2], t_quat[..., 3]
    w_r, x_r, y_r, z_r = q_real[..., 0], q_real[..., 1], q_real[..., 2], q_real[..., 3]
    
    # (w_t, x_t, y_t, z_t) * (w_r, x_r, y_r, z_r)
    # = (w_t*w_r - x_t*x_r - y_t*y_r - z_t*z_r,
    #    w_t*x_r + x_t*w_r + y_t*z_r - z_t*y_r,
    #    w_t*y_r - x_t*z_r + y_t*w_r + z_t*x_r,
    #    w_t*z_r + x_t*y_r - y_t*x_r + z_t*w_r)
    
    q_dual = torch.stack([
        w_t * w_r - x_t * x_r - y_t * y_r - z_t * z_r,
        w_t * x_r + x_t * w_r + y_t * z_r - z_t * y_r,
        w_t * y_r - x_t * z_r + y_t * w_r + z_t * x_r,
        w_t * z_r + x_t * y_r - y_t * x_r + z_t * w_r,
    ], dim=-1) * 0.5
    
    return q_real, q_dual


def dual_quaternion_to_se3(q_real: torch.Tensor, q_dual: torch.Tensor) -> torch.Tensor:
    """
    Convert dual quaternion back to SE(3) transform.
    
    Args:
        q_real: (..., 4) real part (rotation quaternion)
        q_dual: (..., 4) dual part (translation quaternion)
    
    Returns:
        T: (..., 4, 4) SE(3) transforms
    """
    # Normalize rotation quaternion
    q_real = F.normalize(q_real, dim=-1)
    
    # Convert rotation quaternion to matrix
    R = quaternion_to_rotation_matrix(q_real)
    
    # Extract translation: t = 2 * qₜ * qᵣ*
    # For dual quaternions: qₜ = 0.5 * t_quat * qᵣ, so t = 2 * qₜ * qᵣ*
    # Simplifies to: t = R(qᵣ) @ t_original (the original translation vector)
    # Actually, we need: t = 2 * qₜ * conj(qᵣ), extract vector part
    w_r, x_r, y_r, z_r = q_real[..., 0], q_real[..., 1], q_real[..., 2], q_real[..., 3]
    w_d, x_d, y_d, z_d = q_dual[..., 0], q_dual[..., 1], q_dual[..., 2], q_dual[..., 3]
    
    # Quaternion multiplication: qₜ * conj(qᵣ) where conj = (w, -x, -y, -z)
    # Result = (w1*w2 - v1·v2, w1*v2 + w2*v1 + v1×v2)
    # where v1 = (x_d, y_d, z_d), v2_conj = (-x_r, -y_r, -z_r)
    
    # Scalar part: w_d*w_r - (x_d*(-x_r) + y_d*(-y_r) + z_d*(-z_r)) = w_d*w_r + x_d*x_r + y_d*y_r + z_d*z_r
    # Vector part: w_d*v2_conj + w_r*v1 + v1 × v2_conj
    #   = w_d*(-x_r, -y_r, -z_r) + w_r*(x_d, y_d, z_d) + (x_d, y_d, z_d) × (-x_r, -y_r, -z_r)
    #   = (-w_d*x_r + w_r*x_d + (y_d*(-z_r) - z_d*(-y_r)),
    #      -w_d*y_r + w_r*y_d + (z_d*(-x_r) - x_d*(-z_r)),
    #      -w_d*z_r + w_r*z_d + (x_d*(-y_r) - y_d*(-x_r)))
    
    t_x = 2.0 * (-w_d * x_r + w_r * x_d - y_d * z_r + z_d * y_r)
    t_y = 2.0 * (-w_d * y_r + w_r * y_d - z_d * x_r + x_d * z_r)
    t_z = 2.0 * (-w_d * z_r + w_r * z_d - x_d * y_r + y_d * x_r)
    
    t = torch.stack([t_x, t_y, t_z], dim=-1)
    
    # Construct SE(3) matrix
    batch_shape = R.shape[:-2]
    T = torch.zeros(*batch_shape, 4, 4, device=R.device, dtype=R.dtype)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    
    return T


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
    Paper reference: USplat4D §4.3 Eq. 10
    
    Args:
        positions: (N,3) original positions of nodes to interpolate
        rotations: (N,4) original quaternions of nodes to interpolate
        transforms: (M_k,4,4) SE(3) transforms of key nodes
        weights: (N,k) normalized blending weights for each neighbor
        neighbor_indices: (N,k) indices of key neighbors (local to key nodes)
    
    Returns:
        blended_positions: (N,3) interpolated positions
        blended_rotations: (N,4) interpolated quaternions (w, x, y, z)
    
    Reference: Kavan et al. 2007, "Skinning with Dual Quaternions"
    """
    
    N = positions.shape[0]
    k = neighbor_indices.shape[1]
    
    # Get transforms for each node's k neighbors: (N, k, 4, 4)
    neighbor_transforms = transforms[neighbor_indices]
    
    # Convert each neighbor transform to dual quaternion
    # q_real: (N, k, 4), q_dual: (N, k, 4)
    q_real_neighbors, q_dual_neighbors = se3_to_dual_quaternion(neighbor_transforms)
    
    # Ensure consistent quaternion hemisphere (flip if dot product < 0 with first neighbor)
    # This prevents rotation interpolation going the long way around
    q_real_ref = q_real_neighbors[:, 0:1, :]  # (N, 1, 4) - first neighbor as reference
    dots = (q_real_neighbors * q_real_ref).sum(dim=-1, keepdim=True)  # (N, k, 1)
    flip_sign = (dots < 0).float() * -2.0 + 1.0  # -1 if dot < 0, else +1
    q_real_neighbors = q_real_neighbors * flip_sign  # (N, k, 4)
    q_dual_neighbors = q_dual_neighbors * flip_sign  # (N, k, 4)
    
    # Weighted blending of dual quaternions
    # q̂ = Σ wᵢⱼ * q̂ⱼ = (Σ wᵢⱼ * qᵣⱼ) + ε(Σ wᵢⱼ * qₜⱼ)
    weights_expanded = weights.unsqueeze(-1)  # (N, k, 1)
    q_real_blended = (weights_expanded * q_real_neighbors).sum(dim=1)  # (N, 4)
    q_dual_blended = (weights_expanded * q_dual_neighbors).sum(dim=1)  # (N, 4)
    
    # Normalize the blended dual quaternion
    # Real part must be unit quaternion
    norm = torch.norm(q_real_blended, dim=-1, keepdim=True)
    q_real_blended = q_real_blended / (norm + 1e-8)
    q_dual_blended = q_dual_blended / (norm + 1e-8)
    
    # Ensure orthogonality: q_real · q_dual = 0
    # Project out any component of q_dual parallel to q_real
    dot_product = (q_real_blended * q_dual_blended).sum(dim=-1, keepdim=True)
    q_dual_blended = q_dual_blended - dot_product * q_real_blended
    
    # Convert blended dual quaternion back to SE(3)
    T_blended = dual_quaternion_to_se3(q_real_blended, q_dual_blended)  # (N, 4, 4)
    
    # Apply blended transform to original positions
    # p_new = R_blended @ p_old + t_blended
    R_blended = T_blended[:, :3, :3]  # (N, 3, 3)
    t_blended = T_blended[:, :3, 3]   # (N, 3)
    
    blended_positions = torch.bmm(R_blended, positions.unsqueeze(-1)).squeeze(-1) + t_blended
    
    # Extract blended rotation as quaternion
    blended_rotations = q_real_blended  # (N, 4) already normalized
    
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
    
    # DQB interpolation from key nodes (USplat4D §4.3 Eq. 10)
    key_indices = graph_dict['key_indices'].cuda()
    key_transforms = graph_dict['key_transforms'].cuda()  # (M_k, 4, 4)
    non_key_edges = graph_dict['non_key_edges'].cuda()  # (M_n, k) local indices to key nodes
    non_key_weights = graph_dict['non_key_weights'].cuda()  # (M_n, k) normalized weights
    M_k = key_indices.shape[0]
    
    # Clamp edge indices to valid range
    non_key_edges_clamped = torch.clamp(non_key_edges, 0, M_k - 1)
    
    # Get current rotations for non-key nodes
    r_curr = torch.nn.functional.normalize(params['unnorm_rotations'][non_key_indices])
    
    # Apply Dual Quaternion Blending to interpolate motion from k neighbors
    p_dqb, q_dqb = dual_quaternion_blending(
        positions=p_curr,
        rotations=r_curr,
        transforms=key_transforms,
        weights=non_key_weights,
        neighbor_indices=non_key_edges_clamped,
    )
    
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
