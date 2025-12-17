"""
Apply motion to Gaussians using optimized SE(3) transforms.

This module applies the learned SE(3) transforms from temporal graph
to Gaussian positions and rotations.
"""

import torch
from typing import Dict


def apply_motion_to_gaussians(
    params: dict,
    graph_dict: Dict,
    in_place: bool = True,
) -> dict:
    """
    Apply SE(3) transforms from temporal graph to Gaussian parameters.
    
    - Key nodes: Apply their individual SE(3) transforms
    - Non-key nodes: Interpolate from assigned key nodes (simplified 1-to-1)
    
    Args:
        params: Gaussian parameters dict
        graph_dict: Temporal graph with key_transforms, assignments
        in_place: If True, modify params directly; else return copy
    
    Returns:
        params: Modified parameters (same dict if in_place=True)
    """
    
    if not in_place:
        params = {k: v.clone() for k, v in params.items()}
    
    key_indices = graph_dict['key_indices'].cuda()
    non_key_indices = graph_dict['non_key_indices'].cuda()
    key_transforms = graph_dict['key_transforms'].cuda()  # (M_k, 4, 4)
    non_key_assignments = graph_dict['non_key_assignments'].cuda()  # (M_n,)
    
    M_k = key_indices.shape[0]
    M_n = non_key_indices.shape[0]
    
    if M_k == 0:
        return params
    
    # 1) Apply transforms to key nodes
    for i in range(M_k):
        g_idx = key_indices[i]
        T = key_transforms[i]  # (4, 4)
        R = T[:3, :3]  # (3, 3)
        t = T[:3, 3]   # (3,)
        
        # Transform position: p' = R @ p + t
        p_old = params['means3D'][g_idx]
        params['means3D'][g_idx] = R @ p_old + t
        
        # Transform rotation: q' = R_to_quat(R @ quat_to_R(q))
        # Simplified: assume small rotations, skip for now
        # TODO: Proper quaternion composition
    
    # 2) Apply transforms to non-key nodes (via assignment)
    if M_n > 0:
        for i in range(M_n):
            g_idx = non_key_indices[i]
            assigned_key_local_idx = non_key_assignments[i]
            
            if assigned_key_local_idx < 0 or assigned_key_local_idx >= M_k:
                continue
            
            # Use the same transform as assigned key node
            T = key_transforms[assigned_key_local_idx]
            R = T[:3, :3]
            t = T[:3, 3]
            
            p_old = params['means3D'][g_idx]
            params['means3D'][g_idx] = R @ p_old + t
    
    return params


def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to quaternion (w, x, y, z).
    
    Args:
        R: (3, 3) or (N, 3, 3) rotation matrix
    
    Returns:
        q: (4,) or (N, 4) quaternion
    """
    if R.dim() == 2:
        R = R.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    
    # Shepperd's method for numerical stability
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    
    q = torch.zeros(R.shape[0], 4, device=R.device, dtype=R.dtype)
    
    # Case 1: trace > 0
    mask = trace > 0
    s = torch.sqrt(trace[mask] + 1.0) * 2
    q[mask, 0] = 0.25 * s
    q[mask, 1] = (R[mask, 2, 1] - R[mask, 1, 2]) / s
    q[mask, 2] = (R[mask, 0, 2] - R[mask, 2, 0]) / s
    q[mask, 3] = (R[mask, 1, 0] - R[mask, 0, 1]) / s
    
    # Case 2-4: trace <= 0 (TODO: handle other cases)
    # For simplicity, use case 1 formula with clamping
    mask = ~mask
    s = torch.sqrt(torch.clamp(trace[mask] + 1.0, min=1e-6)) * 2
    q[mask, 0] = 0.25 * s
    q[mask, 1] = (R[mask, 2, 1] - R[mask, 1, 2]) / (s + 1e-6)
    q[mask, 2] = (R[mask, 0, 2] - R[mask, 2, 0]) / (s + 1e-6)
    q[mask, 3] = (R[mask, 1, 0] - R[mask, 0, 1]) / (s + 1e-6)
    
    # Normalize
    q = q / (torch.norm(q, dim=1, keepdim=True) + 1e-8)
    
    if squeeze:
        q = q.squeeze(0)
    
    return q


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions: q1 * q2.
    
    Args:
        q1, q2: (4,) or (N, 4) quaternions (w, x, y, z)
    
    Returns:
        q: (4,) or (N, 4) quaternion product
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.stack([w, x, y, z], dim=-1)
