# usplat4d/motion_interpolation.py
"""
Non-key motion interpolation for USplat4D (§4.2(c)).

After optimizing key nodes, propagate motion to non-key nodes via interpolation.
"""

import torch
from typing import Dict
from .state import TemporalState


def interpolate_nonkey_motion(
    params: dict,
    state: TemporalState,
    graph_dict: Dict,
) -> dict:
    """
    Interpolate non-key node motion from assigned key nodes (§4.2(c)).
    
    Implements simplified Dual Quaternion Blending (DQB):
    - Each non-key node inherits motion from its assigned key node
    - Future: Full DQB with weighted blending from k nearest key nodes
    
    Args:
        params: Gaussian parameters dict
        state: TemporalState
        graph_dict: Temporal graph with assignments
    
    Returns:
        updated_params: Parameters with interpolated non-key positions
    """
    
    if graph_dict is None:
        return params
    
    key_indices = graph_dict['key_indices'].cuda()
    non_key_indices = graph_dict['non_key_indices'].cuda()
    assignments = graph_dict['non_key_assignments'].cuda()  # (M_n,) local indices into key_indices
    key_transforms = graph_dict['key_transforms'].cuda()  # (M_k, 4, 4)
    
    M_n = non_key_indices.shape[0]
    if M_n == 0:
        return params
    
    # Get current positions
    positions = params['means3D'].detach().clone()
    
    # Apply key node transforms to their assigned non-key nodes
    for i in range(M_n):
        non_key_idx = non_key_indices[i]
        key_local_idx = assignments[i]
        
        # Get the transform from assigned key node
        T = key_transforms[key_local_idx]  # (4, 4)
        R = T[:3, :3]  # Rotation
        t = T[:3, 3]   # Translation
        
        # Apply transform: p_new = R * p_old + t
        p_old = positions[non_key_idx]
        p_new = R @ p_old + t
        
        # Update position
        positions[non_key_idx] = p_new
    
    # Create updated params
    updated_params = params.copy()
    updated_params['means3D'] = torch.nn.Parameter(positions.cuda())
    
    return updated_params


def update_key_node_transforms(
    state: TemporalState,
    graph_dict: Dict,
    delta_positions: torch.Tensor,
) -> Dict:
    """
    Update SE(3) transforms for key nodes based on position changes.
    
    Computes relative transforms from position deltas.
    
    Args:
        state: TemporalState
        graph_dict: Temporal graph
        delta_positions: (M_k, 3) position changes for key nodes
    
    Returns:
        updated_graph_dict: Graph with updated transforms
    """
    
    M_k = delta_positions.shape[0]
    
    # Update transforms with translation deltas
    # For now: only translation, rotation stays identity
    updated_transforms = graph_dict['key_transforms'].clone()
    
    for i in range(M_k):
        delta_t = delta_positions[i]  # (3,)
        # Update translation component
        updated_transforms[i, :3, 3] = updated_transforms[i, :3, 3] + delta_t
    
    # Create updated graph dict
    updated_graph = graph_dict.copy()
    updated_graph['key_transforms'] = updated_transforms
    
    return updated_graph
