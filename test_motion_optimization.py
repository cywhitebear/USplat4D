#!/usr/bin/env python
"""
Test motion optimization module.
"""

import torch
from usplat4d.state import TemporalState
from usplat4d.motion_optimization import (
    compute_key_node_loss,
    compute_non_key_loss,
    compute_motion_regularization_loss,
)


def test_motion_losses():
    """Test motion loss computation."""
    
    print("=" * 60)
    print("Testing Motion Optimization Losses")
    print("=" * 60)
    
    N = 100
    M_k = 5
    M_n = 95
    
    # Mock parameters
    params = {
        'means3D': torch.nn.Parameter(torch.randn(N, 3).cuda()),
        'unnorm_rotations': torch.nn.Parameter(torch.randn(N, 4).cuda()),
    }
    
    init_params = {
        'means3D': params['means3D'].data.clone() + torch.randn(N, 3).cuda() * 0.1,
        'unnorm_rotations': params['unnorm_rotations'].data.clone(),
    }
    
    # Mock state
    state = TemporalState(
        scene_radius=5.0,
        max_2D_radius=torch.zeros(N).cuda(),
        means2D_gradient_accum=torch.zeros(N).cuda(),
        denom=torch.zeros(N).cuda(),
        seen_any=torch.zeros(N, dtype=torch.bool).cuda(),
    )
    
    # Add uncertainty window
    for _ in range(3):
        state.uncertainty_window.append(torch.rand(N) * 0.5)
    
    # Mock graph dict
    key_indices = torch.arange(M_k)
    non_key_indices = torch.arange(M_k, N)
    
    graph_dict = {
        'timestep': 2,
        'key_indices': key_indices,
        'non_key_indices': non_key_indices,
        'key_key_edges': torch.randint(0, M_k, (M_k, 3)),
        'key_key_weights': F.softmax(torch.randn(M_k, 3), dim=1),
        'non_key_assignments': torch.randint(0, M_k, (M_n,)),
        'non_key_weights': torch.ones(M_n),
        'key_transforms': torch.eye(4).unsqueeze(0).repeat(M_k, 1, 1),
    }
    
    print(f"\nTesting loss functions...")
    print(f"  Key nodes: {M_k}")
    print(f"  Non-key nodes: {M_n}")
    
    # Test key node loss
    key_loss = compute_key_node_loss(params, state, graph_dict, init_params, t=2)
    print(f"\n✓ Key node loss: {key_loss.item():.6f}")
    assert key_loss.item() >= 0, "Loss should be non-negative"
    
    # Test non-key loss  
    non_key_loss = compute_non_key_loss(params, state, graph_dict, init_params, t=2)
    print(f"✓ Non-key loss: {non_key_loss.item():.6f}")
    assert non_key_loss.item() >= 0, "Loss should be non-negative"
    
    # Test motion regularization
    reg_loss = compute_motion_regularization_loss(params, state, graph_dict)
    print(f"✓ Motion regularization loss: {reg_loss.item():.6f}")
    assert reg_loss.item() >= 0, "Loss should be non-negative"
    
    # Test backward pass
    print(f"\nTesting backward pass...")
    total_loss = key_loss + non_key_loss + reg_loss
    total_loss.backward()
    
    assert params['means3D'].grad is not None, "Gradients should be computed"
    print(f"✓ Gradients computed successfully")
    print(f"  Gradient norm: {params['means3D'].grad.norm().item():.6f}")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    import torch.nn.functional as F
    test_motion_losses()
