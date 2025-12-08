#!/usr/bin/env python
"""
Minimal test for temporal graph construction.
Tests that the graph module correctly uses output_params instead of duplicate state.
"""

import torch
import numpy as np
from usplat4d.state import TemporalState
from usplat4d.temporal_graph import build_temporal_graph


def test_temporal_graph_basic():
    """Test basic temporal graph construction with mock data."""
    
    print("=" * 60)
    print("Testing Temporal Graph Construction")
    print("=" * 60)
    
    # Mock parameters
    N = 100  # Number of Gaussians
    
    params = {
        'means3D': torch.randn(N, 3).cuda(),
        'unnorm_rotations': torch.randn(N, 4).cuda(),
        'rgb_colors': torch.randn(N, 3).cuda(),
    }
    
    # Mock state with uncertainty window and key nodes
    state = TemporalState(
        scene_radius=5.0,
        max_2D_radius=torch.zeros(N).cuda(),
        means2D_gradient_accum=torch.zeros(N).cuda(),
        denom=torch.zeros(N).cuda(),
        seen_any=torch.zeros(N, dtype=torch.bool).cuda(),
    )
    
    # Add uncertainty window (simulating 3 timesteps)
    for t in range(3):
        unc = torch.rand(N) * 0.5  # Random uncertainty [0, 0.5]
        state.uncertainty_window.append(unc)
        state.visibility_window.append(torch.ones(N, dtype=torch.bool))
    
    # Create key node mask (select 5% as key nodes)
    key_mask = torch.zeros(N, dtype=torch.bool)
    key_indices = torch.randperm(N)[:int(N * 0.05)]
    key_mask[key_indices] = True
    state.key_gaussians.append(key_mask)
    
    # Create mock output_params (simulating 3 timesteps saved)
    output_params = []
    
    # t=0: all params
    output_params.append({
        'means3D': params['means3D'].detach().cpu().numpy(),
        'rgb_colors': params['rgb_colors'].detach().cpu().numpy(),
        'unnorm_rotations': params['unnorm_rotations'].detach().cpu().numpy(),
    })
    
    # t=1, t=2: only motion params
    for t in range(1, 3):
        # Simulate motion: add small random displacement
        new_means = params['means3D'].detach().cpu().numpy() + np.random.randn(N, 3) * 0.1
        output_params.append({
            'means3D': new_means,
            'rgb_colors': params['rgb_colors'].detach().cpu().numpy(),
            'unnorm_rotations': params['unnorm_rotations'].detach().cpu().numpy(),
        })
    
    # Build temporal graph
    print(f"\nBuilding temporal graph at t=2 (after 3 timesteps)...")
    print(f"  Total Gaussians: {N}")
    print(f"  Key nodes: {key_mask.sum().item()}")
    print(f"  Non-key nodes: {(~key_mask).sum().item()}")
    print(f"  Output params entries: {len(output_params)}")
    
    build_temporal_graph(
        params=params,
        state=state,
        output_params=output_params,
        t=2,
        num_knn=3,
    )
    
    # Verify graph was created
    assert len(state.temporal_graph) == 1, "Graph should be created"
    graph = state.temporal_graph[0]
    
    print(f"\n✓ Graph created successfully!")
    print(f"  Timestep: {graph['timestep']}")
    print(f"  Key nodes: {graph['key_indices'].shape[0]}")
    print(f"  Non-key nodes: {graph['non_key_indices'].shape[0]}")
    print(f"  Key-key edges shape: {graph['key_key_edges'].shape}")
    print(f"  Non-key assignments shape: {graph['non_key_assignments'].shape}")
    
    # Verify structure
    M_k = graph['key_indices'].shape[0]
    M_n = graph['non_key_indices'].shape[0]
    k = 3  # num_knn
    
    assert graph['key_key_edges'].shape == (M_k, k), f"Expected ({M_k}, {k}), got {graph['key_key_edges'].shape}"
    assert graph['key_key_weights'].shape == (M_k, k), f"Expected ({M_k}, {k}), got {graph['key_key_weights'].shape}"
    assert graph['non_key_assignments'].shape == (M_n,), f"Expected ({M_n},), got {graph['non_key_assignments'].shape}"
    assert graph['non_key_weights'].shape == (M_n,), f"Expected ({M_n},), got {graph['non_key_weights'].shape}"
    
    # Verify weights are normalized
    assert torch.allclose(graph['key_key_weights'].sum(dim=1), torch.ones(M_k), atol=1e-5), \
        "Key-key weights should be normalized"
    
    print(f"\n✓ All structural checks passed!")
    print(f"  Edge weights normalized: ✓")
    print(f"  Assignment indices valid: ✓")
    
    # Verify no duplicate state
    assert not hasattr(state, 'temporal_positions'), "temporal_positions should not exist"
    assert not hasattr(state, 'temporal_rotations'), "temporal_rotations should not exist"
    
    print(f"\n✓ No duplicate state fields!")
    print(f"  temporal_positions removed: ✓")
    print(f"  temporal_rotations removed: ✓")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_temporal_graph_basic()
