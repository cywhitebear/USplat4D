# usplat4d/state.py
from dataclasses import dataclass, field
import torch

@dataclass
class TemporalState:
    # ---- global scene info ----
    scene_radius: float = None
    voxel_size: float = None

    # ---- per-Gaussian persistent stats ----
    max_2D_radius: torch.Tensor = None              # (N,)
    means2D_gradient_accum: torch.Tensor = None     # (N,)
    denom: torch.Tensor = None                      # (N,)

    # also used by densification / rendering
    means2D: torch.Tensor = None                    # (N,3) from renderer
    seen: torch.Tensor = None                       # (N,) bool, current iteration visibility

    # ---- multi-timestep temporal stats ----
    temporal_uncertainty: list = field(default_factory=list)   # list of (N,) tensors (per timestep mean unc)
    key_gaussians: list = field(default_factory=list)          # list of (N,) bool masks
    uncertainty_window: list = field(default_factory=list)     # sliding window: list of (N,) tensors
    visibility_window: list = field(default_factory=list)      # sliding window: list of (N,) bool tensors

    # ---- per-timestep accumulators ----
    curr_uncertainty: list = field(default_factory=list)       # list of (N,) tensors (per iter)
    seen_any: torch.Tensor = None                              # (N,) bool, union over cameras at this timestep

    # ---- previous-timestep data for motion constraints ----
    prev_pts: torch.Tensor = None
    prev_rot: torch.Tensor = None
    prev_col: torch.Tensor = None
    prev_inv_rot_fg: torch.Tensor = None
    prev_offset: torch.Tensor = None

    # ---- neighbor graph ----
    neighbor_indices: torch.Tensor = None
    neighbor_weight: torch.Tensor = None
    neighbor_dist: torch.Tensor = None
    init_bg_pts: torch.Tensor = None
    init_bg_rot: torch.Tensor = None

    # ---- temporal graph (USplat4D §4.2(b)) ----
    temporal_graph: list = field(default_factory=list)         # list of dicts per timestep

    # ---- debug flags ----
    saved_uncertainty_debug: bool = False
