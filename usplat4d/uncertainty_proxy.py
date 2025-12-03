# usplat4d/uncertainty_proxy.py
import numpy as np
import torch

def compute_proxy_uncertainty_from_scale_opacity(
    gaussian_scales: torch.Tensor,    # (N,3) or (N,) floats
    gaussian_opacity: torch.Tensor,   # (N,) in [0,1]
    depth_scale_factor: float = 4.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Heuristic proxy uncertainty per Gaussian.

    u_g = (mean_plane_scale * 1.0 + depth_scale * scale_z) * (1 - opacity + eps)

    Rationale:
      - larger spatial scale -> higher uncertainty
      - larger depth extent -> more uncertain (multiply by depth_scale_factor)
      - higher opacity => lower uncertainty (more confident)
    """
    if gaussian_scales.ndim == 1:
        mean_plane = gaussian_scales.abs()  # (N,)
        z_scale = torch.zeros_like(mean_plane)
    else:
        # assume (N,3) with axes [sx, sy, sz]
        sx = gaussian_scales[:, 0].abs()
        sy = gaussian_scales[:, 1].abs()
        sz = gaussian_scales[:, 2].abs()
        mean_plane = 0.5 * (sx + sy)
        z_scale = sz

    base = mean_plane + depth_scale_factor * z_scale
    u = base * (1.0 - gaussian_opacity.clamp(min=0.0, max=1.0) + eps)
    return u  # (N,)


def normalize_for_colormap(u: torch.Tensor) -> np.ndarray:
    u_np = u.detach().cpu().numpy().astype(np.float32)
    u_min, u_max = u_np.min(), u_np.max()
    if u_max - u_min < 1e-12:
        u_norm = np.zeros_like(u_np)
    else:
        u_norm = (u_np - u_min) / (u_max - u_min)
    return u_norm  # [0,1]


def compute_frame_mean_uncertainty(uncertainty: torch.Tensor) -> float:
    return float(uncertainty.detach().cpu().mean().item())
