# usplat4d/uncertainty_overlay_pil.py
import numpy as np
import torch
from PIL import Image, ImageDraw

def normalize_for_colormap(u: torch.Tensor) -> np.ndarray:
    u_np = u.detach().cpu().numpy().astype(np.float32)
    u_min, u_max = float(u_np.min()), float(u_np.max())
    if u_max - u_min < 1e-12:
        return np.zeros_like(u_np)
    return (u_np - u_min) / (u_max - u_min)

def colormap_jet(value: float) -> tuple:
    """Simple jet-like RGB colormap for value in [0,1]."""
    v = max(0.0, min(1.0, value))
    r = int(255 * np.clip(1.5 - abs(4*v - 3), 0, 1))
    g = int(255 * np.clip(1.5 - abs(4*v - 2), 0, 1))
    b = int(255 * np.clip(1.5 - abs(4*v - 1), 0, 1))
    return (r, g, b)

def overlay_uncertainty_on_image_pil(
    image_np: np.ndarray,           # (H,W,3) float or uint8
    centers2d: torch.Tensor,        # (N,2) pixel coords
    uncertainty: torch.Tensor,      # (N,)
    radii_px: torch.Tensor,         # (N,)
    out_path: str
):
    """Draw circles colored by uncertainty using PIL."""
    # convert image
    if image_np.dtype != np.uint8:
        img = (np.clip(image_np, 0, 1) * 255).astype(np.uint8)
    else:
        img = image_np.copy()

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img, "RGBA")

    centers = centers2d.detach().cpu().numpy()
    if centers.shape[1] >= 2:
        centers = centers[:, :2]   # use only (x,y)
    else:
        raise ValueError("means2D must have at least 2 columns (x,y).")
    radii = radii_px.detach().cpu().numpy()
    u_norm = normalize_for_colormap(uncertainty)

    for (x, y), r, u in zip(centers, radii, u_norm):
        xi, yi = int(x), int(y)
        rr = max(1, int(r))
        color = colormap_jet(float(u))
        # semi-transparent fill
        draw.ellipse(
            (xi - rr, yi - rr, xi + rr, yi + rr),
            fill=color + (1,),   # RGBA, alpha=100 for transparency
            outline=color + (4,)
        )

    pil_img.save(out_path)
