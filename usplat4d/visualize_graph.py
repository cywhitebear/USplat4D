# usplat4d/visualize_graph.py
"""
Visualize temporal graph by marking key/non-key nodes.
"""

import torch
import numpy as np
from PIL import Image, ImageDraw


def visualize_graph_on_image(
    image_np,
    centers2d,
    key_mask,
    radii_px=None,
    out_path="graph_vis.png"
):
    """
    Visualize temporal graph by coloring key nodes vs non-key nodes.
    
    Args:
        image_np: (H, W, 3) RGB image [0, 1]
        centers2d: (N, 2) 2D centers [u, v]
        key_mask: (N,) bool tensor, True for key nodes
        radii_px: (N,) pixel radii (optional)
        out_path: Output file path
    """
    
    H, W = image_np.shape[:2]
    
    # Convert to uint8
    img_uint8 = (np.clip(image_np, 0, 1) * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    draw = ImageDraw.Draw(pil_img, 'RGBA')
    
    centers2d_np = centers2d.cpu().numpy() if isinstance(centers2d, torch.Tensor) else centers2d
    key_mask_np = key_mask.cpu().numpy() if isinstance(key_mask, torch.Tensor) else key_mask
    
    if radii_px is not None:
        radii_np = radii_px.cpu().numpy() if isinstance(radii_px, torch.Tensor) else radii_px
    else:
        radii_np = np.ones(len(centers2d_np)) * 2
    
    # Draw non-key nodes first (blue)
    for i, (center, is_key) in enumerate(zip(centers2d_np, key_mask_np)):
        if is_key:
            continue
        u, v = center
        if 0 <= u < W and 0 <= v < H:
            r = max(1, int(radii_np[i] * 0.5))
            draw.ellipse(
                [u - r, v - r, u + r, v + r],
                fill=(0, 100, 255, 5),  # Blue, semi-transparent
                outline=None
            )
    
    # Draw key nodes on top (red)
    for i, (center, is_key) in enumerate(zip(centers2d_np, key_mask_np)):
        if not is_key:
            continue
        u, v = center
        if 0 <= u < W and 0 <= v < H:
            r = max(2, int(radii_np[i] * 0.7))
            draw.ellipse(
                [u - r, v - r, u + r, v + r],
                fill=(255, 50, 50, 150),  # Red, more opaque
                outline=(255, 0, 0, 255)
            )
    
    pil_img.save(out_path)
    print(f"[visualize_graph] Saved to {out_path}")
