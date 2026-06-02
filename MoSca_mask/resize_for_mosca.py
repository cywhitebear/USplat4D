#!/usr/bin/env python3
"""
Robust mask resizing script for MoSca dataset.

This script handles colored masks (e.g., red masks with black backgrounds) by:
1. Detecting non-black pixels as foreground
2. Using configurable thresholds for robustness
3. Providing both PIL-only and numpy-optimized implementations
4. Preserving binary mask quality during resizing

Usage: python resize_for_mosca.py
"""

import os
from pathlib import Path

from PIL import Image

# Optional numpy import for performance optimization
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


SOURCE_BASE = Path("/data/dataset/custom_mosca_trained_masked")
DEST_BASE = Path("/data/dataset/custom_mosca_trained_masked_4x")


def list_sequences(source_base: Path) -> list[Path]:
    if not source_base.exists():
        return []
    return [p for p in sorted(source_base.iterdir()) if p.is_dir()]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def compute_quarter_size(width: int, height: int) -> tuple[int, int]:
    new_w = max(1, width // 4)
    new_h = max(1, height // 4)
    return new_w, new_h


def resize_image_file(src_path: Path, dst_path: Path) -> None:
    with Image.open(src_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        new_size = compute_quarter_size(w, h)
        img_small = img.resize(new_size, resample=Image.LANCZOS)
        img_small.save(dst_path)


def generate_foreground_mask(image: Image.Image) -> Image.Image:
    """
    Generate a robust foreground mask from an image that may contain colored masks.
    
    Args:
        image: PIL Image that may contain colored foreground masks
        
    Returns:
        Binary mask image (0 for background, 255 for foreground)
    """
    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Method 1: Pure PIL approach (no numpy dependency)
    # Convert to grayscale using luminance formula
    gray_image = image.convert('L')
    
    # Method 2: Check for non-black pixels by examining RGB values
    # Create a mask by checking if any channel is above threshold
    width, height = image.size
    mask_data = []
    
    black_threshold = 30  # Adjust based on your needs
    
    for y in range(height):
        row = []
        for x in range(width):
            r, g, b = image.getpixel((x, y))
            # Pixel is foreground if any channel is above threshold
            is_foreground = r > black_threshold or g > black_threshold or b > black_threshold
            row.append(255 if is_foreground else 0)
        mask_data.append(row)
    
    # Create mask image from data
    mask_image = Image.new('L', (width, height))
    for y in range(height):
        for x in range(width):
            mask_image.putpixel((x, y), mask_data[y][x])
    
    return mask_image


def generate_foreground_mask_numpy(image: Image.Image) -> Image.Image:
    """
    Alternative numpy-based implementation for better performance on large images.
    Generate a robust foreground mask from an image that may contain colored masks.
    
    Args:
        image: PIL Image that may contain colored foreground masks
        
    Returns:
        Binary mask image (0 for background, 255 for foreground)
    """
    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array for easier processing
    img_array = np.array(image)
    
    # Method 1: Detect non-black pixels (most common case)
    # Check if any channel is significantly above black threshold
    black_threshold = 10  # Adjust based on your needs
    non_black_mask = np.any(img_array > black_threshold, axis=2)
    
    # Method 2: Detect pixels that are not pure black (0,0,0)
    pure_black_mask = np.all(img_array == 0, axis=2)
    foreground_mask = ~pure_black_mask
    
    # Method 3: Use luminance threshold (for grayscale-like masks)
    # Convert to grayscale and threshold
    gray = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
    luminance_mask = gray > 20
    
    # Combine methods: use the most restrictive approach
    # A pixel is foreground if it's not pure black AND has some luminance
    combined_mask = foreground_mask & non_black_mask
    
    # Convert back to PIL Image
    mask_image = Image.fromarray((combined_mask * 255).astype(np.uint8), mode='L')
    
    return mask_image


def resize_mask_file(src_path: Path, dst_path: Path) -> None:
    with Image.open(src_path) as m:
        # Generate robust foreground mask using the best available method
        if HAS_NUMPY:
            m = generate_foreground_mask_numpy(m)
        else:
            m = generate_foreground_mask(m)
        
        # First binarize at source resolution
        m = m.point(lambda v: 255 if v >= 128 else 0)
        w, h = m.size
        new_size = compute_quarter_size(w, h)
        # Use nearest to avoid interpolation
        m_small = m.resize(new_size, resample=Image.NEAREST)
        # Re-binarize to eliminate any non-binary artifacts
        m_small = m_small.point(lambda v: 255 if v >= 128 else 0)
        m_small.save(dst_path)


def iter_files(folder: Path, exts: tuple[str, ...]) -> list[Path]:
    if not folder.exists():
        return []
    return [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in exts]


def process_sequence(seq_dir: Path) -> None:
    src_img_dir = seq_dir / "images"
    src_mask_dir = seq_dir / "mask"

    dst_seq_dir = DEST_BASE / seq_dir.name
    dst_img_dir = dst_seq_dir / "images"
    dst_mask_dir = dst_seq_dir / "mask"

    ensure_dir(dst_img_dir)
    ensure_dir(dst_mask_dir)

    image_exts = (".jpg", ".jpeg", ".png")
    mask_exts = (".png", ".jpg", ".jpeg")

    for img_path in iter_files(src_img_dir, image_exts):
        dst_path = dst_img_dir / img_path.name
        resize_image_file(img_path, dst_path)

    for mask_path in iter_files(src_mask_dir, mask_exts):
        # Always save mask as PNG to preserve exact labels
        dst_name = mask_path.stem + ".png"
        dst_path = dst_mask_dir / dst_name
        resize_mask_file(mask_path, dst_path)


def main() -> None:
    ensure_dir(DEST_BASE)
    sequences = list_sequences(SOURCE_BASE)
    print(f"Found {len(sequences)} sequences in {SOURCE_BASE}")
    for idx, seq in enumerate(sequences, start=1):
        print(f"[{idx}/{len(sequences)}] seq={seq.name}")
        process_sequence(seq)


if __name__ == "__main__":
    main()


