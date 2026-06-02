# my code
import numpy as np
import torch
import cv2
import torch.nn.functional as F

def associate_pts_value(projected: torch.Tensor, value_map: torch.Tensor) -> torch.Tensor:
    """
    Args:
        projected (torch.Tensor): Nx2 tensor with [x, y] image coordinates (float).
        value_map (torch.Tensor): [H, W] float32 tensor

    Returns:
        torch.Tensor: tensor [N], True where projected point is in bounds and on a foreground pixel.
    """
    H, W = value_map.shape
    x, y = projected[:, 0], projected[:, 1]

    # Check bounds
    in_bounds = (x >= 0) & (x < W) & (y >= 0) & (y < H)

    # Convert to integer indices (clamp to avoid indexing errors)
    x_int = x.long().clamp(0, W - 1)
    y_int = y.long().clamp(0, H - 1)

    # set value to each projected points
    values = 1e-4*torch.ones_like(in_bounds) # small value
    values[in_bounds] = value_map[y_int[in_bounds], x_int[in_bounds]]

    return values

def valid_and_visible(projected: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
    """
    Args:
        projected (torch.Tensor): Nx2 tensor with [x, y] image coordinates (float).
        gt_mask (torch.Tensor): [H, W] float32 tensor, values in {0.0, 1.0}

    Returns:
        torch.Tensor: Boolean tensor [N], True where projected point is in bounds and on a foreground pixel.
    """
    H, W = gt_mask.shape
    x, y = projected[:, 0], projected[:, 1]

    # Check bounds
    in_bounds = (x >= 0) & (x < W) & (y >= 0) & (y < H)

    # Convert to integer indices (clamp to avoid indexing errors)
    x_int = x.long().clamp(0, W - 1)
    y_int = y.long().clamp(0, H - 1)

    # Check if gt_mask is 1.0 at those positions
    visible = torch.zeros_like(in_bounds, dtype=torch.bool)
    visible[in_bounds] = gt_mask[y_int[in_bounds], x_int[in_bounds]] >= 0.5 # == 1.0 has bug (values be be close but not equal to 1.0)

    return visible

def project_points(pts: torch.Tensor, K: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """
    Project 3D points to 2D image coordinates.

    Args:
        pts (torch.Tensor): Nx3 tensor of 3D points in world coordinates.
        K (torch.Tensor): 3x3 camera intrinsic matrix.
        T (torch.Tensor): 4x4 camera extrinsic matrix (camera-to-world or world-to-camera).

    Returns:
        torch.Tensor: Nx2 tensor of 2D projected points in image coordinates.

    Usage: 
        projected = project_points(pts, K, w2c)
    """
    assert pts.shape[1] == 3, "pts should be Nx3"
    assert K.shape == (3, 3), "K should be 3x3"
    assert T.shape == (4, 4), "T should be 4x4"

    # Convert to homogeneous coordinates: Nx4
    pts_h = torch.cat([pts, torch.ones_like(pts[:, :1])], dim=-1)  # Nx4

    # Transform points from world to camera coordinates (assuming T = [R|t], world-to-camera)
    pts_cam = (T @ pts_h.T).T  # Nx4

    # Drop the homogeneous dimension for projection
    pts_cam = pts_cam[:, :3]

    # Apply intrinsics (project to 2D)
    pts_2d = (K @ pts_cam.T).T  # Nx3

    # Normalize by z (perspective divide)
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:3]  # Nx2

    return pts_2d

def draw_projected_points(img: np.ndarray, projected_pts: torch.Tensor, radius=4, color=(0, 255, 0)) -> np.ndarray:
    """
    Draw 2D projected points on an image.

    Args:
        img (np.ndarray): The image to draw on (H x W x 3).
        projected_pts (torch.Tensor): Nx2 tensor of 2D points. [w,h]
        radius (int): Radius of the drawn circles.
        color (tuple): BGR color of the circles.

    Returns:
        np.ndarray: Image with points drawn.
    """
    img_draw = img.copy()
    for pt in projected_pts:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:  # check if point is in bounds
            cv2.circle(img_draw, (x, y), radius, color, -1)
    return img_draw

def draw_projected_points_colored(
    img: np.ndarray,
    projected_pts: torch.Tensor,
    values: torch.Tensor,
    radius=4,
    colormap=cv2.COLORMAP_JET,
    vmin=None,
    vmax=None
) -> np.ndarray:
    """
    Draw 2D projected points on an image, with color based on per-point values.

    Args:
        img (np.ndarray): H x W x 3 image.
        projected_pts (torch.Tensor): Nx2 tensor of 2D points (w, h).
        values (torch.Tensor): [N] tensor of values to determine color.
        radius (int): Circle radius.
        colormap (int): OpenCV colormap to use (default: JET).
        vmin (float, optional): Min value for normalization. If None, use min(values).
        vmax (float, optional): Max value for normalization. If None, use max(values).

    Returns:
        np.ndarray: Image with circles drawn.
    """
    img_draw = img.copy()
    projected_pts_np = projected_pts.cpu().numpy()
    values_np = values.cpu().numpy()

    if vmin is None:
        vmin = values_np.min()
    if vmax is None:
        vmax = values_np.max()
    values_norm = (values_np - vmin) / (vmax - vmin + 1e-8)  # Normalize to [0, 1]

    # Map normalized values to 0-255
    values_uint8 = np.clip(values_norm * 255, 0, 255).astype(np.uint8)
    colors = cv2.applyColorMap(values_uint8, colormap)  # (N, 1, 3)

    for pt, color in zip(projected_pts_np, colors):
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            color_bgr = tuple(int(c) for c in color[0])  # Color needs to be BGR tuple
            cv2.circle(img_draw, (x, y), radius, color_bgr, -1)

    return img_draw

def draw_projected_points_masked(
    img: np.ndarray,
    projected_pts: torch.Tensor,
    in_mask: torch.Tensor,
    radius=1,
    color_in=(0, 255, 255),   # Red in BGR if cv2 # Blue in RGB if iio 
    color_out=(255, 0, 0)   # Blue in BGR if cv2 # Red in RGB if iio
) -> np.ndarray:
    """
    Draw 2D projected points on an image, with color based on whether they fall inside a mask.

    Args:
        img (np.ndarray): H x W x 3 image.
        projected_pts (torch.Tensor): Nx2 tensor of 2D points (w, h).
        in_mask (torch.Tensor): Boolean tensor of shape [N], True if point is inside gt_mask.
        radius (int): Circle radius.
        color_in (tuple): Color for in-mask points (default: blue).
        color_out (tuple): Color for out-of-mask points (default: red).

    Returns:
        np.ndarray: Image with circles drawn.
    """
    img_draw = img.copy()
    projected_pts_np = projected_pts.cpu().numpy()
    in_mask_np = in_mask.cpu().numpy()

    for pt, valid in zip(projected_pts_np, in_mask_np):
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            if valid: # test
                continue
            color = color_in if valid else color_out
            cv2.circle(img_draw, (x, y), radius, color, -1)

    return img_draw

def overlay_mask_yellow(img: np.ndarray, mask: np.ndarray, color= (0,255,255)) -> np.ndarray:
    """
    Overlay yellow tint on the non-mask region (mask == 0), keeping mask == 1 unchanged.

    Args:
        img (np.ndarray): H x W x 3 uint8 image.
        mask (np.ndarray): H x W float32 or uint8 mask, values in {0, 1}.

    Returns:
        np.ndarray: Image with yellow overlay on mask=0 region.
    """
    assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3
    assert mask.shape[:2] == img.shape[:2]

    # Normalize mask to binary uint8
    mask_bin = (mask > 0.5).astype(np.uint8)  # [H, W], 1 where foreground

    # Create yellow overlay: [0, 255, 255] in BGR
    yellow = np.full_like(img, color, dtype=np.uint8)

    # Create alpha mask: 0.5 opacity where mask == 0
    alpha = 0.7
    mask_3ch = np.repeat(mask_bin[:, :, None], 3, axis=2)  # [H, W, 3]

    # Blend where mask == 0
    out_img = img.copy()
    out_img[mask_3ch == 0] = (
        alpha * yellow[mask_3ch == 0] + (1 - alpha) * img[mask_3ch == 0]
    ).astype(np.uint8)

    return out_img

def stack_images_side_by_side(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Stack two images horizontally. Pads them to the same height if needed.

    Args:
        img1 (np.ndarray): First image (H x W x 3).
        img2 (np.ndarray): Second image (H x W x 3).

    Returns:
        np.ndarray: Concatenated image.
    """
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape
    max_h = max(h1, h2)

    # Pad height if needed
    if h1 < max_h:
        pad = np.zeros((max_h - h1, w1, 3), dtype=img1.dtype)
        img1 = np.vstack([img1, pad])
    if h2 < max_h:
        pad = np.zeros((max_h - h2, w2, 3), dtype=img2.dtype)
        img2 = np.vstack([img2, pad])

    return np.hstack([img1, img2])


def compute_depth_gradient(depth):
    """ 
    Compute gradients of a batch of depth maps and keep the same shape.
    
    Args:
        depth (torch.Tensor): shape (n, H, W)
    
    Returns:
        grad_x (torch.Tensor): shape (n, H, W)
        grad_y (torch.Tensor): shape (n, H, W)
    """
    n, H, W = depth.shape
    depth = depth.unsqueeze(1)  # (n, 1, H, W)

    # Sobel kernels
    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=depth.dtype, device=depth.device).view(1, 1, 3, 3)
    
    sobel_y = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=depth.dtype, device=depth.device).view(1, 1, 3, 3)

    grad_x = F.conv2d(depth, sobel_x, padding=1)
    grad_y = F.conv2d(depth, sobel_y, padding=1)

    grad_x = grad_x.squeeze(1)  # (n, H, W)
    grad_y = grad_y.squeeze(1)  # (n, H, W)

    grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)  # add epsilon to avoid sqrt(0)

    return grad_magnitude, grad_x, grad_y


if __name__=='__main__':
    pass