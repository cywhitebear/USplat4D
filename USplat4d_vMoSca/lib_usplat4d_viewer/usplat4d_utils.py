import torch
import numpy as np
import math

def test_camera_distance(cam1_c2w, cam2_c2w):
    # 1) Translation difference (Euclidean)
    t1 = cam1_c2w[:, :3, 3]
    t2 = cam2_c2w[:, :3, 3]
    delta_t = (t2 - t1).norm().item()

    return delta_t*1000 # mm

def test_camera_angle(cam1_c2w, cam2_c2w):
    # 2) Rotation difference  
    #    R_diff = R2 * R1ᵀ
    R1 = cam1_c2w[:, :3, :3]
    R2 = cam2_c2w[:, :3, :3]
    R_diff = R2 @ R1.transpose(-1, -2) # R1.T = R1^-1

    #    the angle of that relative rotation:
    trace = R_diff.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    angle_rad = torch.acos((trace - 1) / 2.0)
    angle_deg = torch.rad2deg(angle_rad)
    return angle_deg

def find_global_rotation(R_est: torch.Tensor, R_ref: torch.Tensor) -> torch.Tensor:
    """
    R_est, R_ref: (N,3,3) tensor of estimated and reference rotations.
    Returns R_align: (3,3) tensor so that R_align @ R_est[i] ≈ R_ref[i].
    """
    # 1) build correlation H = sum_i R_ref[i] @ R_est[i].T
    H = torch.zeros(3,3, device=R_est.device, dtype=R_est.dtype)
    for i in range(R_est.shape[0]):
        H += R_ref[i] @ R_est[i].transpose(0,1)

    # 2) SVD
    U, S, Vt = torch.linalg.svd(H, full_matrices=False)

    # 3) Procrustes: R = U V^T, fix reflection if needed
    R_align = U @ Vt
    if torch.det(R_align) < 0:
        U[:, -1] *= -1
        R_align = U @ Vt

    return R_align