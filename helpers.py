import torch
import os
import open3d as o3d
import numpy as np
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera


def setup_camera(w, h, k, w2c, near=0.01, far=100):
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = torch.tensor(w2c).cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)
    cam = Camera(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=w2c,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_center,
        prefiltered=False
    )
    return cam


def params2rendervar(params, t=0, variables=None):
    if variables is not None:
        means3D, rotations = evaluate_trajectory(params, variables, t)
    else:
        means3D = params['means3D']
        rotations = torch.nn.functional.normalize(params['unnorm_rotations'])
    rendervar = {
        'means3D': means3D,
        'colors_precomp': params['rgb_colors'],
        'rotations': rotations,
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(params['log_scales']),
        'means2D': torch.zeros_like(means3D, requires_grad=True, device="cuda") + 0
    }
    return rendervar


def l1_loss_v1(x, y):
    return torch.abs((x - y)).mean()


def l1_loss_v2(x, y):
    return (torch.abs(x - y).sum(-1)).mean()


def weighted_l2_loss_v1(x, y, w):
    return torch.sqrt(((x - y) ** 2) * w + 1e-20).mean()


def weighted_l2_loss_v2(x, y, w):
    return torch.sqrt(((x - y) ** 2).sum(-1) * w + 1e-20).mean()


def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z]).T


def o3d_knn(pts, num_knn):
    indices = []
    sq_dists = []
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    for p in pcd.points:
        [_, i, d] = pcd_tree.search_knn_vector_3d(p, num_knn + 1)
        indices.append(i[1:])
        sq_dists.append(d[1:])
    return np.array(sq_dists), np.array(indices)


def params2cpu(params, is_initial_timestep):
    if is_initial_timestep:
        res = {k: v.detach().cpu().contiguous().numpy() for k, v in params.items()}
    else:
        res = {k: v.detach().cpu().contiguous().numpy() for k, v in params.items() if
               k in ['means3D', 'rgb_colors', 'unnorm_rotations']}
    return res


def save_params(output_params, seq, exp):
    to_save = {}
    for k in output_params[0].keys():
        if k in output_params[1].keys():
            to_save[k] = np.stack([params[k] for params in output_params])
        else:
            to_save[k] = output_params[0][k]
    os.makedirs(f"./output/{exp}/{seq}", exist_ok=True)
    np.savez(f"./output/{exp}/{seq}/params", **to_save)


def evaluate_trajectory(params, variables, t):
    """
    t may be int or float.
    Uses linear interpolation for means3D and slerp for rotations.
    If traj_continuous is empty -> discrete fallback.
    """
    if 'traj_snapshots' not in variables or len(variables['traj_snapshots']) == 0:
        means3D = params['means3D']
        rotations = torch.nn.functional.normalize(params['unnorm_rotations'])
        return means3D, rotations

    traj_snaps = variables['traj_snapshots']
    nums_t = len(traj_snaps)

    if t <= 0:
        return traj_snaps[0]['means3D'], torch.nn.functional.normalize(traj_snaps[0]['rotations'])
    if t >= num_t - 1:
        return traj_snaps[-1]['means3D'], torch.nn.functional.normalize(traj_snaps[-1]['rotations'])

    t0 = int(np.floor(t))
    t1 = t0 + 1
    alpha = t - t0
    means3D_0 = traj_snaps[t0]['means3D']
    means3D_1 = traj_snaps[t1]['means3D']
    rotations_0 = torch.nn.functional.normalize(traj_snaps[t0]['rotations'])
    rotations_1 = torch.nn.functional.normalize(traj_snaps[t1]['rotations'])
    means3D = lerp(means3D_0, means3D_1, alpha)
    rotations = slerp(rotations_0, rotations_1, alpha)
    return means3D, rotations

def lerp(a, b, t):
    return a * (1 - t) + b * t


def slerp(q1, q2, t, tolerance=1e-7):
    dot = (q1 * q2).sum(-1)
    q2_ = torch.where(dot[:, None] < 0, -q2, q2)
    dot = torch.abs(dot)
    theta = torch.acos(torch.clamp(dot, -1 + tolerance, 1 - tolerance))
    sin_theta = torch.sin(theta)
    s1 = torch.sin((1 - t) * theta) / (sin_theta + tolerance)
    s2 = torch.sin(t * theta) / (sin_theta + tolerance)
    return (q1 * s1[:, None]) + (q2_ * s2[:, None])
    