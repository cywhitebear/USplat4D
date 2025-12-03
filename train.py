import torch
import os
import json
import copy
import numpy as np
from PIL import Image
from random import randint
from tqdm import tqdm
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from helpers import setup_camera, l1_loss_v1, l1_loss_v2, weighted_l2_loss_v1, weighted_l2_loss_v2, quat_mult, \
    o3d_knn, params2rendervar, params2cpu, save_params
from external import calc_ssim, calc_psnr, build_rotation, densify, update_params_and_optimizer
from usplat4d.uncertainty_proxy import compute_proxy_uncertainty_from_scale_opacity
from usplat4d.uncertainty_overlay_pil import overlay_uncertainty_on_image_pil
from usplat4d.state import TemporalState


def get_dataset(t, md, seq):
    dataset = []
    for c in range(len(md['fn'][t])):
        w, h, k, w2c = md['w'], md['h'], md['k'][t][c], md['w2c'][t][c]
        cam = setup_camera(w, h, k, w2c, near=1.0, far=100)
        fn = md['fn'][t][c]
        im = np.array(copy.deepcopy(Image.open(f"./data/{seq}/ims/{fn}")))
        im = torch.tensor(im).float().cuda().permute(2, 0, 1) / 255
        seg = np.array(copy.deepcopy(Image.open(f"./data/{seq}/seg/{fn.replace('.jpg', '.png')}"))).astype(np.float32)
        seg = torch.tensor(seg).float().cuda()
        seg_col = torch.stack((seg, torch.zeros_like(seg), 1 - seg))
        dataset.append({
            'cam': cam,
            'im': im,
            'seg': seg_col,
            'id': c,
            'K': torch.tensor(k).cuda().float(),        # 3x3 intrinsics
            'w2c': torch.tensor(w2c).cuda().float(),    # 4x4 world-to-camera
            'width': w,
            'height': h,
        })
    return dataset


def get_batch(todo_dataset, dataset):
    if not todo_dataset:
        todo_dataset = dataset.copy()
    curr_data = todo_dataset.pop(randint(0, len(todo_dataset) - 1))
    return curr_data


def initialize_params(seq, md):
    init_pt_cld = np.load(f"./data/{seq}/init_pt_cld.npz")["data"]
    seg = init_pt_cld[:, 6]
    max_cams = 50
    sq_dist, _ = o3d_knn(init_pt_cld[:, :3], 3)
    mean3_sq_dist = sq_dist.mean(-1).clip(min=0.0000001)
    params = {
        'means3D': init_pt_cld[:, :3],
        'rgb_colors': init_pt_cld[:, 3:6],
        'seg_colors': np.stack((seg, np.zeros_like(seg), 1 - seg), -1),
        'unnorm_rotations': np.tile([1, 0, 0, 0], (seg.shape[0], 1)),
        'logit_opacities': np.zeros((seg.shape[0], 1)),
        'log_scales': np.tile(np.log(np.sqrt(mean3_sq_dist))[..., None], (1, 3)),
        'cam_m': np.zeros((max_cams, 3)),
        'cam_c': np.zeros((max_cams, 3)),
    }
    params = {k: torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True)) for k, v in
              params.items()}
    cam_centers = np.linalg.inv(md['w2c'][0])[:, :3, 3]
    scene_radius = 1.1 * np.max(np.linalg.norm(
        cam_centers - np.mean(cam_centers, 0)[None], axis=-1
    ))

    N = params['means3D'].shape[0]

    state = TemporalState(
        scene_radius=scene_radius,
        max_2D_radius=torch.zeros(N, device="cuda"),
        means2D_gradient_accum=torch.zeros(N, device="cuda"),
        denom=torch.zeros(N, device="cuda"),
        seen_any=torch.zeros(N, device="cuda", dtype=torch.bool),
    )

    return params, state



def initialize_optimizer(params, state: TemporalState):
    lrs = {
        'means3D': 0.00016 * state.scene_radius,
        'rgb_colors': 0.0025,
        'seg_colors': 0.0,
        'unnorm_rotations': 0.001,
        'logit_opacities': 0.05,
        'log_scales': 0.001,
        'cam_m': 1e-4,
        'cam_c': 1e-4,
    }
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items()]
    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


def get_loss(params, curr_data, state: TemporalState, is_initial_timestep):
    losses = {}

    rendervar = params2rendervar(params)
    rendervar['means2D'].retain_grad()
    im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    curr_id = curr_data['id']
    im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
    losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))
    state.means2D = rendervar['means2D']  # Gradient only accum from colour render for densification

    # ------------------------------------------------------------
    # USplat4D-style photometric uncertainty per Gaussian
    # ------------------------------------------------------------
    pts_world = params['means3D']                   # (N,3)
    N = pts_world.shape[0]

    # homogeneous
    ones = torch.ones((N, 1), device=pts_world.device)
    pts_h = torch.cat([pts_world, ones], dim=1)      # (N,4)

    # project using w2c
    w2c = curr_data['w2c']                            # (4,4)
    pts_cam = (w2c @ pts_h.T).T                       # (N,4)
    Xc = pts_cam[:, 0]
    Yc = pts_cam[:, 1]
    Zc = pts_cam[:, 2].clamp(min=1e-6)

    # intrinsics
    K = curr_data['K']
    fx = K[0, 0]; fy = K[1, 1]
    cx = K[0, 2]; cy = K[1, 2]

    # pixel centers
    u = fx * (Xc / Zc) + cx
    v = fy * (Yc / Zc) + cy

    # clamp to image
    H = curr_data['height']
    W = curr_data['width']
    u_clamped = u.clamp(0, W - 1).long()
    v_clamped = v.clamp(0, H - 1).long()

    # gather pixel RGB
    pred_rgb = im[:, v_clamped, u_clamped].permute(1, 0)   # (N,3)
    gt_rgb   = curr_data['im'][:, v_clamped, u_clamped].permute(1, 0)

    # per-Gaussian photometric residual
    uncertainty = torch.abs(pred_rgb - gt_rgb).mean(dim=1)   # (N,)

    # accumulate for t >= 1
    if not is_initial_timestep:
        state.curr_uncertainty.append(uncertainty.detach().cpu())


    # ------------------------------------------------------------
    # USplat4D: one overlay dump
    # ------------------------------------------------------------
    if not state.saved_uncertainty_debug:
        with torch.no_grad():
            rendered_np = im.detach().clamp(0.0, 1.0).cpu().permute(1, 2, 0).numpy()

            centers2d_px = torch.stack([u, v], dim=1)          # (N,2)

            overlay_path = f"./output/uncertainty_debug_t{curr_id:02d}.png"
            overlay_uncertainty_on_image_pil(
                image_np=rendered_np,
                centers2d=centers2d_px,
                uncertainty=uncertainty,
                radii_px=radius,
                out_path=overlay_path,
            )
            print(f"[USplat4D DEBUG] Saved uncertainty overlay: {overlay_path}")

            state.saved_uncertainty_debug = True


    segrendervar = params2rendervar(params)
    segrendervar['colors_precomp'] = params['seg_colors']
    seg, _, _, = Renderer(raster_settings=curr_data['cam'])(**segrendervar)
    losses['seg'] = 0.8 * l1_loss_v1(seg, curr_data['seg']) + 0.2 * (1.0 - calc_ssim(seg, curr_data['seg']))

    if not is_initial_timestep:
        is_fg = (params['seg_colors'][:, 0] > 0.5).detach()
        fg_pts = rendervar['means3D'][is_fg]
        fg_rot = rendervar['rotations'][is_fg]

        rel_rot = quat_mult(fg_rot, state.prev_inv_rot_fg)
        rot = build_rotation(rel_rot)
        neighbor_pts = fg_pts[state.neighbor_indices]
        curr_offset = neighbor_pts - fg_pts[:, None]
        curr_offset_in_prev_coord = (rot.transpose(2, 1)[:, None] @ curr_offset[:, :, :, None]).squeeze(-1)
        losses['rigid'] = weighted_l2_loss_v2(curr_offset_in_prev_coord, state.prev_offset,
                                              state.neighbor_weight)

        losses['rot'] = weighted_l2_loss_v2(rel_rot[state.neighbor_indices], rel_rot[:, None],
                                            state.neighbor_weight)

        curr_offset_mag = torch.sqrt((curr_offset ** 2).sum(-1) + 1e-20)
        losses['iso'] = weighted_l2_loss_v1(curr_offset_mag, state.neighbor_dist, state.neighbor_weight)

        losses['floor'] = torch.clamp(fg_pts[:, 1], min=0).mean()

        bg_pts = rendervar['means3D'][~is_fg]
        bg_rot = rendervar['rotations'][~is_fg]
        losses['bg'] = l1_loss_v2(bg_pts, state.init_bg_pts) + l1_loss_v2(bg_rot, state.init_bg_rot)

        losses['soft_col_cons'] = l1_loss_v2(params['rgb_colors'], state.prev_col)

    loss_weights = {'im': 1.0, 'seg': 3.0, 'rigid': 4.0, 'rot': 4.0, 'iso': 2.0, 'floor': 2.0, 'bg': 20.0,
                    'soft_col_cons': 0.01}
    loss = sum([loss_weights[k] * v for k, v in losses.items()])

    seen = radius > 0
    state.max_2D_radius[seen] = torch.max(radius[seen], state.max_2D_radius[seen])
    state.seen = seen
    if not is_initial_timestep:
        if state.seen_any.shape[0] != seen.shape[0]:
            state.seen_any = torch.zeros_like(seen, dtype=torch.bool)
        state.seen_any |= seen

    return loss, state


def initialize_per_timestep(params, state: TemporalState, optimizer):
    pts = params['means3D']
    rot = torch.nn.functional.normalize(params['unnorm_rotations'])
    new_pts = pts + (pts - state.prev_pts)
    new_rot = torch.nn.functional.normalize(rot + (rot - state.prev_rot))

    is_fg = params['seg_colors'][:, 0] > 0.5
    prev_inv_rot_fg = rot[is_fg]
    prev_inv_rot_fg[:, 1:] = -1 * prev_inv_rot_fg[:, 1:]
    fg_pts = pts[is_fg]
    prev_offset = fg_pts[state.neighbor_indices] - fg_pts[:, None]
    state.prev_inv_rot_fg = prev_inv_rot_fg.detach()
    state.prev_offset = prev_offset.detach()
    state.prev_col = params['rgb_colors'].detach()
    state.prev_pts = pts.detach()
    state.prev_rot = rot.detach()

    new_params = {'means3D': new_pts, 'unnorm_rotations': new_rot}
    params = update_params_and_optimizer(new_params, params, optimizer)

    return params, state


def initialize_post_first_timestep(params, state: TemporalState, optimizer, num_knn=20):
    is_fg = params['seg_colors'][:, 0] > 0.5
    init_fg_pts = params['means3D'][is_fg]
    init_bg_pts = params['means3D'][~is_fg]
    init_bg_rot = torch.nn.functional.normalize(params['unnorm_rotations'][~is_fg])
    neighbor_sq_dist, neighbor_indices = o3d_knn(init_fg_pts.detach().cpu().numpy(), num_knn)
    neighbor_weight = np.exp(-2000 * neighbor_sq_dist)
    neighbor_dist = np.sqrt(neighbor_sq_dist)
    state.neighbor_indices = torch.tensor(neighbor_indices).cuda().long().contiguous()
    state.neighbor_weight = torch.tensor(neighbor_weight).cuda().float().contiguous()
    state.neighbor_dist = torch.tensor(neighbor_dist).cuda().float().contiguous()

    state.init_bg_pts = init_bg_pts.detach()
    state.init_bg_rot = init_bg_rot.detach()
    state.prev_pts = params['means3D'].detach()
    state.prev_rot = torch.nn.functional.normalize(params['unnorm_rotations']).detach()
    params_to_fix = ['logit_opacities', 'log_scales', 'cam_m', 'cam_c']
    for param_group in optimizer.param_groups:
        if param_group["name"] in params_to_fix:
            param_group['lr'] = 0.0
    return state


def report_progress(params, data, i, progress_bar, every_i=100):
    if i % every_i == 0:
        im, _, _, = Renderer(raster_settings=data['cam'])(**params2rendervar(params))
        curr_id = data['id']
        im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
        psnr = calc_psnr(im, data['im']).mean()
        progress_bar.set_postfix({"train img 0 PSNR": f"{psnr:.{7}f}"})
        progress_bar.update(every_i)


def train(seq, exp):
    if os.path.exists(f"./output/{exp}/{seq}"):
        print(f"Experiment '{exp}' for sequence '{seq}' already exists. Exiting.")
        return
    md = json.load(open(f"./data/{seq}/train_meta.json", 'r'))  # metadata
    num_timesteps = len(md['fn'])
    params, state = initialize_params(seq, md)
    optimizer = initialize_optimizer(params, state)
    output_params = []

    # temporal containers
    state.temporal_uncertainty = []    # list of per-timestep (N,) tensors (u_mean_t)
    state.curr_uncertainty = []   # per-iteration per-timestep uncertainties (N,)
    state.uncertainty_window = []      # sliding window of last W timesteps, (N,) each, CPU
    state.visibility_window = []       # sliding window of last W timesteps, (N,) bool, CPU


    for t in range(num_timesteps):
        dataset = get_dataset(t, md, seq)
        todo_dataset = []
        is_initial_timestep = (t == 0)

        if not is_initial_timestep:
            N = params['means3D'].shape[0]
            if state.seen_any.shape[0] != N:
                state.seen_any = torch.zeros(N, device="cuda", dtype=torch.bool)
            else:
                state.seen_any.zero_()

        if not is_initial_timestep:
            params, state = initialize_per_timestep(params, state, optimizer)
        num_iter_per_timestep = 10000 if is_initial_timestep else 2000
        progress_bar = tqdm(range(num_iter_per_timestep), desc=f"timestep {t}")
        for i in range(num_iter_per_timestep):
            curr_data = get_batch(todo_dataset, dataset)
            loss, state = get_loss(params, curr_data, state, is_initial_timestep)
            loss.backward()
            with torch.no_grad():
                report_progress(params, dataset[0], i, progress_bar)
                if is_initial_timestep:
                    params, state = densify(params, state, optimizer, i)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        progress_bar.close()

        # --- summarize uncertainty for this timestep ---
        # Only summarize uncertainty for t >= 1 (no densification step t=0)
        if not is_initial_timestep and len(state.curr_uncertainty) > 0:
            # stack: (num_iters, N)
            u_stack = torch.stack(state.curr_uncertainty, dim=0)   # (num_iters, N)
            # mean over iterations → (N,)
            u_mean = u_stack.mean(0)                                           # (N,)

            # store per-timestep uncertainty (for debugging / saving)
            state.temporal_uncertainty.append(u_mean.cpu())

            # update sliding windows (significant period)
            state.uncertainty_window.append(u_mean.cpu())               # (N,)
            state.visibility_window.append(state.seen_any.cpu()) # (N,)

            window_size = 5   # paper's minimum significant period
            if len(state.uncertainty_window) > window_size:
                state.uncertainty_window.pop(0)
                state.visibility_window.pop(0)

            # --- key-node selection with significant period ---
            W = len(state.uncertainty_window)
            T_min = 5  # significant period length

            if W >= T_min:
                # shape: (W, N)
                u_win = torch.stack(state.uncertainty_window, dim=0)        # float
                v_win = torch.stack(state.visibility_window, dim=0).float() # 0/1

                # significant period: number of visible frames in window
                sig_counts = v_win.sum(dim=0)                                      # (N,)

                # only Gaussians with enough visibility are considered
                valid_mask = sig_counts >= T_min                                   # (N,) bool

                if valid_mask.any():
                    # average uncertainty over significant period
                    weighted_sum = (u_win * v_win).sum(dim=0)                      # (N,)
                    avg_unc = torch.full_like(weighted_sum, float('inf'))          # (N,)
                    avg_unc[valid_mask] = weighted_sum[valid_mask] / sig_counts[valid_mask]

                    # lowest 2% avg uncertainty among valid Gaussians
                    tau = torch.quantile(avg_unc[valid_mask], 0.02)
                    base_key_mask = (avg_unc <= tau) & valid_mask                  # (N,) bool

                    # --------------------------------------------------
                    # voxel-based spatial sparsification (page 5)
                    # --------------------------------------------------
                    pts = params['means3D'].detach()                               # (N,3), cuda
                    N = pts.shape[0]

                    # set voxel size once, based on scene radius
                    if state.voxel_size is None:
                        # heuristic: coarse grid ~ 5% of scene radius
                        state.voxel_size = state.scene_radius * 0.05
                    voxel_size = state.voxel_size

                    # indices of base key candidates
                    idx_key = torch.nonzero(base_key_mask, as_tuple=False).view(-1)
                    if idx_key.numel() == 0:
                        key_mask = base_key_mask
                        state.key_gaussians.append(key_mask.cpu())
                        print(
                            f"[key-selection] t={t}, "
                            f"valid={valid_mask.sum().item()}, "
                            f"base_selected={base_key_mask.sum().item()}, "
                            f"voxel_selected=0, "
                            f"tau={tau.item():.6f}"
                        )
                    else:
                        pts_key = pts[idx_key].cpu()                               # (M,3)
                        avg_unc_key = avg_unc[idx_key].cpu()                       # (M,)

                        # voxel coordinates
                        coords = torch.floor(pts_key / voxel_size).to(torch.int64) # (M,3)

                        # choose one Gaussian per voxel: lowest uncertainty in that voxel
                        # we do this in Python for clarity; M is small (~2% of N)
                        coords_np = coords.numpy()
                        avg_unc_np = avg_unc_key.numpy()
                        idx_key_np = idx_key.cpu().numpy()

                        voxel_best = {}  # (vx,vy,vz) -> (best_unc, best_idx)
                        for v, u_val, g_idx in zip(coords_np, avg_unc_np, idx_key_np):
                            key_tuple = (int(v[0]), int(v[1]), int(v[2]))
                            if key_tuple not in voxel_best or u_val < voxel_best[key_tuple][0]:
                                voxel_best[key_tuple] = (u_val, g_idx)

                        # build final key_mask
                        key_mask = torch.zeros(N, dtype=torch.bool)
                        for _, (_, g_idx) in voxel_best.items():
                            key_mask[g_idx] = True

                        state.key_gaussians.append(key_mask.cpu())

                        print(
                            f"[key-selection] t={t}, "
                            f"valid={valid_mask.sum().item()}, "
                            f"base_selected={base_key_mask.sum().item()}, "
                            f"voxel_selected={key_mask.sum().item()}, "
                            f"tau={tau.item():.6f}"
                        )

                else:
                    print(f"[key-selection] t={t}, no Gaussians with significant period >= {T_min}")
            else:
                print(f"[key-selection] t={t}, window={W} < T_min={T_min}, skip key-node selection")

        state.curr_uncertainty = []

        output_params.append(params2cpu(params, is_initial_timestep))
        if is_initial_timestep:
            state = initialize_post_first_timestep(params, state, optimizer)
    save_params(output_params, seq, exp)
    if len(state.temporal_uncertainty) > 0:
        # (N, T) tensor
        unc_mat = torch.stack(state.temporal_uncertainty, dim=1)
        os.makedirs(f"./output/{exp}/{seq}", exist_ok=True)
        torch.save(unc_mat, f"./output/{exp}/{seq}/uncertainty_proxy.pt")
        print(f"[USplat4D] Saved proxy temporal uncertainty to ./output/{exp}/{seq}/uncertainty_proxy.pt")


if __name__ == "__main__":
    exp_name = "exp_uncertainty_1"
    # datasets = ["basketball", "boxes", "football", "juggle", "softball", "tennis"]
    datasets = ["basketball"]
    for sequence in datasets:
        train(sequence, exp_name)
        torch.cuda.empty_cache()
