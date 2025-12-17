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

from usplat4d.state import TemporalState
from usplat4d.uncertainty_overlay_pil import overlay_uncertainty_on_image_pil
from usplat4d.uncertainty_window import update_uncertainty_window
from usplat4d.keynode_selection import select_key_nodes_from_window
from usplat4d.uncertainty import compute_gaussian_uncertainty
from usplat4d.temporal_graph import build_temporal_graph
from usplat4d.visualize_graph import visualize_graph_on_image
from usplat4d.motion_optimization import (
    compute_key_node_loss,
    compute_non_key_loss,
    compute_motion_regularization_loss,
)


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


def add_transform_parameters_to_optimizer(optimizer, state: TemporalState, lr=0.001):
    """
    Add SE(3) transform parameters from temporal graph to optimizer.
    
    Called after graph construction to enable transform optimization.
    """
    if len(state.temporal_graph) == 0:
        return
    
    for graph_dict in state.temporal_graph:
        key_transforms = graph_dict['key_transforms']
        if isinstance(key_transforms, torch.nn.Parameter):
            # Check if already in optimizer
            param_in_optimizer = any(
                key_transforms is group['params'][0] 
                for group in optimizer.param_groups
            )
            if not param_in_optimizer:
                optimizer.add_param_group({
                    'params': [key_transforms],
                    'name': f"key_transforms_t{graph_dict['timestep']}",
                    'lr': lr,
                })


def compute_centers2d(params, curr_data):
    """
    Project Gaussians to 2D for visualization purposes.
    """
    pts_world = params["means3D"]
    N = pts_world.shape[0]
    ones = torch.ones((N, 1), device=pts_world.device)
    pts_h = torch.cat([pts_world, ones], dim=1)
    w2c = curr_data["w2c"]
    pts_cam = (w2c @ pts_h.T).T
    K = curr_data["K"]
    u = K[0, 0] * (pts_cam[:, 0] / pts_cam[:, 2].clamp(min=1e-6)) + K[0, 2]
    v = K[1, 1] * (pts_cam[:, 1] / pts_cam[:, 2].clamp(min=1e-6)) + K[1, 2]
    return torch.stack([u, v], dim=1)


def get_loss(params, curr_data, state: TemporalState, is_initial_timestep, t=None):
    losses = {}

    # 1) render RGB with paper-faithful uncertainty outputs (USplat4D §4.1, Eq. 2-3)
    rendervar = params2rendervar(params)
    rendervar["means2D"].retain_grad()
    
    # Modified rasterizer returns: (color, radius, depth, gaussian_ids, transmittance_alpha)
    render_outputs = Renderer(raster_settings=curr_data["cam"])(**rendervar)
    
    # Unpack outputs - check if rasterizer is modified (5 outputs) or not (3 outputs)
    if len(render_outputs) == 5:
        im, radius, depth, gaussian_ids, transmittance_alpha = render_outputs
        rasterizer_modified = True
    else:
        # Fallback: old rasterizer with 3 outputs
        im, radius, depth = render_outputs
        rasterizer_modified = False
    
    curr_id = curr_data["id"]
    im = torch.exp(params["cam_m"][curr_id])[:, None, None] * im + params["cam_c"][curr_id][:, None, None]
    losses["im"] = 0.8 * l1_loss_v1(im, curr_data["im"]) + 0.2 * (1.0 - calc_ssim(im, curr_data["im"]))

    state.means2D = rendervar["means2D"]

    # 2) per-Gaussian uncertainty (USplat4D §4.1, Eq. 3, 5)
    if rasterizer_modified:
        # Paper-faithful uncertainty using transmittance and gaussian IDs
        uncertainty = compute_gaussian_uncertainty(
            gaussians_per_pixel=gaussian_ids,
            transmittance_alpha=transmittance_alpha,
            num_gaussians=params['means3D'].shape[0],
            convergence_threshold=0.1,  # ηc in Eq. 5 (relaxed from 0.05)
            rendered_image=im.permute(1, 2, 0),  # [H, W, 3]
            target_image=curr_data["im"].permute(1, 2, 0),  # [H, W, 3]
            phi_constant=1e7,  # φ for non-converged pixels (much larger than typical σ²)
        )
        
        # Debug: print uncertainty statistics (only first time per timestep)
        if not is_initial_timestep and len(state.curr_uncertainty) == 0 and t is not None:
            with torch.no_grad():
                n_phi = (uncertainty >= 1e7).sum().item()
                u_converged = uncertainty[uncertainty < 1e7]  # Exclude φ-penalized
                u_min = u_converged.min().item() if len(u_converged) > 0 else 0.0
                u_max = u_converged.max().item() if len(u_converged) > 0 else 0.0
                print(f"[uncertainty] t={t}, "
                      f"all: [{uncertainty.min().item():.1e}, {uncertainty.max().item():.1e}], "
                      f"converged ({len(u_converged)}/{len(uncertainty)}): "
                      f"[{u_min:.1e}, {u_max:.1e}], "
                      f"non-converged: {n_phi}")
    else:
        # Fallback: proxy uncertainty (project to pixel and sample)
        print("[WARNING] Rasterizer not modified - using proxy uncertainty")
        pts_world = params["means3D"]
        N = pts_world.shape[0]
        ones = torch.ones((N, 1), device=pts_world.device)
        pts_h = torch.cat([pts_world, ones], dim=1)
        w2c = curr_data["w2c"]
        pts_cam = (w2c @ pts_h.T).T
        K = curr_data["K"]
        u = (K[0, 0] * (pts_cam[:, 0] / pts_cam[:, 2].clamp(min=1e-6)) + K[0, 2]).clamp(0, curr_data["width"] - 1).long()
        v = (K[1, 1] * (pts_cam[:, 1] / pts_cam[:, 2].clamp(min=1e-6)) + K[1, 2]).clamp(0, curr_data["height"] - 1).long()
        pred_rgb = im[:, v, u].permute(1, 0)
        gt_rgb = curr_data["im"][:, v, u].permute(1, 0)
        uncertainty = torch.abs(pred_rgb - gt_rgb).mean(dim=1)
    
    if not is_initial_timestep:
        state.curr_uncertainty.append(uncertainty.detach().cpu())

    # one overlay dump for visual sanity check (first iteration of each timestep, camera 0)
    if not is_initial_timestep and len(state.curr_uncertainty) == 1 and curr_id == 0 and t is not None:
        with torch.no_grad():
            centers2d_px = compute_centers2d(params, curr_data)
            rendered_np = im.detach().clamp(0.0, 1.0).cpu().permute(1, 2, 0).numpy()
            overlay_path = f"./output/uncertainty_debug_t{t:02d}_cam{curr_id:02d}.png"
            overlay_uncertainty_on_image_pil(
                image_np=rendered_np,
                centers2d=centers2d_px,
                uncertainty=uncertainty,
                radii_px=radius,
                out_path=overlay_path,
            )
            print(f"[USplat4D DEBUG] Saved uncertainty overlay: {overlay_path}")

    # 3) segmentation render
    segrendervar = params2rendervar(params)
    segrendervar["colors_precomp"] = params["seg_colors"]
    seg_outputs = Renderer(raster_settings=curr_data["cam"])(**segrendervar)
    seg = seg_outputs[0]  # Only need color output
    losses["seg"] = 0.8 * l1_loss_v1(seg, curr_data["seg"]) + 0.2 * (1.0 - calc_ssim(seg, curr_data["seg"]))

    # 4) motion regularization, floor, bg, color consistency
    if not is_initial_timestep:
        is_fg = (params["seg_colors"][:, 0] > 0.5).detach()
        fg_pts = rendervar["means3D"][is_fg]
        fg_rot = rendervar["rotations"][is_fg]

        rel_rot = quat_mult(fg_rot, state.prev_inv_rot_fg)
        rot = build_rotation(rel_rot)
        neighbor_pts = fg_pts[state.neighbor_indices]
        curr_offset = neighbor_pts - fg_pts[:, None]
        curr_offset_in_prev_coord = (
            rot.transpose(2, 1)[:, None] @ curr_offset[:, :, :, None]
        ).squeeze(-1)

        losses["rigid"] = weighted_l2_loss_v2(
            curr_offset_in_prev_coord, state.prev_offset, state.neighbor_weight
        )

        losses["rot"] = weighted_l2_loss_v2(
            rel_rot[state.neighbor_indices], rel_rot[:, None], state.neighbor_weight
        )

        curr_offset_mag = torch.sqrt((curr_offset ** 2).sum(-1) + 1e-20)
        losses["iso"] = weighted_l2_loss_v1(
            curr_offset_mag, state.neighbor_dist, state.neighbor_weight
        )

        losses["floor"] = torch.clamp(fg_pts[:, 1], min=0).mean()

        bg_pts = rendervar["means3D"][~is_fg]
        bg_rot = rendervar["rotations"][~is_fg]
        losses["bg"] = l1_loss_v2(bg_pts, state.init_bg_pts) + l1_loss_v2(
            bg_rot, state.init_bg_rot
        )

        losses["soft_col_cons"] = l1_loss_v2(params["rgb_colors"], state.prev_col)

    # 5) Motion optimization losses (USplat4D §4.3)
    if not is_initial_timestep and len(state.temporal_graph) > 0:
        # Get the latest temporal graph
        graph_dict = state.temporal_graph[-1]
        
        # Key node loss (Eq. 9)
        if hasattr(state, 'init_params') and state.init_params is not None:
            losses["motion_key"] = compute_key_node_loss(
                params=params,
                state=state,
                graph_dict=graph_dict,
                init_params=state.init_params,
                t=len(state.temporal_graph) - 1,
            )
        
        # Non-key node loss (Eq. 11)
        losses["motion_non_key"] = compute_non_key_loss(
            params=params,
            state=state,
            graph_dict=graph_dict,
            init_params=state.init_params,
            t=len(state.temporal_graph) - 1,
        )
        
        # Motion regularization (smooth SE(3) transforms)
        losses["motion_reg"] = compute_motion_regularization_loss(
            params=params,
            state=state,
            graph_dict=graph_dict,
        )
    
    # 6) aggregate losses and update visibility statistics
    loss_weights = {
        "im": 1.0,
        "seg": 3.0,
        "rigid": 4.0,
        "rot": 4.0,
        "iso": 2.0,
        "floor": 2.0,
        "bg": 20.0,
        "soft_col_cons": 0.01,
        "motion_key": 1.0,
        "motion_non_key": 0.5,
        "motion_reg": 0.1,
    }
    loss = sum(loss_weights.get(k, 0) * v for k, v in losses.items())

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
        render_out = Renderer(raster_settings=data['cam'])(**params2rendervar(params))
        im = render_out[0]
        curr_id = data['id']
        im = torch.exp(params['cam_m'][curr_id])[:, None, None] * im + params['cam_c'][curr_id][:, None, None]
        psnr = calc_psnr(im, data['im']).mean()
        progress_bar.set_postfix({"train img 0 PSNR": f"{psnr:.{7}f}"})
        progress_bar.update(every_i)


def train(seq, exp):
    if os.path.exists(f"./output/{exp}/{seq}"):
        print(f"Experiment '{exp}' for sequence '{seq}' already exists. Exiting.")
        return

    md = json.load(open(f"./data/{seq}/train_meta.json", "r"))
    num_timesteps = len(md["fn"])

    # 1) global init
    params, state = initialize_params(seq, md)
    optimizer = initialize_optimizer(params, state)
    output_params = []
    
    # Store initial params for motion optimization reference (§4.3, Eq. 9)
    state.init_params = {k: v.detach().clone() for k, v in params.items()}

    # 2) temporal containers are already in TemporalState by default

    # Detect if monocular or multi-view based on number of cameras
    num_cameras = len(md['fn'][0])  # Number of cameras at timestep 0
    is_monocular = (num_cameras == 1)
    state.is_monocular = is_monocular  # Store in state for later use
    print(f"[init] Detected {'monocular' if is_monocular else 'multi-view'} setup with {num_cameras} cameras")
    
    for t in range(num_timesteps):
        # ---- Stage 1: load dataset for timestep t ----
        dataset = get_dataset(t, md, seq)
        todo_dataset = []
        is_initial_timestep = t == 0

        # ---- Stage 2: reset per-timestep visibility ----
        if not is_initial_timestep:
            N = params["means3D"].shape[0]
            if state.seen_any is None or state.seen_any.shape[0] != N:
                state.seen_any = torch.zeros(N, device="cuda", dtype=torch.bool)
            else:
                state.seen_any.zero_()

        # ---- Stage 3: initialize timestep (temporal prediction) ----
        if not is_initial_timestep:
            params, state = initialize_per_timestep(params, state, optimizer)

        num_iter_per_timestep = 10000 if is_initial_timestep else 2000
        progress_bar = tqdm(range(num_iter_per_timestep), desc=f"timestep {t}")

        # ---- Stage 4: per-iteration optimization ----
        for i in range(num_iter_per_timestep):
            curr_data = get_batch(todo_dataset, dataset)
            loss, state = get_loss(params, curr_data, state, is_initial_timestep, t=t)
            loss.backward()
            with torch.no_grad():
                report_progress(params, dataset[0], i, progress_bar)
                if is_initial_timestep:
                    params, state = densify(params, state, optimizer, i)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        progress_bar.close()

        # ---- Stage 5: summarize uncertainty + key-node selection + temporal graph ----
        if not is_initial_timestep and len(state.curr_uncertainty) > 0:
            u_stack = torch.stack(state.curr_uncertainty, dim=0)  # (num_iters, N)
            u_mean = u_stack.mean(0)                              # (N,)

            update_uncertainty_window(
                state,
                u_mean=u_mean,
                seen_any=state.seen_any,
                window_size=5,  # T_min in paper
            )

            select_key_nodes_from_window(
                t=t,
                params=params,
                state=state,
                T_min=5,
                quantile=0.02,
                voxel_ratio=0.05,
            )

            # Build temporal graph after key-node selection
            # Extract camera rotation from first camera's w2c matrix
            w2c = dataset[0]['w2c']  # [4, 4] world-to-camera
            c2w = torch.inverse(w2c)  # [4, 4] camera-to-world
            camera_rotation = c2w[:3, :3]  # [3, 3] rotation only
            
            build_temporal_graph(
                params=params,
                state=state,
                output_params=output_params,
                t=t,
                num_knn=5,
                camera_rotation=camera_rotation,
                is_monocular=is_monocular,
            )
            
            # Add transform parameters to optimizer for motion optimization
            add_transform_parameters_to_optimizer(optimizer, state, lr=0.001)
            
            # Visualize graph (key vs non-key nodes) on first camera view
            if len(state.temporal_graph) > 0 and t > 1:  # After at least 2 non-initial timesteps
                with torch.no_grad():
                    graph = state.temporal_graph[-1]
                    key_mask_full = torch.zeros(params['means3D'].shape[0], dtype=torch.bool)
                    key_mask_full[graph['key_indices']] = True
                    
                    # Render current frame for visualization
                    rendervar = params2rendervar(params)
                    vis_outputs = Renderer(raster_settings=dataset[0]['cam'])(**rendervar)
                    im_vis, radius_vis = vis_outputs[0], vis_outputs[1]
                    curr_id = dataset[0]['id']
                    im_vis = torch.exp(params['cam_m'][curr_id])[:, None, None] * im_vis + params['cam_c'][curr_id][:, None, None]
                    
                    # Project to 2D
                    pts_world = params["means3D"]
                    N = pts_world.shape[0]
                    ones = torch.ones((N, 1), device=pts_world.device)
                    pts_h = torch.cat([pts_world, ones], dim=1)
                    w2c = dataset[0]["w2c"]
                    pts_cam = (w2c @ pts_h.T).T
                    K = dataset[0]["K"]
                    u = K[0, 0] * (pts_cam[:, 0] / pts_cam[:, 2].clamp(min=1e-6)) + K[0, 2]
                    v = K[1, 1] * (pts_cam[:, 1] / pts_cam[:, 2].clamp(min=1e-6)) + K[1, 2]
                    centers2d = torch.stack([u, v], dim=1)
                    
                    rendered_np = im_vis.detach().clamp(0.0, 1.0).cpu().permute(1, 2, 0).numpy()
                    graph_vis_path = f"./output/graph_vis_t{t:02d}.png"
                    visualize_graph_on_image(
                        image_np=rendered_np,
                        centers2d=centers2d,
                        key_mask=key_mask_full,
                        radii_px=radius_vis,
                        out_path=graph_vis_path,
                    )
                    print(f"[USplat4D] Graph visualization saved: {graph_vis_path}")

        state.curr_uncertainty = []

        # ---- Stage 6: finalize timestep ----
        output_params.append(params2cpu(params, is_initial_timestep))
        if is_initial_timestep:
            state = initialize_post_first_timestep(params, state, optimizer)

    save_params(output_params, seq, exp)

    if len(state.temporal_uncertainty) > 0:
        unc_mat = torch.stack(state.temporal_uncertainty, dim=1)
        os.makedirs(f"./output/{exp}/{seq}", exist_ok=True)
        torch.save(unc_mat, f"./output/{exp}/{seq}/uncertainty_proxy.pt")
        print(
            f"[USplat4D] Saved proxy temporal uncertainty to "
            f"./output/{exp}/{seq}/uncertainty_proxy.pt"
        )


if __name__ == "__main__":
    exp_name = "exp_motion_opt_test"
    datasets = ["basketball"]
    for sequence in datasets:
        train(sequence, exp_name)
        torch.cuda.empty_cache()
