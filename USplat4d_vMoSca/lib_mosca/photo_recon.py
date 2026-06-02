# from origial reconstruction.py import all rendering related stuff
import sys, os, os.path as osp
import torch
import logging
from tqdm import tqdm
from omegaconf import OmegaConf
from misc import configure_logging, get_timestamp
import numpy as np
import imageio
import matplotlib.pyplot as plt
from matplotlib import cm
import kornia
import colorsys
import torch.nn.functional as F
import roma
import time
import wandb
import random
import gc

sys.path.append(osp.dirname(osp.abspath(__file__)))

from pytorch3d.ops import knn_points
from dynamic_gs import DynSCFGaussian
from static_gs import StaticGaussian
from mosca import MoSca
from camera import MonocularCameras
from lib_ugraph.ugraph_gs import UGraphGaussian
from lib_ugraph.ugraph_model4 import get_graph_rigidity_interpolation_loss, get_DQB_nonkeyGaussian, get_local_ridigity_loss # graph_model3/4
from lib_ugraph.loss_dy import get_dynamic_loss, get_dynamic_loss_by_input
from photo_recon_utils import fetch_leaves_in_world_frame, estimate_normal_map
from lib_render.render_helper import GS_BACKEND, render, RGB2SH
from lib_prior.prior_loading import Saved2D
from gs_utils.gs_optim_helper import update_learning_rate
from gs_utils.loss_helper import (
    compute_rgb_loss,
    compute_dep_loss,
    compute_normal_loss,
    compute_normal_reg_loss,
    compute_dep_reg_loss,
)
from dynamic_solver import get_dynamic_curves

from photo_recon_utils import (
    fetch_leaves_in_world_frame,
    GSControlCFG,
    OptimCFG,
    apply_gs_control_s,
    apply_gs_control_d,
    error_grow_dyn_model,
    identify_traj_id,
    get_colorplate,
)
from photo_recon_viz_utils import (
    viz_hist,
    viz_dyn_hist,
    viz2d_total_video,
    viz3d_total_video,
    viz_curve,
)

from nerfview import CameraState
from lib_usplat4d_viewer.usplat4d_viewer import DynamicViewer
from lib_usplat4d_viewer.usplat4d_viser import get_server
from lib_usplat4d_viewer.usplat4d_plot import (
    plot_camera_trajectories, 
    set_axes_equal, 
    plot_trajectories_plotly, 
    plot_camera_frustums, 
)
from lib_usplat4d_viewer.usplat4d_utils import (
    test_camera_distance, 
    test_camera_angle,
    find_global_rotation
)

def get_recon_cfg(cfg_fn=None):
    if cfg_fn is None:
        logging.info("No cfg_fn provided, use dummy cfg")
        # get dummy
        cfg = OmegaConf.create()
        cfg.s_ctrl = OmegaConf.create()
        cfg.d_ctrl = OmegaConf.create()
        return cfg
    cfg = OmegaConf.load(cfg_fn)
    for key in ["photometric", "geometric"]:
        if not hasattr(cfg, key):
            setattr(cfg, key, OmegaConf.create())
    for key in ["d_ctrl", "s_ctrl"]:
        if not hasattr(cfg.photometric, key):
            setattr(cfg.photometric, key, None)
    OmegaConf.set_readonly(cfg, True)
    logging.info(f"Load cfg from {cfg_fn}: {cfg}")
    return cfg


class DynReconstructionSolver:
    def __init__(
        self,
        working_dir,
        dir_saving_this_run_model,
        device=torch.device("cuda:0"),
        # cfg
        radius_init_factor=4.0,  # ! 1.0
        opacity_init_factor=0.95,
    ):
        self.src = working_dir
        self.device = device
        timestamp = get_timestamp()
        configure_logging(
            osp.join(self.src, f"dynamic_reconstruction_{timestamp}.log"),
            debug=False,
        )

        self.log_dir = self.src
        self.viz_dir = osp.join(self.src, f"mosca_photo_viz_{timestamp}")
        os.makedirs(self.viz_dir, exist_ok=True)

        self.radius_init_factor = radius_init_factor
        self.opacity_init_factor = opacity_init_factor

        self.dir_saving_this_run_model=dir_saving_this_run_model

        return

    @torch.no_grad()
    def identify_fg_mask_by_nearest_curve(
        self, s2d: Saved2D, cams: MonocularCameras, viz_fname=None
    ):
        # Use GT masks if available, otherwise use curve-based method
        if s2d.has_gt_masks:
            logging.info("Using GT masks for foreground identification")
            fg_mask_list = s2d.gt_masks.clone()
            if viz_fname is not None:
                viz_rgb = s2d.rgb.clone().cpu()
                viz_fg_mask_list = fg_mask_list * s2d.dep_mask.to(fg_mask_list)
                viz_rgb = viz_rgb * viz_fg_mask_list.float()[..., None] + viz_rgb * 0.1 * (
                    1 - viz_fg_mask_list.float()[..., None]
                )
                imageio.mimsave(osp.join(self.viz_dir, viz_fname), viz_rgb.numpy())
            
            fg_mask_list = fg_mask_list.to(cams.rel_focal.device).bool()
            s2d.register_2d_identification(
                static_2d_mask=~fg_mask_list, dynamic_2d_mask=fg_mask_list
            )
            return fg_mask_list
        
        # Original curve-based method
        # get global anchor
        curve_xyz, curve_mask, _, _ = get_dynamic_curves(
            s2d, cams, return_all_curves=True
        )
        assert curve_xyz.shape[1] == len(s2d.dynamic_track_mask)

        # only consider the valid case
        static_curve_mean = (
            curve_xyz[:, ~s2d.dynamic_track_mask]
            * curve_mask[:, ~s2d.dynamic_track_mask, None]
        ).sum(0, keepdim=True) / curve_mask[:, ~s2d.dynamic_track_mask, None].sum(
            0, keepdim=True
        ).expand(
            len(curve_xyz), -1, -1
        )
        curve_xyz[:, ~s2d.dynamic_track_mask] = static_curve_mean
        np.savetxt(
            osp.join(self.viz_dir, "fg_id_non_dyn_curve_meaned.xyz"),
            static_curve_mean.reshape(-1, 3).cpu().numpy(),
            fmt="%.4f",
        )

        with torch.no_grad():
            fg_mask_list = []
            for query_tid in tqdm(range(s2d.T)):
                query_dep = s2d.dep[query_tid].clone()  # H,W
                query_xyz_cam = cams.backproject(
                    cams.get_homo_coordinate_map(), query_dep
                )
                query_xyz_world = cams.trans_pts_to_world(
                    query_tid, query_xyz_cam
                )  # H,W,3

                # find the nearest distance and acc sk weight
                # use all the curve at this position to id the fg and bg
                _, knn_id, _ = knn_points(
                    query_xyz_world.reshape(1, -1, 3), curve_xyz[query_tid][None], K=1
                )
                knn_id = knn_id[0, :, 0]
                fg_mask = s2d.dynamic_track_mask[knn_id].reshape(s2d.H, s2d.W)
                fg_mask_list.append(fg_mask.cpu())
            fg_mask_list = torch.stack(fg_mask_list, 0)
        if viz_fname is not None:
            viz_rgb = s2d.rgb.clone().cpu()
            viz_fg_mask_list = fg_mask_list * s2d.dep_mask.to(fg_mask_list)
            viz_rgb = viz_rgb * viz_fg_mask_list.float()[..., None] + viz_rgb * 0.1 * (
                1 - viz_fg_mask_list.float()[..., None]
            )
            imageio.mimsave(osp.join(self.viz_dir, viz_fname), viz_rgb.numpy())

        fg_mask_list = fg_mask_list.to(cams.rel_focal.device).bool()
        s2d.register_2d_identification(
            static_2d_mask=~fg_mask_list, dynamic_2d_mask=fg_mask_list
        )
        return fg_mask_list

    @torch.no_grad()
    def compute_normals_for_s2d(
        self,
        s2d,
        cams,
        patch_size=7,
        nn_dist_th=0.03,
        nn_min_cnt=4,
        viz_fn=None,
        viz_subsample=4,
    ):
        # compute normal maps for s2d
        logging.info(f"Computing normal maps from depth maps using local SVD...")
        # the computed normals are always pointing backward, on -z direction
        ray_direction = cams.backproject(
            s2d.homo_map.clone(), torch.ones_like(s2d.homo_map[..., 0])
        )
        ray_direction = F.normalize(ray_direction, dim=-1)
        normal_map_list = []
        new_mask_list = []
        for t in tqdm(range(s2d.T)):
            dep = s2d.dep[t].clone()
            dep_mask = s2d.dep_mask[t].clone()
            normal_map = torch.zeros(*dep.shape, 3).to(dep)
            xyz = cams.backproject(s2d.homo_map[dep_mask].clone(), dep[dep_mask])
            vtx_map = torch.zeros_like(normal_map).float()
            vtx_map[dep_mask] = xyz
            normal_map, mask = estimate_normal_map(
                vtx_map, dep_mask, patch_size, nn_dist_th, nn_min_cnt
            )

            normal = normal_map[mask]
            inner = (normal * ray_direction[mask]).sum(-1)
            correct_orient = inner < 0
            sign = torch.ones_like(normal[..., :1])
            sign[~correct_orient] = -1.0
            normal = normal.clone() * sign
            normal_map[mask] = normal
            new_mask = dep_mask * mask
            normal_map_list.append(normal_map)
            new_mask_list.append(new_mask)
        ret_nrm = torch.stack(normal_map_list, 0)
        ret_mask = torch.stack(new_mask_list, 0)

        if viz_fn is not None:
            viz_fn = osp.join(self.viz_dir, viz_fn)
            logging.info(f"Viz normal maps to {viz_fn}")
            viz_frames = (
                ((-ret_nrm + 1) / 2.0 * 255).detach().cpu().numpy().astype(np.uint8)
            )
            if len(viz_frames) > 50:
                _step = max(1, len(viz_frames) // 50)
            else:
                _step = 1
            # skip to boost viz
            viz_frames = viz_frames[:, ::viz_subsample, ::viz_subsample, :]
            imageio.mimsave(viz_fn, viz_frames[::_step])

        s2d.register_buffer("nrm", ret_nrm.detach().clone())
        s2d.dep_mask = ret_mask.detach().clone()
        return

    @torch.no_grad()
    def get_static_model(
        self,
        s2d: Saved2D,
        cams,
        n_init=30000,
        radius_max=0.05,
        max_sph_order=0,
        image_stride=1,
        viz_fn=None,
        mask_type="static_depth",
    ):
        device = self.device

        if mask_type == "static_depth":
            gather_mask = s2d.sta_mask * s2d.dep_mask
        elif mask_type == "depth":
            gather_mask = s2d.dep_mask
        else:
            raise ValueError(f"Unknown mask_type={mask_type}")

        mu_init, q_init, s_init, o_init, rgb_init, id_init, _ = (
            fetch_leaves_in_world_frame(
                cams=cams,
                n_attach=n_init,
                input_mask_list=gather_mask,
                input_dep_list=s2d.dep,
                input_rgb_list=s2d.rgb,
                save_xyz_fn=viz_fn,
                subsample=image_stride,
            )
        )
        s_model: StaticGaussian = StaticGaussian(
            init_mean=mu_init.clone().to(device),
            init_q=q_init,
            init_s=s_init * self.radius_init_factor,
            init_o=o_init * self.opacity_init_factor,
            init_rgb=rgb_init,
            init_id=id_init,
            max_scale=radius_max,
            min_scale=0.0,
            max_sph_order=max_sph_order,
        )
        s_model.to(device)
        return s_model

    @torch.no_grad()
    def get_dynamic_model(
        self,
        s2d: Saved2D,
        cams: MonocularCameras,
        scf: MoSca,
        n_init=10000,
        image_stride=1,
        radius_max=0.05,
        max_sph_order=0,
        leaf_local_flag=True,
        topo_th_ratio=None,
        dyn_o_flag=True,
        additional_mask=None,
        nn_fusion=-1,
        max_node_num=100000,
    ):
        device = self.device
        collect_t_list = torch.arange(0, s2d.T, 1)
        logging.info(f"Collect GS at t={collect_t_list}")
        input_mask_list = s2d.dyn_mask * s2d.dep_mask
        if additional_mask is not None:
            assert additional_mask.shape == s2d.dep_mask.shape
            input_mask_list = input_mask_list * additional_mask
        mu_init, q_init, s_init, o_init, rgb_init, id_init, time_init = (
            fetch_leaves_in_world_frame(
                cams=cams,
                n_attach=n_init,
                input_mask_list=input_mask_list,
                input_dep_list=s2d.dep,
                input_rgb_list=s2d.rgb,
                subsample=image_stride,
            )
        )
        # * Reset SCF topo th!
        if topo_th_ratio is not None:
            old_th_ratio = scf.topo_th_ratio
            scf.topo_th_ratio = torch.ones_like(scf.topo_th_ratio) * topo_th_ratio
            logging.info(
                f"Reset SCF topo th ratio from {old_th_ratio} to {scf.topo_th_ratio}"
            )

        # * Init the scf
        d_model: DynSCFGaussian = DynSCFGaussian(
            scf=scf,
            max_scale=radius_max,
            min_scale=0.0,
            max_sph_order=max_sph_order,
            device=device,
            leaf_local_flag=leaf_local_flag,
            dyn_o_flag=dyn_o_flag,
            nn_fusion=nn_fusion,
            max_node_num=max_node_num,
        )
        d_model.to(device)

        # * Init the leaves
        optimizer = torch.optim.Adam(d_model.get_optimizable_list())
        unique_tid = time_init.unique()
        logging.info("Attach to Dynamic Scaffold ...")
        for tid in tqdm(unique_tid):
            t_mask = time_init == tid
            d_model.append_new_gs(
                optimizer,
                tid=tid,
                mu_w=mu_init[t_mask],
                quat_w=q_init[t_mask],
                scales=s_init[t_mask] * self.radius_init_factor,
                opacity=o_init[t_mask] * self.opacity_init_factor,
                rgb=rgb_init[t_mask],
            )
        # d_model.scf.update_topology()
        d_model.summary()
        return d_model
    def get_graph_model(self,
            state_dict,
            graph_dir,
            device,
        ):
        if graph_dir is None:
            return None
        else:
            ugraph_model: UGraphGaussian = UGraphGaussian.init_from_state_dict(state_dict, graph_dir, device)

        return ugraph_model
    def photometric_fit(
        self,
        s2d: Saved2D,
        cams: MonocularCameras,
        s_model: StaticGaussian,
        d_model: DynSCFGaussian = None,
        ugraph_model: UGraphGaussian = None,
        optim_cam_after_steps=0,
        total_steps=8000,
        topo_update_feq=50,
        skinning_corr_start_steps=1e10,
        s_gs_ctrl_cfg: GSControlCFG = GSControlCFG(
            densify_steps=400,
            reset_steps=2000,
            prune_steps=400,
            densify_max_grad=0.00025,
            densify_percent_dense=0.01,
            prune_opacity_th=0.012,
            reset_opacity=0.01,
        ),
        d_gs_ctrl_cfg: GSControlCFG = GSControlCFG(
            densify_steps=400,
            reset_steps=2000,
            prune_steps=400,
            densify_max_grad=0.00015,
            densify_percent_dense=0.01,
            prune_opacity_th=0.012,
            reset_opacity=0.01,
        ),
        s_gs_ctrl_start_ratio=0.2,
        s_gs_ctrl_end_ratio=0.9,
        d_gs_ctrl_start_ratio=0.2,
        d_gs_ctrl_end_ratio=0.9,
        # optim
        optimizer_cfg: OptimCFG = OptimCFG(
            lr_cam_f=0.0,
            lr_cam_q=0.0001,
            lr_cam_t=0.0001,
            lr_p=0.0003,
            lr_q=0.002,
            lr_s=0.01,
            lr_o=0.1,
            lr_sph=0.005,
            # dyn
            lr_np=0.001,
            lr_nq=0.01,
            lr_w=0.3,
        ),
        # cfg loss
        lambda_rgb=1.0,
        lambda_dep=1.0,
        lambda_mask=0.5,
        dep_st_invariant=True,
        lambda_normal=1.0,
        lambda_depth_normal=0.05,  # from GOF
        lambda_distortion=100.0,  # from GOF
        lambda_arap_coord=3.0,
        lambda_arap_len=0.0,
        lambda_vel_xyz_reg=0.0,
        lambda_vel_rot_reg=0.0,
        lambda_acc_xyz_reg=0.5,
        lambda_acc_rot_reg=0.5,
        lambda_small_w_reg=0.0,
        #
        lambda_track=0.0,
        track_flow_chance=0.0,
        track_flow_interval_candidates=[1],
        track_loss_clamp=100.0,
        track_loss_protect_steps=100,
        track_loss_interval=3,  # 1/3 steps are used for track loss and does not count the grad
        track_loss_start_step=-1,
        track_loss_end_step=100000,
        ######
        reg_radius=None,
        geo_reg_start_steps=0,
        viz_interval=1000,
        viz_cheap_interval=1000,
        viz_skip_t=5,
        viz_move_angle_deg=30.0,
        phase_name="photometric",
        use_decay=True,
        decay_start=2000,
        temporal_diff_shift=[1, 3, 6],
        temporal_diff_weight=[0.6, 0.3, 0.1],
        # * error grow
        dyn_error_grow_steps=[],
        dyn_error_grow_th=0.2,
        dyn_error_grow_num_frames=4,
        dyn_error_grow_subsample=1,
        # ! warning, the corr loss is in pixel int unit!!
        dyn_node_densify_steps=[],
        dyn_node_densify_grad_th=0.2,
        dyn_node_densify_record_start_steps=2000,
        dyn_node_densify_max_gs_per_new_node=100000,
        # * scf pruning
        dyn_scf_prune_steps=[],
        dyn_scf_prune_sk_th=0.02,
        # other
        random_bg=False,
        default_bg_color=[1.0, 1.0, 1.0],
        # * DynGS cleaning
        photo_s2d_trans_steps=[],
    ):
        logging.info(f"Finetune with GS-BACKEND={GS_BACKEND.lower()}")

        torch.cuda.empty_cache()
        n_frame = 8 # 1 #8

        d_flag = d_model is not None
        ugraph_flag = ugraph_model is not None

        def checkmark(flag):
            return "✅" if flag else "❌"
        
        print(f"d_model     : {checkmark(d_flag)}")
        print(f"ugraph_model: {checkmark(ugraph_flag)}")

        corr_flag = lambda_track > 0.0 and d_flag
        if corr_flag:
            logging.info(
                f"Enabel Flow/Track backing with supervision interval={track_loss_interval}"
            )

        optimizer_static = torch.optim.Adam(
            s_model.get_optimizable_list(**optimizer_cfg.get_static_lr_dict)
        )
        if d_flag:
            optimizer_dynamic = torch.optim.Adam(
                d_model.get_optimizable_list(**optimizer_cfg.get_dynamic_lr_dict)
            )
            if reg_radius is None:
                reg_radius = int(np.array(temporal_diff_shift).max()) * 2
            logging.info(f"Set reg_radius={reg_radius} for dynamic model")
            sup_mask_type = "all"
        else:
            sup_mask_type = "static"
        cam_param_list = cams.get_optimizable_list(**optimizer_cfg.get_cam_lr_dict)[:2]
        if len(cam_param_list) > 0:
            optimizer_cam = torch.optim.Adam(
                cams.get_optimizable_list(**optimizer_cfg.get_cam_lr_dict)[:2]
            )
        else:
            optimizer_cam = None
        if ugraph_flag:
            # optimizer_ugraph = torch.optim.Adam(
            #     ugraph_model.get_optimizable_list(**optimizer_cfg.get_ugraph_lr_dict)
            # )
            optimizers_ugraph, schedulers_ugraph = ugraph_model.configure_optimizers(optimizer_cfg.get_ugraph_lr_dict)
        else:
            # optimizer_ugraph = None
            optimizers_ugraph = None
            schedulers_ugraph = None
        if use_decay:
            gs_scheduling_func_dict, cam_scheduling_func_dict = (
                optimizer_cfg.get_scheduler(total_steps=total_steps - decay_start)
            )
        else:
            gs_scheduling_func_dict, cam_scheduling_func_dict = {}, {}
            # ugraph_scheduling_func_dict = {}

        loss_rgb_list, loss_dep_list, loss_nrm_list = [], [], []
        loss_fg_mask_list = []
        loss_mask_list = []
        loss_dep_nrm_reg_list, loss_distortion_reg_list = [], []

        loss_arap_coord_list, loss_arap_len_list = [], []
        loss_vel_xyz_reg_list, loss_vel_rot_reg_list = [], []
        loss_acc_xyz_reg_list, loss_acc_rot_reg_list = [], []
        s_n_count_list, d_n_count_list = [], []
        d_m_count_list = []
        loss_sds_list = []
        loss_small_w_list = []
        loss_track_list = []

        s_gs_ctrl_start = int(total_steps * s_gs_ctrl_start_ratio)
        d_gs_ctrl_start = int(total_steps * d_gs_ctrl_start_ratio)
        s_gs_ctrl_end = int(total_steps * s_gs_ctrl_end_ratio)
        d_gs_ctrl_end = int(total_steps * d_gs_ctrl_end_ratio)
        assert s_gs_ctrl_start >= 0
        assert d_gs_ctrl_start >= 0

        latest_track_event = 0
        base_u, base_v = np.meshgrid(np.arange(s2d.W), np.arange(s2d.H))
        base_uv = np.stack([base_u, base_v], -1)
        base_uv = torch.tensor(base_uv, device=s2d.rgb.device).long()

        # prepare a color-plate for seamntic rendering
        if d_flag:
            # ! for now the group rendering only works for dynamic joitn mode
            n_group_static = len(s_model.group_id.unique())
            n_group_dynamic = len(d_model.scf.unique_grouping)
            color_plate = get_colorplate(n_group_static + n_group_dynamic)
            # random permute
            color_permute = torch.randperm(len(color_plate))
            color_plate = color_plate[color_permute]
            s_model.get_cate_color(
                color_plate=color_plate[:n_group_static].to(s_model.device)
            )
            d_model.get_cate_color(
                color_plate=color_plate[n_group_static:].to(d_model.device)
            )

        last_step = 0
        self.s_model=s_model
        self.d_model=d_model
        self.ugraph_model=ugraph_model
        self.d_model.set_ugraph(ugraph_model) ## if ugraph_model preseted, this is not necessary.
        
        self.start_viewer(num_frames=cams.T,work_dir=self.log_dir)
        self.init_wandb(log_dir=self.log_dir)
        def infinite_batches(N, batch_size):
            indices = list(range(N))
            while True:  # loop over epochs implicitly
                random.shuffle(indices)
                for i in range(0, N, batch_size):
                    yield indices[i:i + batch_size]

        batch_generator = infinite_batches(N=cams.T, batch_size=n_frame)
        for step in (
            pbar := tqdm(range(total_steps))
        ):
            if self.viewer is not None:
                while self.viewer.state.status == "paused": # != 
                    time.sleep(0.1)
                # self.viewer.lock.acquire()

            # # * control the w correction
            # if d_flag and step == skinning_corr_start_steps:
            #     logging.info(
            #         f"at {step} stop all the topology update and add skinning weight correction"
            #     )
            #     d_model.set_surface_deform()
            corr_exe_flag = (
                corr_flag
                and step > latest_track_event + track_loss_protect_steps
                and step % track_loss_interval == 0
                and step >= track_loss_start_step
                and step < track_loss_end_step
            )
            optimizer_static.zero_grad()
            if optimizer_cam is not None:
                optimizer_cam.zero_grad()
            cams.zero_grad()
            s_model.zero_grad()
            s2d.zero_grad()
            if d_flag:
                optimizer_dynamic.zero_grad()
                d_model.zero_grad()
                # if step % topo_update_feq == 0:
                #     d_model.scf.update_topology()
            if ugraph_flag:
                # optimizer_ugraph.zero_grad()
                for opt in optimizers_ugraph.values():
                    # opt.step()
                    opt.zero_grad(set_to_none=True)
                

            if step > decay_start:
                for k, v in gs_scheduling_func_dict.items():
                    update_learning_rate(v(step), k, optimizer_static)
                    if d_flag:
                        update_learning_rate(v(step), k, optimizer_dynamic)
                if optimizer_cam is not None:
                    for k, v in cam_scheduling_func_dict.items():
                        update_learning_rate(v(step), k, optimizer_cam)
            if ugraph_flag: # CHECK TODO
                # for k, v in ugraph_scheduling_func_dict.items():
                #     update_learning_rate(v(step), k, optimizer_ugraph)
                # for k, v in schedulers_ugraph.items():
                for sched in schedulers_ugraph.values():
                    sched.step() # TODO: how to input step value
                        

            # view_ind_list = np.random.choice(cams.T, n_frame, replace=False).tolist()
            view_ind_list = next(batch_generator)
            if corr_exe_flag:
                # select another ind different than the view_ind_list
                corr_dst_ind_list, corr_flow_flag_list = [], []
                for view_ind in view_ind_list:
                    flow_flag = np.random.rand() < track_flow_chance
                    corr_flow_flag_list.append(flow_flag)
                    if flow_flag:
                        corr_dst_ind_candidates = []
                        for flow_interval in track_flow_interval_candidates:
                            if view_ind + flow_interval < cams.T:
                                corr_dst_ind_candidates.append(view_ind + flow_interval)
                            if view_ind - flow_interval >= 0:
                                corr_dst_ind_candidates.append(view_ind - flow_interval)
                        corr_dst_ind = np.random.choice(corr_dst_ind_candidates)
                        corr_dst_ind_list.append(corr_dst_ind)
                    else:
                        corr_dst_ind = view_ind
                        while corr_dst_ind == view_ind:
                            corr_dst_ind = np.random.choice(cams.T)
                        corr_dst_ind_list.append(corr_dst_ind)
                corr_dst_ind_list = np.array(corr_dst_ind_list)
            else:
                corr_dst_ind_list = view_ind_list
                corr_flow_flag_list = [False] * n_frame


            # prepare DQB nonkeyGaussians
            if ugraph_flag:
                use_DQB_directly=False
                ts=torch.tensor(view_ind_list,device=d_model.device)
                if use_DQB_directly:
                    transls_trainable_ts, oris_wxyz_trainable_ts=get_DQB_nonkeyGaussian(ugraph_model.precompute,ts)
                else:
                    transls_trainable_ts=ugraph_model.precompute.params['transls'][ts]
                    oris_wxyz_trainable_ts=ugraph_model.precompute.params['oris_wxyz'][ts]
            # end prepare
            render_dict_list, corr_render_dict_list = [], []
            loss_rgb, loss_dep, loss_nrm = 0.0, 0.0, 0.0
            loss_dep_nrm_reg, loss_distortion_reg = 0.0, 0.0
            loss_mask = 0.0
            loss_track = 0.0
            loss_fg_mask = 0.0
            for _inner_loop_i, view_ind in enumerate(view_ind_list):
                dst_ind = corr_dst_ind_list[_inner_loop_i]
                flow_flag = corr_flow_flag_list[_inner_loop_i]
                gs5 = [list(s_model())]

                add_buffer = None
                if corr_exe_flag:
                    # ! detach bg pts
                    dst_xyz = torch.cat([gs5[0][0].detach(), d_model(dst_ind)[0]], 0)
                    dst_xyz_cam = cams.trans_pts_to_cam(dst_ind, dst_xyz)
                    if GS_BACKEND in ["native_add3"]:
                        add_buffer = dst_xyz_cam

                if d_flag:
                    gs5.append(list(d_model(view_ind)))
                    # if ugraph_flag == False:
                    #     gs5.append(list(d_model(view_ind)))
                    # else:
                    #     # ugraph replace mu and fr
                    #     mu, fr, s, o, sph = list(d_model(view_ind))
                    #     transls_t=transls_trainable_ts[_inner_loop_i]
                    #     oris_wxyz_t=oris_wxyz_trainable_ts[_inner_loop_i]
                    #     # transls_t=ugraph_model.precompute.params['transls'][view_ind] # old
                    #     # oris_wxyz_t=ugraph_model.precompute.params['oris_wxyz'][view_ind] # old
                    #     oris_wxyz_t = F.normalize(oris_wxyz_t, p=2, dim=-1)
                    #     # oris_xyzw_t=roma.quat_xyzw_to_wxyz(oris_wxyz_t) # [BUG]
                    #     oris_xyzw_t=roma.quat_wxyz_to_xyzw(oris_wxyz_t)
                    #     rotmat_t=roma.unitquat_to_rotmat(oris_xyzw_t)
                    #     mu_ugraph=transls_t
                    #     fr_ugraph=rotmat_t
                    #     gs5.append([mu_ugraph, fr_ugraph, s, o, sph])
                    #     # gs5.append([mu, fr, s, o, sph])

                if random_bg:
                    bg_color = np.random.rand(3).tolist()
                else:
                    bg_color = default_bg_color  # [1.0, 1.0, 1.0]
                if GS_BACKEND in ["natie_add3"]:
                    # the render internally has another protection, because if not set, the grad has bug
                    bg_color += [0.0, 0.0, 0.0]

                render_dict = render(
                    gs5,
                    s2d.H,
                    s2d.W,
                    cams.K(s2d.H, s2d.W),
                    cams.T_cw(view_ind),
                    bg_color=bg_color,
                    add_buffer=add_buffer,
                )
                render_dict_list.append(render_dict)

                # compute losses
                rgb_sup_mask = s2d.get_mask_by_key(sup_mask_type)[view_ind]
                _l_rgb, _, _, _ = compute_rgb_loss(
                    s2d.rgb[view_ind].detach().clone(), render_dict, rgb_sup_mask
                )
                dep_sup_mask = rgb_sup_mask * s2d.dep_mask[view_ind]
                _l_dep, _, _, _ = compute_dep_loss(
                    s2d.dep[view_ind].detach().clone(),
                    render_dict,
                    dep_sup_mask,
                    st_invariant=dep_st_invariant,
                )
                loss_rgb = loss_rgb + _l_rgb
                loss_dep = loss_dep + _l_dep

                # Optionally, compute foreground mask loss
                # if d_flag and s2d.has_gt_masks:
                #     render_dict_fg = render(
                #         list(d_model(view_ind)),
                #         s2d.H,
                #         s2d.W,
                #         cams.K(s2d.H, s2d.W),
                #         cams.T_cw(view_ind),
                #         bg_color=bg_color,
                #         add_buffer=None,
                #     )
                #     fg_mask_t = s2d.dyn_mask[view_ind].unsqueeze(0)  # [1,H,W]
                #     _l_fg_mask = torch.nn.functional.l1_loss(render_dict_fg["alpha"], fg_mask_t)
                #     loss_fg_mask = loss_fg_mask + _l_fg_mask
                loss_fg_mask = torch.zeros_like(loss_rgb)

                if corr_exe_flag:
                    # * Track Loss
                    if GS_BACKEND in ["native_add3"]:
                        corr_render_dict = render_dict
                        rendered_xyz_map = render_dict["buf"].permute(1, 2, 0)  # H,W,3
                    else:
                        corr_render_dict = render(
                            # # ! use detached bg gs
                            # [[it.detach() for it in gs5[0]], gs5[1]],
                            # ! debug, align wiht .bck old version
                            gs5,
                            s2d.H,
                            s2d.W,
                            cams.K(s2d.H, s2d.W),
                            cams.T_cw(view_ind),
                            bg_color=[0.0, 0.0, 0.0],
                            colors_precomp=dst_xyz_cam,
                        )
                        rendered_xyz_map = corr_render_dict["rgb"].permute(
                            1, 2, 0
                        )  # H,W,3
                    corr_render_dict_list.append(corr_render_dict)
                    # get the flow
                    with torch.no_grad():
                        if flow_flag:
                            flow_ind = s2d.flow_ij_to_listind_dict[(view_ind, dst_ind)]
                            flow = s2d.flow[flow_ind].detach().clone()
                            flow_mask = s2d.flow_mask[flow_ind].detach().clone().bool()
                            track_src = base_uv.clone().detach()[flow_mask]
                            flow = flow[flow_mask]
                            track_dst = track_src.float() + flow
                        else:
                            # contruct target by track
                            track_valid = (
                                s2d.track_mask[view_ind] & s2d.track_mask[dst_ind]
                            )
                            track_src = s2d.track[view_ind][track_valid][..., :2]
                            track_dst = s2d.track[dst_ind][track_valid][..., :2]
                        src_fetch_index = (
                            track_src[:, 1].long() * s2d.W + track_src[:, 0].long()
                        )
                    if len(track_src) == 0:
                        _loss_track = torch.zeros_like(_l_rgb)
                    else:
                        warped_xyz_cam = rendered_xyz_map.reshape(-1, 3)[
                            src_fetch_index
                        ]
                        # filter the pred, only add loss to points that are infront of the camera
                        track_loss_mask = warped_xyz_cam[:, 2] > 1e-4
                        if track_loss_mask.sum() == 0:
                            _loss_track = torch.zeros_like(_l_rgb)
                        else:
                            pred_track_dst = cams.project(warped_xyz_cam)
                            L = min(s2d.W, s2d.H)
                            pred_track_dst[:, :1] = (
                                (pred_track_dst[:, :1] + s2d.W / L) / 2.0 * L
                            )
                            pred_track_dst[:, 1:] = (
                                (pred_track_dst[:, 1:] + s2d.H / L) / 2.0 * L
                            )
                            _loss_track = (pred_track_dst - track_dst).norm(dim=-1)[
                                track_loss_mask
                            ]
                            _loss_track = torch.clamp(
                                _loss_track, 0.0, track_loss_clamp
                            )
                            _loss_track = _loss_track.mean()
                else:
                    _loss_track = torch.zeros_like(_l_rgb)
                loss_track = loss_track + _loss_track

                # * GOF normal and regularization
                if GS_BACKEND == "gof":
                    _l_nrm, _, _, _ = compute_normal_loss(
                        s2d.nrm[view_ind].detach().clone(), render_dict, dep_sup_mask
                    )
                    loss_nrm = loss_nrm + _l_nrm
                    if step > geo_reg_start_steps:
                        _l_reg_nrm, _, _, _ = compute_normal_reg_loss(
                            s2d, cams, render_dict
                        )
                        _l_reg_distortion, _ = compute_dep_reg_loss(
                            s2d.rgb[view_ind].detach().clone(), render_dict
                        )
                    else:
                        _l_reg_nrm = torch.zeros_like(_l_rgb)
                        _l_reg_distortion = torch.zeros_like(_l_rgb)
                    loss_dep_nrm_reg = loss_dep_nrm_reg + _l_reg_nrm
                    loss_distortion_reg = loss_distortion_reg + _l_reg_distortion
                else:
                    loss_nrm = torch.zeros_like(loss_rgb)
                    loss_dep_nrm_reg = torch.zeros_like(loss_rgb)
                    loss_distortion_reg = torch.zeros_like(loss_rgb)

                ############
                if d_flag and lambda_mask > 0.0:
                    # * do the mask loss, including the background
                    s_cate_sph, s_gid2color = s_model.get_cate_color(
                        perm=torch.randperm(len(s_model.group_id.unique()))
                    )
                    d_cate_sph, d_gid2color = d_model.get_cate_color(
                        perm=torch.randperm(len(d_model.scf.unique_grouping))
                    )
                    with torch.no_grad():
                        inst_map = s2d.inst[view_ind]
                        gt_mask = torch.zeros_like(s2d.rgb[0])
                        for gid, color in d_gid2color.items():
                            gt_mask[inst_map == gid] = color[None]
                        for gid, color in s_gid2color.items():
                            gt_mask[inst_map == gid] = color[None]
                    gs5[1][-1] = d_cate_sph
                    gs5[0][-1] = s_cate_sph
                    render_dict = render(
                        gs5,
                        s2d.H,
                        s2d.W,
                        cams.K(s2d.H, s2d.W),
                        cams.T_cw(view_ind),
                        bg_color=[0.0, 0.0, 0.0],
                    )
                    pred_mask = render_dict["rgb"].permute(1, 2, 0)
                    l_mask = torch.nn.functional.mse_loss(pred_mask, gt_mask)
                    loss_mask = loss_mask + l_mask
                else:
                    loss_mask = torch.zeros_like(loss_rgb)

            if True:
                loss_arap_coord = torch.zeros_like(loss_rgb)
                loss_arap_len = torch.zeros_like(loss_rgb)

            if True:
                loss_vel_xyz_reg = loss_vel_rot_reg = loss_acc_xyz_reg = (
                    loss_acc_rot_reg
                ) = torch.zeros_like(loss_rgb)

            if True:
                loss_small_w = torch.zeros_like(loss_rgb)

            ##############################################################
            # ugraph_model
            if ugraph_flag and d_flag:
                ts=torch.tensor(view_ind_list,device=d_model.device)
                # is_skip_nonkey_graph=use_DQB_directly # use DQB result for training directly
                # if step%3==0:
                #     is_skip_key_graph=False
                # else:
                #     is_skip_key_graph=True
                # # if step>last_step+10:
                # #     is_skip_key_graph=False
                # #     last_step=step
                # # else:
                # #     is_skip_key_graph=True
                # is_skip_key_graph=False
                # graph_rigidity_interpolation_loss, graph_rigidity_interpolation_stats=get_graph_rigidity_interpolation_loss(ugraph_model.precompute,ts,is_skip_key_graph,is_skip_nonkey_graph)
                # ugraph_loss=graph_rigidity_interpolation_loss
                
                # if step%3==1:
                #     graph_rigidity_interpolation_loss, graph_rigidity_interpolation_stats=get_graph_rigidity_interpolation_loss(ugraph_model.precompute,ts,is_skip_key_graph=True,is_skip_nonkey_graph=False)
                # if step%3==0:
                if True:
                    graph_rigidity_interpolation_loss, graph_rigidity_interpolation_stats=get_graph_rigidity_interpolation_loss(ugraph_model.precompute,ts,is_skip_key_graph=False,is_skip_nonkey_graph=False)
                else:
                    graph_rigidity_interpolation_loss=torch.zeros_like(loss_rgb)
                    graph_rigidity_interpolation_stats = {}
                ugraph_loss=graph_rigidity_interpolation_loss
                ##############################################################
                # local rigidity loss
                # if step%3!=0:
                if False:
                    local_rigidity_loss, local_rigidity_stats = get_local_ridigity_loss(ugraph_model.precompute, ts)
                else:
                    local_rigidity_loss = torch.zeros_like(loss_rgb)
                    local_rigidity_stats = {}
                ##############################################################
                # speed and acceleration loss
                if not use_DQB_directly:
                    dynamic_loss, dynamic_stats = get_dynamic_loss(ugraph_model.precompute,ts)
                else:
                    # dynamic_loss, dynamic_stats = get_dynamic_loss_by_input(transls_trainable_ts, oris_wxyz_trainable_ts,ts) # has BUG!!!
                    transls=ugraph_model.precompute.params['transls'] 
                    oris_wxyz=ugraph_model.precompute.params['oris_wxyz'] 
                    ids_key=ugraph_model.precompute.graph_para['ids_key']
                    ids_key=ids_key.to(transls.device)
                    transls_key=transls[:,ids_key]
                    oris_wxyz_key=oris_wxyz[:,ids_key]
                    dynamic_loss, dynamic_stats = get_dynamic_loss_by_input(transls_key, oris_wxyz_key,ts) # key points only
                # dynamic_loss = torch.zeros_like(loss_rgb)
                # dynamic_stats = {}
                
            else:
                ugraph_loss = torch.zeros_like(loss_rgb)
                local_rigidity_loss = torch.zeros_like(loss_rgb)
                dynamic_loss = torch.zeros_like(loss_rgb)
                local_rigidity_stats = {}
                dynamic_stats = {}
                graph_rigidity_interpolation_stats = {}

            ###########################################################################
            lambda_dynamic_loss=1.0
            loss = (
                ugraph_loss * 2.0
                + local_rigidity_loss * 100.0
                + dynamic_loss * lambda_dynamic_loss
                + loss_rgb * lambda_rgb
                + loss_dep * lambda_dep
                + loss_mask * lambda_mask
                + loss_nrm * lambda_normal # gof
                + loss_dep_nrm_reg * lambda_depth_normal # gof
                + loss_distortion_reg * lambda_distortion # gof
                + loss_arap_coord * lambda_arap_coord #
                + loss_arap_len * lambda_arap_len #
                + loss_vel_xyz_reg * lambda_vel_xyz_reg #
                + loss_vel_rot_reg * lambda_vel_rot_reg #
                + loss_acc_xyz_reg * lambda_acc_xyz_reg #
                + loss_acc_rot_reg * lambda_acc_rot_reg #
                + loss_small_w * lambda_small_w_reg #
                + loss_track * lambda_track
                + loss_fg_mask * 1.0
            )
            loss.backward()

            pbar.set_description(f"{step} L:{loss.item():.3f} G:{ugraph_loss.item():.3f}")
            stats = {
                "train/loss": loss.item(),
                "train/ugraph_loss": ugraph_loss.item(),
                "train/local_rigidity_loss": local_rigidity_loss.item(),
                "train/dynamic_loss": dynamic_loss.item(),
                "train/rgb_loss": loss_rgb.item(),
                "train/depth_loss": loss_dep.item(),
                "train/fg_mask_loss": loss_fg_mask.item(),
                "train/mask_loss": loss_mask.item(),
                "train/normal_loss": loss_nrm.item(),  # gof
                "train/depth_normal_reg_loss": loss_dep_nrm_reg.item(),  # gof
                "train/distortion_reg_loss": loss_distortion_reg.item(),  # gof
                "train/arap_coord_loss": loss_arap_coord.item(),
                "train/arap_len_loss": loss_arap_len.item(),
                "train/vel_xyz_reg_loss": loss_vel_xyz_reg.item(),
                "train/vel_rot_reg_loss": loss_vel_rot_reg.item(),
                "train/acc_xyz_reg_loss": loss_acc_xyz_reg.item(),
                "train/acc_rot_reg_loss": loss_acc_rot_reg.item(),
                "train/small_w_reg_loss": loss_small_w.item(),
                "train/track_2d_loss": loss_track.item(),
            }
            stats.update(dynamic_stats) # my
            stats.update(graph_rigidity_interpolation_stats) # my
            stats.update(local_rigidity_stats) # my
            wandb.log(stats)

            optimizer_static.step()
            if d_flag:
                optimizer_dynamic.step()
            if step >= optim_cam_after_steps and optimizer_cam is not None:
                optimizer_cam.step()
            if ugraph_flag:
                # optimizer_ugraph.step()
                for opt in optimizers_ugraph.values():
                    opt.step()

            # gs control
            if (
                (
                    step in d_gs_ctrl_cfg.reset_steps
                    and step >= d_gs_ctrl_start
                    and step < d_gs_ctrl_end
                )
                or (
                    step in s_gs_ctrl_cfg.reset_steps
                    and step >= s_gs_ctrl_start
                    and step < s_gs_ctrl_end
                )
                # or dynamic_to_static_transfer_flag
            ):
                if corr_flag:
                    logging.info(f"Reset event happened, protect tracking loss")
                    latest_track_event = step

            if (
                s_gs_ctrl_cfg is not None
                and step >= s_gs_ctrl_start
                and step < s_gs_ctrl_end
            ):
                apply_gs_control_s(
                    render_list=render_dict_list,
                    model=s_model,
                    gs_control_cfg=s_gs_ctrl_cfg,
                    step=step,
                    optimizer_gs=optimizer_static,
                    first_N=s_model.N,
                    record_flag=(not corr_exe_flag)
                    or (GS_BACKEND not in ["native_add3"]),
                )
            
            if (
                d_gs_ctrl_cfg is not None
                and step >= d_gs_ctrl_start
                and step < d_gs_ctrl_end
                and d_flag
            ):
                apply_gs_control_d(
                    render_list=render_dict_list,
                    model=d_model,
                    gs_control_cfg=d_gs_ctrl_cfg,
                    step=step,
                    optimizer_gs=optimizer_dynamic,
                    last_N=d_model.N,
                    record_flag=(not corr_exe_flag)
                    or (GS_BACKEND not in ["native_add3"]),
                    ugraph_model=ugraph_model,
                    optimizers_ugraph=optimizers_ugraph,
                )

                if corr_exe_flag and step > dyn_node_densify_record_start_steps:
                    # record the geo gradient
                    for corr_render_dict in corr_render_dict_list:
                        d_model.record_corr_grad(
                            # ! normalize the gradient by loss weight.
                            corr_render_dict["viewspace_points"].grad[-d_model.N :]
                            / lambda_track,
                            corr_render_dict["visibility_filter"][-d_model.N :],
                        )

            loss_rgb_list.append(loss_rgb.item())
            loss_dep_list.append(loss_dep.item())
            loss_fg_mask_list.append(loss_fg_mask.item())
            loss_nrm_list.append(loss_nrm.item())
            loss_mask_list.append(loss_mask.item())

            loss_dep_nrm_reg_list.append(loss_dep_nrm_reg.item())
            loss_distortion_reg_list.append(loss_distortion_reg.item())

            loss_arap_coord_list.append(loss_arap_coord.item())
            loss_arap_len_list.append(loss_arap_len.item())
            loss_vel_xyz_reg_list.append(loss_vel_xyz_reg.item())
            loss_vel_rot_reg_list.append(loss_vel_rot_reg.item())
            loss_acc_xyz_reg_list.append(loss_acc_xyz_reg.item())
            loss_acc_rot_reg_list.append(loss_acc_rot_reg.item())
            s_n_count_list.append(s_model.N)
            d_n_count_list.append(d_model.N if d_flag else 0)
            d_m_count_list.append(d_model.M if d_flag else 0)

            loss_small_w_list.append(loss_small_w.item())
            loss_track_list.append(loss_track.item())

            # viz
            viz_flag = viz_interval > 0 and (step % viz_interval == 0)
            if viz_flag:

                if d_flag:
                    viz_hist(d_model, self.viz_dir, f"{phase_name}_step={step}_dynamic")
                    viz_dyn_hist(
                        d_model.scf,
                        self.viz_dir,
                        f"{phase_name}_step={step}_dynamic",
                    )
                    viz_path = osp.join(
                        self.viz_dir, f"{phase_name}_step={step}_3dviz.mp4"
                    )
                    viz3d_total_video(
                        cams,
                        d_model,
                        0,
                        cams.T - 1,
                        save_path=viz_path,
                        res=480,  # 240
                        s_model=s_model,
                    )

                    # * viz grouping
                    if lambda_mask > 0.0:
                        d_model.return_cate_colors_flag = True
                        viz_path = osp.join(
                            self.viz_dir, f"{phase_name}_step={step}_3dviz_group.mp4"
                        )
                        viz3d_total_video(
                            cams,
                            d_model,
                            0,
                            cams.T - 1,
                            save_path=viz_path,
                            res=480,  # 240
                            s_model=s_model,
                        )
                        viz2d_total_video(
                            viz_vid_fn=osp.join(
                                self.viz_dir,
                                f"{phase_name}_step={step}_2dviz_group.mp4",
                            ),
                            s2d=s2d,
                            start_from=0,
                            end_at=cams.T - 1,
                            skip_t=viz_skip_t,
                            cams=cams,
                            s_model=s_model,
                            d_model=d_model,
                            subsample=1,
                            mask_type=sup_mask_type,
                            move_around_angle_deg=viz_move_angle_deg,
                        )
                        d_model.return_cate_colors_flag = False

                viz_hist(s_model, self.viz_dir, f"{phase_name}_step={step}_static")
                viz2d_total_video(
                    viz_vid_fn=osp.join(
                        self.viz_dir, f"{phase_name}_step={step}_2dviz.mp4"
                    ),
                    s2d=s2d,
                    start_from=0,
                    end_at=cams.T - 1,
                    skip_t=viz_skip_t,
                    cams=cams,
                    s_model=s_model,
                    d_model=d_model,
                    subsample=1,
                    mask_type=sup_mask_type,
                    move_around_angle_deg=viz_move_angle_deg,
                )

            if viz_cheap_interval > 0 and (
                step % viz_cheap_interval == 0 or step == total_steps - 1
            ):
                # viz the accumulated grad
                with torch.no_grad():
                    photo_grad = [
                        s_model.xyz_gradient_accum
                        / torch.clamp(s_model.xyz_gradient_denom, min=1e-6)
                    ]
                    corr_grad = [torch.zeros_like(photo_grad[0])]
                    if d_flag:
                        photo_grad.append(
                            d_model.xyz_gradient_accum
                            / torch.clamp(d_model.xyz_gradient_denom, min=1e-6)
                        )
                        corr_grad.append(
                            d_model.corr_gradient_accum
                            / torch.clamp(d_model.corr_gradient_denom, min=1e-6)
                        )

                    photo_grad = torch.cat(photo_grad, 0)
                    viz_grad_color = (
                        torch.clamp(photo_grad, 0.0, d_gs_ctrl_cfg.densify_max_grad)
                        / d_gs_ctrl_cfg.densify_max_grad
                    )
                    viz_grad_color = viz_grad_color.detach().cpu().numpy()
                    viz_grad_color = cm.viridis(viz_grad_color)[:, :3]
                    viz_render_dict = render(
                        [s_model(), d_model(view_ind)],
                        s2d.H,
                        s2d.W,
                        cams.K(s2d.H, s2d.W),
                        cams.T_cw(view_ind),
                        bg_color=[0.0, 0.0, 0.0],
                        colors_precomp=torch.from_numpy(viz_grad_color).to(photo_grad),
                    )
                    viz_grad = (
                        viz_render_dict["rgb"].permute(1, 2, 0).detach().cpu().numpy()
                    )
                    imageio.imsave(
                        osp.join(
                            self.viz_dir, f"{phase_name}_photo_grad_step={step}.jpg"
                        ),
                        viz_grad,
                    )

                    corr_grad = torch.cat(corr_grad, 0)
                    viz_grad_color = (
                        torch.clamp(corr_grad, 0.0, dyn_node_densify_grad_th)
                        / dyn_node_densify_grad_th
                    )
                    viz_grad_color = viz_grad_color.detach().cpu().numpy()
                    viz_grad_color = cm.viridis(viz_grad_color)[:, :3]
                    viz_render_dict = render(
                        [s_model(), d_model(view_ind)],
                        s2d.H,
                        s2d.W,
                        cams.K(s2d.H, s2d.W),
                        cams.T_cw(view_ind),
                        bg_color=[0.0, 0.0, 0.0],
                        colors_precomp=torch.from_numpy(viz_grad_color).to(corr_grad),
                    )
                    viz_grad = (
                        viz_render_dict["rgb"].permute(1, 2, 0).detach().cpu().numpy()
                    )
                    imageio.imsave(
                        osp.join(
                            self.viz_dir, f"{phase_name}_corr_grad_step={step}.jpg"
                        ),
                        viz_grad,
                    )

                fig = plt.figure(figsize=(30, 8))
                for plt_i, plt_pack in enumerate(
                    [
                        ("loss_rgb", loss_rgb_list),
                        ("loss_dep", loss_dep_list),
                        ("loss_fg_mask", loss_fg_mask_list),
                        ("loss_nrm", loss_nrm_list),
                        ("loss_mask", loss_mask_list),
                        ("loss_sds", loss_sds_list),
                        ("loss_dep_nrm_reg", loss_dep_nrm_reg_list),
                        ("loss_distortion_reg", loss_distortion_reg_list),
                        ("loss_arap_coord", loss_arap_coord_list),
                        ("loss_arap_len", loss_arap_len_list),
                        ("loss_vel_xyz_reg", loss_vel_xyz_reg_list),
                        ("loss_vel_rot_reg", loss_vel_rot_reg_list),
                        ("loss_acc_xyz_reg", loss_acc_xyz_reg_list),
                        ("loss_acc_rot_reg", loss_acc_rot_reg_list),
                        ("loss_small_w", loss_small_w_list),
                        ("loss_track", loss_track_list),
                        ("S-N", s_n_count_list),
                        ("D-N", d_n_count_list),
                        ("D-M", d_m_count_list),
                    ]
                ):
                    plt.subplot(2, 10, plt_i + 1)
                    value_end = 0 if len(plt_pack[1]) == 0 else plt_pack[1][-1]
                    plt.plot(plt_pack[1]), plt.title(
                        plt_pack[0] + f" End={value_end:.4f}"
                    )
                plt.savefig(
                    osp.join(self.viz_dir, f"{phase_name}_optim_loss_step={step}.jpg")
                )
                plt.close()
            
            self.save_model_every_n_steps(
                step,
                log_dir=self.log_dir,
                phase_name=phase_name,
                s_model=s_model,
                cams=cams,
                d_model=d_model,
                ugraph_model=ugraph_model,
                GS_BACKEND=GS_BACKEND,
                n_steps=200,
            )

        # save static, camera and dynamic model
        s_save_fn = osp.join(
            self.log_dir, f"{phase_name}_s_model_{GS_BACKEND.lower()}_ugraph.pth"
        )
        torch.save(s_model.state_dict(), s_save_fn)
        torch.save(cams.state_dict(), osp.join(self.log_dir, f"{phase_name}_cam_ugraph.pth"))

        if d_model is not None:
            d_model.ugraph_model=None
            d_save_fn = osp.join(
                self.log_dir, f"{phase_name}_d_model_{GS_BACKEND.lower()}_ugraph.pth"
            )
            torch.save(d_model.state_dict(), d_save_fn)

        if ugraph_model is not None:
            d_model.set_ugraph(ugraph_model)
            ugraph_save_fn = osp.join(
                self.log_dir, f"{phase_name}_ugraph_model_{GS_BACKEND.lower()}_ugraph.pth"
            )
            torch.save(ugraph_model.state_dict(), ugraph_save_fn)
            print("ugraph model saved to {}".format(ugraph_save_fn))


        # viz
        fig = plt.figure(figsize=(30, 8))
        for plt_i, plt_pack in enumerate(
            [
                ("loss_rgb", loss_rgb_list),
                ("loss_dep", loss_dep_list),
                ("loss_fg_mask", loss_fg_mask_list),
                ("loss_nrm", loss_nrm_list),
                ("loss_mask", loss_mask_list),
                ("loss_dep_nrm_reg", loss_dep_nrm_reg_list),
                ("loss_distortion_reg", loss_distortion_reg_list),
                ("loss_arap_coord", loss_arap_coord_list),
                ("loss_arap_len", loss_arap_len_list),
                ("loss_vel_xyz_reg", loss_vel_xyz_reg_list),
                ("loss_vel_rot_reg", loss_vel_rot_reg_list),
                ("loss_acc_xyz_reg", loss_acc_xyz_reg_list),
                ("loss_acc_rot_reg", loss_acc_rot_reg_list),
                ("loss_small_w", loss_small_w_list),
                ("loss_track", loss_track_list),
                ("S-N", s_n_count_list),
                ("D-N", d_n_count_list),
                ("D-M", d_m_count_list),
            ]
        ):
            plt.subplot(2, 10, plt_i + 1)
            plt.plot(plt_pack[1]), plt.title(
                plt_pack[0] + f" End={plt_pack[1][-1]:.6f}"
            )
        plt.savefig(osp.join(self.log_dir, f"{phase_name}_optim_loss.jpg"))
        plt.close()
        viz2d_total_video(
            viz_vid_fn=osp.join(self.log_dir, f"{phase_name}_2dviz.mp4"),
            s2d=s2d,
            start_from=0,
            end_at=cams.T - 1,
            skip_t=viz_skip_t,
            cams=cams,
            s_model=s_model,
            d_model=d_model,
            move_around_angle_deg=viz_move_angle_deg,
            print_text=False,
        )
        viz_path = osp.join(self.log_dir, f"{phase_name}_3Dviz.mp4")
        if d_flag:
            viz3d_total_video(
                cams,
                d_model,
                0,
                cams.T - 1,
                save_path=viz_path,
                res=480,
                s_model=s_model,
            )
            if lambda_mask > 0.0:
                # * viz grouping
                d_model.return_cate_colors_flag = True
                s_model.return_cate_colors_flag = True
                viz_path = osp.join(self.log_dir, f"{phase_name}_3Dviz_group.mp4")
                viz3d_total_video(
                    cams,
                    d_model,
                    0,
                    cams.T - 1,
                    save_path=viz_path,
                    res=480,
                    s_model=s_model,
                )
                viz2d_total_video(
                    viz_vid_fn=osp.join(self.log_dir, f"{phase_name}_2dviz_group.mp4"),
                    s2d=s2d,
                    start_from=0,
                    end_at=cams.T - 1,
                    skip_t=viz_skip_t,
                    cams=cams,
                    s_model=s_model,
                    d_model=d_model,
                    move_around_angle_deg=viz_move_angle_deg,
                    print_text=False,
                )
                d_model.return_cate_colors_flag = False
                s_model.return_cate_colors_flag = False
        return

    @torch.no_grad()
    def render_all(self, cams: MonocularCameras, s_model=None, d_model=None):
        ret = []
        assert s_model is not None or d_model is not None, "No model to render"
        s_gs5 = s_model()
        for t in tqdm(range(cams.T)):
            gs5 = [s_gs5]
            if d_model is not None:
                # gs5.append(d_model(t))
                if self.ugraph_model is None:
                    gs5.append(list(self.d_model(t)))
                else:
                    # ugraph replace mu and fr
                    mu, fr, s, o, sph = list(self.d_model(t))
                    transls_t=self.ugraph_model.precompute.params['transls'][t]
                    oris_wxyz_t=self.ugraph_model.precompute.params['oris_wxyz'][t]
                    oris_wxyz_t = F.normalize(oris_wxyz_t, p=2, dim=-1)
                    # oris_xyzw_t=roma.quat_xyzw_to_wxyz(oris_wxyz_t)
                    oris_xyzw_t=roma.quat_wxyz_to_xyzw(oris_wxyz_t)
                    rotmat_t=roma.unitquat_to_rotmat(oris_xyzw_t)
                    mu_ugraph=transls_t
                    fr_ugraph=rotmat_t
                    gs5.append([mu_ugraph, fr_ugraph, s, o, sph])

            render_dict = render(
                gs5,
                cams.default_H,
                cams.default_W,
                cams.K(),
                cams.T_cw(t),
                bg_color=[1.0, 1.0, 1.0],
            )
            ret.append(render_dict)
        rgb = torch.stack([r["rgb"] for r in ret], 0)
        dep = torch.stack([r["dep"] for r in ret], 0)
        alp = torch.stack([r["alpha"] for r in ret], 0)
        return rgb, dep, alp
    
    def start_viewer(self,num_frames,work_dir,port=9000):
        self.viewer = None
        if port is not None:
            server = get_server(port=port)
            print("Server initialized on port", port)
            # calculate pivot point
            t=0
            center=self.d_model(t,use_ugraph=False)[0][::10].mean(dim=0).cpu().detach().numpy()
            look_at_tuple=(center[0],center[1],center[2])
            # viewer things
            self.viewer_things = {
                'pts1': None,
                'pts2': None,
                'lines': None,}
            # Visualize the object
            self.viewer = DynamicViewer(
                server, self.render_fn, num_frames, work_dir, mode="training",look_at_tuple=look_at_tuple
            )
            print("Renderer initialized")

    @torch.inference_mode()
    def render_fn(self, camera_state: CameraState, img_wh: tuple[int, int]):
        if self.viewer is None:
            print("Viewer is None")
            return np.full((img_wh[1], img_wh[0], 3), 255, dtype=np.uint8)

        W, H = img_wh

        focal = 0.5 * H / np.tan(0.5 * camera_state.fov).item()
        K = torch.tensor(
            [[focal, 0.0, W / 2.0], [0.0, focal, H / 2.0], [0.0, 0.0, 1.0]],
            device=self.device,
        )
        w2c = torch.linalg.inv(
            torch.from_numpy(camera_state.c2w.astype(np.float32)).to(self.device)
        )
        t = (
            int(self.viewer._playback_guis[0].value)
            if not self.viewer._canonical_checkbox.value
            else None
        )
        gs5 = []
        # if self.bg_flag:
        if self.viewer._show_bg_checkbox.value:
            gs5.append(self.s_model())

        if self.ugraph_model is not None:
            # ugraph replace mu and fr
            mu, fr, s, o, sph = list(self.d_model(t))
            if not self.viewer._show_dqb_directly_checkbox.value:
                gs5_ugraph_list=[mu, fr, s, o, sph]
                transls_t=mu
                rotmat_t=fr
            else:
                transls_t = self.ugraph_model.precompute.graph_para['transls_interpolation'][t]
                oris_wxyz_t = self.ugraph_model.precompute.graph_para['oris_interpolation'][t]
                oris_wxyz_t = F.normalize(oris_wxyz_t, p=2, dim=-1)
                # oris_xyzw_t=roma.quat_xyzw_to_wxyz(oris_wxyz_t) # [BUG]
                oris_xyzw_t=roma.quat_wxyz_to_xyzw(oris_wxyz_t)
                rotmat_t=roma.unitquat_to_rotmat(oris_xyzw_t)
                mu_ugraph=transls_t
                fr_ugraph=rotmat_t
                gs5_ugraph_list=[mu_ugraph, fr_ugraph, s, o, sph]
        else:
            gs5_ugraph_list=None

        if self.viewer._show_fg_checkbox.value:
            # if self.ugraph_model is None:
            if self.viewer._show_d_model_checkbox.value: # just for test # d_model's pretrained data
                gs5.append(list(self.d_model(t,use_ugraph=False)))
            else:
                gs5.append(gs5_ugraph_list)

        if self.ugraph_model is not None:
            ids_key=self.ugraph_model.precompute.graph_para["ids_key"]
            contribs_key = self.ugraph_model.precompute.graph_para["contribs_key"] # (nt,nk)

            means_key = transls_t[ids_key]
            # quats_key = oris_xyzw_t[ids_key]
            
            # draw points
            if self.viewer._render_key_pts_contrib_checkbox.value:
                contrib_threshold_key = self.ugraph_model.precompute.graph_para["contrib_threshold_key"] # around 2
                pts1 = np.array(means_key[contribs_key[t] >= contrib_threshold_key].cpu().numpy())
                self.viewer_things['pts1'] = self.viewer.server.scene.add_point_cloud(
                    "/key_confident",
                    points=pts1,  # shape (N, 3)
                    point_shape='diamond',
                    point_size=0.003,
                    colors=np.tile(np.array([[0.0, 0.0, 1.0]]), (pts1.shape[0], 1)),  # shape (N, 3)
                )
                pts2 = np.array(means_key[contribs_key[t]<contrib_threshold_key].cpu().numpy())
                self.viewer_things['pts2'] = self.viewer.server.scene.add_point_cloud(
                    "/key_unconfident",
                    points=pts2,  # shape (N, 3)
                    point_shape='diamond',
                    point_size=0.003,
                    colors=np.tile(np.array([[1.0, 0.0, 0.0]]), (pts2.shape[0], 1)),  # shape (N, 3)
                )
            else:
                if self.viewer_things['pts1'] is not None:
                    self.viewer_things['pts1'].remove()
                if self.viewer_things['pts2'] is not None:
                    self.viewer_things['pts2'].remove()
                self.viewer_things['pts1']=None
                self.viewer_things['pts2']=None

        if gs5 != []:
            try:
                render_dict = render(
                    gs5,
                    H,
                    W,
                    K=K,
                    T_cw=w2c,
                    bg_color=[1.0, 1.0, 1.0],
                )
        
                img = torch.clamp(render_dict["rgb"].permute(1, 2, 0), 0.0, 1.0)
                # I want to render the image given this frame
                # if not self.viewer._render_track_checkbox.value:
                img = (img.cpu().numpy() * 255.0).astype(np.uint8)
            except:
                img = torch.ones((img_wh[0],img_wh[1],3),device=self.device)
                img = (img.cpu().numpy() * 255.0).astype(np.uint8)
        else:
            img = torch.ones((img_wh[0],img_wh[1],3),device=self.device) 
            img = (img.cpu().numpy() * 255.0).astype(np.uint8)
        
        # if isinstance(img, torch.Tensor): # test
        #     img = img.detach().cpu().numpy()
        return img


    def save_model_every_n_steps(
        self,
        step: int,
        log_dir: str,
        phase_name: str,
        s_model,
        cams,
        d_model=None,
        ugraph_model=None,
        GS_BACKEND="PYTORCH",
        n_steps=50,
    ):
        save_steps=[999,1599,1999,2999,3999,4999,5999,6999,7999]
        if step not in save_steps:
            return

        # Create base and step folder
        save_dir = osp.join(self.dir_saving_this_run_model, f"step_{step}") # "saved_ugraph_model"
        
        os.makedirs(save_dir, exist_ok=True)

        # Save s_model
        s_save_fn = osp.join(
            save_dir, f"{phase_name}_s_model_{GS_BACKEND.lower()}.pth"
        )
        torch.save(s_model.state_dict(), s_save_fn)

        # Save cams
        torch.save(
            cams.state_dict(), osp.join(save_dir, f"{phase_name}_cam.pth")
        )

        # Save d_model if present
        if d_model is not None:
            d_save_fn = osp.join(
                save_dir, f"{phase_name}_d_model_{GS_BACKEND.lower()}.pth"
            )
            torch.save(d_model.state_dict(), d_save_fn)

        # Save ugraph_model if present
        if ugraph_model is not None:
            ugraph_save_fn = osp.join(
                save_dir, f"{phase_name}_ugraph_model_{GS_BACKEND.lower()}.pth"
            )
            torch.save(ugraph_model.state_dict(), ugraph_save_fn)

        print(f"✅ Saved model at step {step} → {save_dir}")

    def init_wandb(
        self,
        log_dir: str,
    ):
        wandb.login()
        # log_dir eg: "/data/dataset/iphone_mosca_trained/backpack_/logs/iphone_fit_native_add3/"
        if log_dir[-1]=="/":
            dataset_name=log_dir.split("/")[-4] # 
        else:
            dataset_name=log_dir.split("/")[-3] # 
        run = wandb.init(
            # Set the project where this run will be logged
            # For example: project=f"ugraph_mosca_DQB_{dataset_name}", dataset_name=schoolgirls
            project=f"mosca_DQB_{dataset_name}", # schoolgirls
            # Track hyperparameters and run metadata
            config={
                "info": "Train from precomputed graph and MoSca results.",
                "log_dir": log_dir,
            },
        )

    def tensor_memory_report(self):
        import gc
        import torch
        total_mem_gb = 0
        total_mem_gb_larger = 0
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, "data") and torch.is_tensor(obj.data)):
                    if obj.is_cuda:
                        size_gb = obj.numel() * obj.element_size() / 1024**3
                        total_mem_gb += size_gb
                        if size_gb > 0.1:  # > 100MB
                            total_mem_gb_larger += size_gb
                            print(f"Tensor: {type(obj)}, size: {tuple(obj.shape)}, dtype: {obj.dtype}, "
                                f"grad: {obj.requires_grad}, mem: {size_gb:.3f} GB")
            except (Exception, ModuleNotFoundError, ImportError):
                pass
        print(f"\nTotal GPU tensor memory: {total_mem_gb:.3f} GB")
        print(f"\nTotal GPU tensor memory (only >100MB): {total_mem_gb_larger:.3f} GB")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"Max Allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        print(f"Max Cached: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")


