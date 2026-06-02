import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger as guru
from nerfview import CameraState
from lib_render.render_helper import render
from scipy.spatial.transform import Rotation

# from flow3d.scene_model import SceneModel
# from flow3d.vis.utils import draw_tracks_2d_th, get_server
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
    
from mosca_cfg import load_model_cfg
from dataclasses import dataclass

import argparse
import time
import os, os.path as osp
import os
import json
from tqdm import tqdm
import imageio
import logging
import matplotlib.pyplot as plt

from lib_moca.camera import MonocularCameras
from lib_mosca.dynamic_gs import DynSCFGaussian
from lib_mosca.static_gs import StaticGaussian
from eval_utils.campose_alignment import align_ate_c2b_use_a2b

import roma

class Renderer:
    def __init__(
        self,
        cfg_fn: str,
        device: torch.device,
        work_dir: str,
        port: int = 8899, #| None = None,
        bg_flag: bool = False,
        fg_flag: bool = True,
        load_s2d: bool = False,
        ws: str = '',
        pth_save_dir: str | None = None ,
        use_ugraph = True,
        relative_dir_saved_mosca_model: str | None = None,
        start_viewer: bool = True,
        images_dir: str | None = None,
    ):  
        if pth_save_dir==None:
            pth_save_dir=work_dir
            if relative_dir_saved_mosca_model is not None:
                pth_save_dir=os.path.join(pth_save_dir,relative_dir_saved_mosca_model)
            print("pth_save_dir is set to:",pth_save_dir)
        self.cfg, self.d_model, self.s_model, self.cams = load_model_cfg(cfg_fn, pth_save_dir, use_ugraph=use_ugraph)
        self.device = device
        self.ugraph_model = self.d_model.ugraph_model
        self.images_dir = images_dir

        self.work_dir = work_dir
        self.global_step = 0
        self.epoch = 0

        self.bg_flag = bg_flag
        self.fg_flag = fg_flag

        self.s2d=None
        if load_s2d:
            self.s2d=self.load_s2d(cfg_fn,ws,logdir=work_dir)

        num_frames=self.cams.T
        if start_viewer:
            if use_ugraph:
                self.start_viewer_ugraph(num_frames,work_dir,port)
            else:
                self.start_viewer(num_frames,work_dir,port)
        

    def load_s2d(self,cfg_fn,ws,logdir):
        from lib_prior.prior_loading import Saved2D
        # from lib_render.render_helper import GS_BACKEND
        # from lib_moca.camera import MonocularCameras
        from lib_mosca.misc import seed_everything
        from recon_utils import (
            SEED,
            seed_everything,
            setup_recon_ws,
            auto_get_depth_dir_tap_mode,
            update_s2d_track_identification,
            viz_mosca_curves_before_optim,
            set_epi_mask_to_s2d_for_bg_render,
        )
        from omegaconf import OmegaConf

        # cfg = OmegaConf.load(args.cfg)
        # cli_cfg = OmegaConf.from_dotlist([arg.lstrip("--") for arg in unknown])
        # cfg = OmegaConf.merge(cfg, cli_cfg)

        # ws='./data/iphone/<video_name>/'
        fit_cfg = OmegaConf.load(cfg_fn)
        # logdir = setup_recon_ws(ws, fit_cfg=fit_cfg)
        log_path = logdir

        seed_everything(SEED)
        # ! here the warup do not need to start from low opa, only when mix two component we start from low opa!
        DEPTH_DIR, TAP_MODE = auto_get_depth_dir_tap_mode(ws, fit_cfg)
        DEPTH_BOUNDARY_TH = getattr(fit_cfg, "depth_boundary_th", 1.0)
        DEP_MEDIAN = getattr(fit_cfg, "dep_median", 1.0)
        EPI_TH = getattr(fit_cfg, "photo_warm_epi_th", getattr(fit_cfg, "epi_th", 1e-3))
        # PHOTO_STATIC_WARM_STEPS = getattr(fit_cfg, "photo_static_warm_steps", -1)
        # if PHOTO_STATIC_WARM_STEPS < 0:
        #     logging.info("No static warmup needed")
        #     return
        device = torch.device("cuda:0")
        logging.info(
            f"First run static bg GS warm up to save time before joint optimization"
        )

        s2d = (
            Saved2D(ws)
            .load_epi()
            .load_dep(DEPTH_DIR, DEPTH_BOUNDARY_TH)
            .normalize_depth(median_depth=DEP_MEDIAN)
            .recompute_dep_mask(depth_boundary_th=DEPTH_BOUNDARY_TH)
            .load_track(
                TAP_MODE, min_valid_cnt=getattr(fit_cfg, "tap_loading_min_valid_cnt", 4)
            )
            .rescale_perframe_depth_from_bundle(
                bundle_pth_fn=osp.join(log_path, "bundle", "bundle.pth")
            )
            .load_vos()
            .to(device)
        )
        
        # Load GT masks if enabled
        use_gt_masks = getattr(fit_cfg, "use_gt_masks", True)
        gt_mask_path = getattr(fit_cfg, "gt_mask_path", "mask")
        if use_gt_masks:
            s2d.load_gt_masks(gt_mask_path)
        
        s2d = set_epi_mask_to_s2d_for_bg_render(s2d, EPI_TH, device, use_gt_masks=use_gt_masks)
        return s2d
    
    def start_viewer(self,num_frames,work_dir,port=1000):
        self.viewer = None
        if port is not None:
            server = get_server(port=port)
            print("Server initialized on port", port)
            # calculate pivot point
            t=int(num_frames/2)
            center=self.d_model(t)[0][::10].mean(dim=0).cpu().detach().numpy()
            look_at_tuple=(center[0],center[1],center[2])

            # Visualize the object
            self.viewer = DynamicViewer(
                server, self.render_fn, num_frames, work_dir, mode="rendering",look_at_tuple=look_at_tuple, cams=self.cams
            )
            self.viewer._show_bg_checkbox.value=self.bg_flag
            self.viewer._show_fg_checkbox.value=self.fg_flag
            print("Renderer initialized")
            # Add camera frustums
            self.add_camera_frustums(server)
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
        if self.viewer._show_bg_checkbox.value:
            gs5.append(self.s_model())
        if self.viewer._show_fg_checkbox.value:
            gs5.append(list(self.d_model(t)))

        # if self.bg_flag:
        #     gs5.append(self.s_model())
        # if self.fg_flag:
        #     gs5.append(self.d_model(t))
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
        if not self.viewer._render_track_checkbox.value:
            img = (img.cpu().numpy() * 255.0).astype(np.uint8)
        return img

    @torch.inference_mode() # my
    def render_image(self, t: int, K: torch.tensor, w2c: torch.tensor, H: int, W: int,render_bg=True, render_fg=True, offset_y=0.0):
        # w2c = torch.linalg.inv(
        #     torch.from_numpy(camera_state.c2w.astype(np.float32)).to(self.device)
        # )
        gs5 = []
        if render_bg:
            gs5.append(self.s_model())
        if render_fg:
            gs5.append(list(self.d_model(t)))

        render_dict = render(
            gs5,
            H,
            W,
            K=K,
            T_cw=w2c,
            bg_color=[1.0, 1.0, 1.0],
        )
   
        img = torch.clamp(render_dict["rgb"].permute(1, 2, 0), 0.0, 1.0)
        img = (img.cpu().numpy() * 255.0).astype(np.uint8)
        return img
    
    def start_viewer_ugraph(self,num_frames,work_dir,port=9000):
        self.viewer = None
        if port is not None:
            server = get_server(port=port)
            print("Server initialized on port", port)
            # calculate pivot point
            t=int(num_frames/2)
            center=self.d_model(t,use_ugraph=False)[0][::10].mean(dim=0).cpu().detach().numpy()
            look_at_tuple=(center[0],center[1],center[2])
            # viewer things
            self.viewer_things = {
                'pts1': None,
                'pts2': None,
                'lines': None,}
            # Visualize the object
            self.viewer = DynamicViewer(
                server, self.render_fn_ugraph, num_frames, work_dir, mode="training",look_at_tuple=look_at_tuple, cams=self.cams
            )
            self.viewer._show_bg_checkbox.value=self.bg_flag
            self.viewer._show_fg_checkbox.value=self.fg_flag
            print("Renderer initialized")
            # Add camera frustums
            self.add_camera_frustums(server)

    def add_camera_frustums(self, server, frustum_scale=0.05):
        """Add camera frustums to the viser scene, optionally with images."""
        H = int(self.cams.default_H.item())
        W = int(self.cams.default_W.item())
        K = self.cams.K(H, W)
        fx, fy = K[0, 0].item(), K[1, 1].item()
        fov_y = 2 * np.arctan(H / (2 * fy))
        aspect = W / H
        
        for i in range(self.cams.T):
            c2w = self.cams.T_wc(i).detach().cpu().numpy()
            position = c2w[:3, 3]
            rot = Rotation.from_matrix(c2w[:3, :3])
            quat_xyzw = rot.as_quat()
            wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            
            image = None
            if self.images_dir is not None:
                for fmt in [f"{i:03d}", f"{i:05d}"]:
                    for ext in ['.png', '.jpg', '.jpeg']:
                        image_path = osp.join(self.images_dir, f"{fmt}{ext}")
                        if osp.exists(image_path):
                            image = imageio.imread(image_path)
                            break
                    if image is not None:
                        break
            
            server.scene.add_camera_frustum(
                name=f"/cameras/cam_{i:03d}",
                fov=fov_y,
                aspect=aspect,
                scale=frustum_scale,
                image=image,
                wxyz=wxyz,
                position=position,
            )
        print(f"Added {self.cams.T} camera frustums")

    @torch.inference_mode()
    def render_fn_ugraph(self, camera_state: CameraState, img_wh: tuple[int, int]):
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
                img = (img.cpu().numpy() * 127.0).astype(np.uint8)
        else:
            img = torch.ones((img_wh[0],img_wh[1],3),device=self.device) 
            img = (img.cpu().numpy() * 127.0).astype(np.uint8)
        
        # if isinstance(img, torch.Tensor): # test
        #     img = img.detach().cpu().numpy()
        return img
def render_test(
    H,
    W,
    cams: MonocularCameras,
    s_model: StaticGaussian,
    d_model: DynSCFGaussian,
    train_camera_T_wi,
    test_camera_T_wi,
    test_camera_tid,
    save_dir=None,
    fn_list=None,
    focal_length=None,
    cx=None, 
    cy=None, 
    is_davis=True,
    # cover_factor=0.3,
):
    # prior2d: Prior2D = self.prior2d
    # device = self.device
    device = d_model.device

    # first align the camera
    solved_cam_T_wi = torch.stack([cams.T_wc(i) for i in range(cams.T)], 0)
    
    if is_davis:
        aligned_train_camera_T_wi = align_ate_c2b_use_a2b(
            traj_a=train_camera_T_wi,
            traj_b=solved_cam_T_wi.detach().cpu(),
            tfm_method="sim3"
        )

        aligned_test_camera_T_wi = align_ate_c2b_use_a2b(
            traj_a=train_camera_T_wi,
            traj_b=solved_cam_T_wi.detach().cpu(),
            traj_c=test_camera_T_wi,
            tfm_method="sim3"
        )
        dist_t = test_camera_distance(
            solved_cam_T_wi.detach().cpu(),
            aligned_train_camera_T_wi,
        )
        print(f"dist_t: {dist_t}")

        rot_r_tfm = test_camera_angle(
            aligned_train_camera_T_wi,
            solved_cam_T_wi.detach().cpu(),
        )
        print(f"rot_r_tfm: {rot_r_tfm}")

        # SE3
        aligned_train_camera_T_wi = align_ate_c2b_use_a2b(
            traj_a=aligned_train_camera_T_wi,
            traj_b=solved_cam_T_wi.detach().cpu(),
            tfm_method="se3"
        )

        aligned_test_camera_T_wi = align_ate_c2b_use_a2b(
            traj_a=aligned_train_camera_T_wi,
            traj_b=solved_cam_T_wi.detach().cpu(),
            traj_c=aligned_test_camera_T_wi,
            tfm_method="se3"
        )

        # scale
        aligned_train_camera_T_wi[:, :3, 3] = aligned_train_camera_T_wi[:, :3, 3] / 20
        aligned_test_camera_T_wi[:, :3, 3] = aligned_test_camera_T_wi[:, :3, 3] / 20

        # # Pure Rotation with SVD
        # R_aligned = find_global_rotation(
        #     R_est=aligned_train_camera_T_wi[:, :3, :3],
        #     R_ref=solved_cam_T_wi[:, :3, :3].detach().cpu(),
        # )
        # print(f"R_aligned: {R_aligned}")

        # aligned_train_camera_T_wi[:, :3, :3] = R_aligned @ aligned_train_camera_T_wi[:, :3, :3]
        # rot_r = test_camera_angle(
        #     aligned_train_camera_T_wi,
        #     solved_cam_T_wi.detach().cpu(),
        # )
        # print(f"rot_r: {rot_r}")
        # dist_t = test_camera_distance(
        #     solved_cam_T_wi.detach().cpu(),
        #     aligned_train_camera_T_wi,
        # )
        # print(f"dist_t: {dist_t}")

        # aligned_test_camera_T_wi[:, :3, :3] = R_aligned @ aligned_test_camera_T_wi[:, :3, :3]

    else: # iphone
        aligned_test_camera_T_wi = align_ate_c2b_use_a2b(
            traj_a=train_camera_T_wi,
            traj_b=solved_cam_T_wi.detach().cpu(),
            traj_c=test_camera_T_wi,
            tfm_method="sim3"
        )

    # render
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    K = torch.eye(3).to(device)
    K[0, 0] = K[0, 0] * 0 + focal_length
    K[1, 1] = K[1, 1] * 0 + focal_length
    K[0, 2] = K[0, 2] * 0 + cx
    K[1, 2] = K[1, 2] * 0 + cy

    test_ret = []
    for i in tqdm(range(len(test_camera_tid))):
        working_t = test_camera_tid[i]
        render_dict = render(
            [d_model(working_t)],
            H,
            W,
            K,
            T_cw=torch.linalg.inv(aligned_test_camera_T_wi[i]).to(device),
        )
        rgb = render_dict["rgb"].permute(1, 2, 0).detach().cpu().numpy()
        rgb = np.clip(rgb, 0, 1)  # ! important
        test_ret.append(rgb)

        if save_dir:
            rgb_uint8 = (rgb * 255).astype(np.uint8)
            imageio.imwrite(osp.join(save_dir, f"{fn_list[i]}.png"), rgb_uint8)
    return test_ret, aligned_train_camera_T_wi, aligned_test_camera_T_wi

def read_iphone_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    W, H = data["image_size"]
    focal_length = data["focal_length"]
    cx, cy = data["principal_point"]

    return [ W, H, focal_length, cx, cy]

@dataclass
class normalization:
    scale: float = 1.0
    transfm: torch.tensor = torch.eye(4)

def normzalize_camera_pose(
        T_w2c: torch.tensor,
        norm: normalization = None,
    ) -> torch.tensor:
    normalized_T_w2c = T_w2c.clone()
    if norm is not None:
        normalized_T_w2c = normalized_T_w2c @ torch.linalg.inv(norm.transfm)
        normalized_T_w2c[:3, 3] /= norm.scale
        
    return normalized_T_w2c


def load_som_iphone_cameras(dir_path, norm: normalization = None, default_img_size=(720, 960)):
    ######################################
    # ! dycheck use Nerfies convention, which is opencv format
    # https://github.com/google/nerfies#datasets
    # https://github.com/KAIR-BAIR/dycheck/blob/main/docs/DATASETS.md
    ######################################
    if not osp.exists(dir_path):
        return None
    all_fns = [f for f in os.listdir(dir_path) if f.endswith(".json")]
    all_fns.sort()
    all_cam_ids = np.array([int(f.split("_")[0].split(".")[0]) for f in all_fns])
    all_cam_ids = np.unique(all_cam_ids).tolist()
    logging.info(f"Loading iphone cameras from {dir_path} with camid={all_cam_ids}")
    ret = {}
    for cam_id in all_cam_ids:
        cam_fns = [f for f in all_fns if f.startswith(f"{cam_id}")]
        cam_fns.sort()
        T_wi_list, fovdeg_list, t_list = [], [], []
        cxcy_ratio_list = []
        # ! iphone dataset only has one fov
        for fn in tqdm(cam_fns):
            time_ind = int(fn.split("_")[1].split(".")[0])  # ! time start from 0, not 1
            t_list.append(time_ind)

            with open(osp.join(dir_path, fn), "r") as f:
                cam_data = json.load(f)

            # ! for shape of motion camera
            if "image_size" not in cam_data:
                cam_data["image_size"] = default_img_size  # W,H

            R_cw = np.asarray(cam_data["orientation"])
            t_cw = -R_cw @ np.asarray(cam_data["position"])
            T_cw = torch.concat(
                [
                    torch.from_numpy(R_cw).float(),
                    torch.from_numpy(t_cw).unsqueeze(1).float(),
                ],
                dim=1,
            )
            T_cw = torch.concat(
                [
                    T_cw,
                    torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=T_cw.device),
                ],
                dim=0,
            )
            if norm is not None:
                # print(f"norm: {norm}")
                T_cw = normzalize_camera_pose(T_cw, norm=norm)
            else:
                print("norm is None")

            focal = cam_data["focal_length"]
            T_wi = torch.linalg.inv(T_cw)
            # world = R.T @ local
            # https://github.com/google/nerfies/blob/1a38512214cfa14286ef0561992cca9398265e13/nerfies/camera.py#L263
            # the optical axis is the last row
            T_wi_list.append(T_wi)

            # ! get the fov in current code format, the focal with the shortest side
            short_size = min(cam_data["image_size"])  # -1, +1
            fovdeg = np.rad2deg(2 * np.arctan(short_size / (2 * focal)))

            # also load camera center
            principal_point = np.asarray(cam_data["principal_point"])
            image_size = np.asarray(cam_data["image_size"])
            cx_ratio = principal_point[0] / image_size[0]
            cy_ratio = principal_point[1] / image_size[1]
            cxcy_ratio_list.append([cx_ratio, cy_ratio])

            fovdeg_list.append(fovdeg)
        ret[cam_id] = {
            "T_wi": torch.stack(T_wi_list, 0).float(),
            "fovdeg": fovdeg_list,
            "cxcy_ratio": cxcy_ratio_list,
            "t": t_list,
            "filenames": [f[:-5] for f in cam_fns],
        }
    return ret

def load_som_gt_poses(train_cams_dir, test_cams_dir, norm_dict_path, seq_name='apple', subsample_step=1):
    with open(norm_dict_path, 'r') as f:
        norm_dict = json.load(f)
    norm = normalization(
        scale=norm_dict[seq_name]["scale"],
        transfm=torch.tensor(norm_dict[seq_name]["transfm"]).reshape(4, 4),
    )
    gt_training_cams = load_som_iphone_cameras(train_cams_dir, norm=norm)
    assert (
        len(gt_training_cams) == 1
    ), "Only support Mono camera for now in Iphone dataset"
    # subsample the gt camera
    gt_training_cams = {k: v[::subsample_step] for k, v in gt_training_cams[0].items()}
    gt_testing_cams = load_som_iphone_cameras(test_cams_dir, norm=None)
    gt_training_fov = gt_training_cams["fovdeg"][0]
    logging.info(
        f"Iphone gt training camera fov is {gt_training_cams['fovdeg'][0]:.3f} deg."
    )
    gt_training_cam_T_wi = gt_training_cams["T_wi"]
    # ! this bound which test frames can be retrieved
    gt_training_tids = gt_training_cams["t"]
    gt_training_cxcy_ratio = gt_training_cams["cxcy_ratio"]

    gt_testing_cam_T_wi_list, gt_testing_tids_list = [], []
    gt_testing_fov_list, gt_testing_fns_list = [], []
    gt_testing_cxcy_ratio_list = []

    if gt_testing_cams is None:
        return (
            gt_training_cam_T_wi,
            None,
            None,
            None,
            gt_training_fov,
            None,
            gt_training_cxcy_ratio,
            None,
        )
    for it in gt_testing_cams.values():
        sample_index = [i for i, t in enumerate(it["t"]) if t in gt_training_tids]
        gt_testing_tids = [gt_training_tids.index(it["t"][i]) for i in sample_index]
        gt_testing_tids_list.append(gt_testing_tids)
        gt_testing_cam_T_wi = it["T_wi"][sample_index]
        gt_testing_cam_T_wi_list.append(gt_testing_cam_T_wi)
        gt_testing_fns_list.append([it["filenames"][i] for i in sample_index])
        # ! assume all cam stays the same across time, use the first one
        gt_testing_fov_list.append(it["fovdeg"][0])
        gt_testing_cxcy_ratio_list.append(it["cxcy_ratio"][0])
    # todo: can use gt camera for evaluation
    return (
        gt_training_cam_T_wi,
        gt_testing_cam_T_wi_list,
        gt_testing_tids_list,
        gt_testing_fns_list,
        gt_training_fov,
        gt_testing_fov_list,
        gt_training_cxcy_ratio,
        gt_testing_cxcy_ratio_list,
    )

def run_render_val(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    renderer = Renderer(args.cfg_fn, device, args.work_dir, bg_flag=False, fg_flag=True)

    json_files = sorted(os.listdir(args.test_cam_dir))
    json_file = os.path.join(args.test_cam_dir, json_files[0])
    
    (
        W, H, focal_length, cx, cy
    ) = read_iphone_json(json_file)
    
    (
        gt_training_cam_T_wi,
        gt_testing_cam_T_wi_list,
        gt_testing_tids_list,
        gt_testing_fns_list,
        gt_training_fov,
        gt_testing_fov_list,
        gt_training_cxcy_ratio,
        gt_testing_cxcy_ratio_list,
    ) = load_som_gt_poses(
        train_cams_dir=args.train_cam_dir, 
        test_cams_dir=args.test_cam_dir, 
        norm_dict_path=args.norm_dict_path,
        seq_name=args.seq_name,
    )
    # If the test camera pose is fixed and each with only 1 file
    if len(gt_testing_tids_list[0]) != renderer.cams.T:
        gt_testing_tids_list = [list(range(renderer.cams.T)) for _ in range(len(gt_testing_tids_list))]
        gt_testing_cam_T_wi_list = [t.repeat(renderer.cams.T, 1, 1) for t in gt_testing_cam_T_wi_list]

        expanded_testing_fns_list = []
        for i, item in enumerate(gt_testing_fns_list):
            new_list = [f"{i}_{j:04d}" for j in range(renderer.cams.T)]
            expanded_testing_fns_list.append(new_list)
    else: 
        expanded_testing_fns_list = gt_testing_fns_list
    
    camera_plot_list = [
        gt_training_cam_T_wi,
        gt_testing_cam_T_wi_list[0],
        gt_testing_cam_T_wi_list[1],
        gt_testing_cam_T_wi_list[2],
    ]
    i = gt_testing_tids_list[0][0]
    mean = renderer.d_model(i)[0].mean(dim=0)
    
    object_center = mean

    # plot camera frustum
    fig_orig = plot_camera_frustums(
        camera_plot_list, 
        object_center=torch.tensor([0.0, 0.0, 0.0], device=device),
    )
    fig_orig.write_html("./trj_orig.html")
    
    camera_plot_opt_list = []
    for test_i, test_cam in enumerate(gt_testing_cam_T_wi_list):
        frames, solved_cams, test_cams = render_test(
            H=H, 
            W=W, 
            cams=renderer.cams, 
            s_model=renderer.s_model, 
            d_model=renderer.d_model, 
            train_camera_T_wi=gt_training_cam_T_wi,
            test_camera_T_wi=gt_testing_cam_T_wi_list[test_i],
            test_camera_tid=gt_testing_tids_list[test_i],
            save_dir=args.save_dir+f"/pose_{test_i}",
            fn_list=expanded_testing_fns_list[test_i],
            focal_length=focal_length,
            cx=cx, 
            cy=cy,
        )
        frames = [(frame * 255).astype(np.uint8) for frame in frames]
        imageio.mimsave(osp.join(args.save_dir, f"test_cam{test_i}.mp4"), frames, fps=30)

        if len(camera_plot_opt_list) == 0:
            camera_plot_opt_list.append(solved_cams)
        camera_plot_opt_list.append(test_cams)
    assert len(camera_plot_opt_list) == len(camera_plot_list)
    fig_opt = plot_camera_frustums(
        camera_plot_opt_list, 
        object_center=object_center,
    )
    fig_opt.write_html("./trj_opt.html")

def main():
    parser = argparse.ArgumentParser(prog="mosca_renderer")
    parser.add_argument("--cfg_fn", type=str, required=True, help="MoSca default config")
    parser.add_argument("--work_dir", type=str, required=True, help="work dir with ckpt")
    parser.add_argument("--test_cam_dir", type=str, required=True, help="test camera dir that I want to render")
    parser.add_argument("--save_dir", type=str, required=True, help="save dir to save the rendered images")

    parser.add_argument("--train_cam_dir", type=str, required=True, help="train camera dir that as reference")
    parser.add_argument("--seq_name", type=str, default="apple", help="sequence name")
    parser.add_argument("--norm_dict_path", type=str, required=True, help="normalization dict path")
    parser.set_defaults(func=run_render_val)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()