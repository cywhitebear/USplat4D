import torch
import os.path as osp
import logging
import numpy as np
from viz_utils import *
from mosca_cfg import load_model_cfg

logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)

def viz_main(
    save_dir,
    log_dir,
    cfg_fn,
    N=5,
    move_angle_deg=60.0,
    H_3d=960,
    # H_3d=640,
    W_3d=960,
    fov_3d=70,
    back_ratio_3d=0.8,
    up_ratio=0.4,
    bg_color=[1.0, 1.0, 1.0],
    center_point=[0.0, 0.3, 1.1],
):
    cfg, d_model, s_model, cams = load_model_cfg(cfg_fn, log_dir)
    H, W = cams.default_H, cams.default_W

    rel_focal_3d = 1.0 / np.tan(np.deg2rad(fov_3d) / 2.0)

    key_steps = np.linspace(0, cams.T - 1, N, dtype=int).tolist()

    # * Get pose
    global_pose_list = get_global_3D_cam_T_cw(
        s_model,
        d_model,
        cams,
        H,
        W,
        cams.T // 2,
        back_ratio=back_ratio_3d,
        up_ratio=up_ratio,
    )
    global_pose_list = global_pose_list[None].expand(cams.T, -1, -1)
    training_pose_list = [cams.T_cw(t) for t in range(cams.T)]

    # * #############################################################################
    save_fn_prefix = osp.join(save_dir, f"3D_moving_cam")
    # viz_single_2d_camera_video(
    #     H_3d,
    #     W_3d,
    #     cams,
    #     s_model,
    #     d_model,
    #     save_fn_prefix,
    #     global_pose_list,
    #     rel_focal=rel_focal_3d,
    #     bg_color=bg_color,
    # )

    # viz 3D
    save_fn_prefix = osp.join(save_dir, f"3D_moving_node")
    # viz_single_2d_node_video(
    #     H_3d,
    #     W_3d,
    #     cams,
    #     s_model,
    #     d_model,
    #     save_fn_prefix,
    #     global_pose_list,
    #     rel_focal=rel_focal_3d,
    #     bg_color=bg_color,
    # )

    save_fn_prefix = osp.join(save_dir, f"3D_moving_flow")
    # viz_single_2d_flow_video(
    #     H_3d,
    #     W_3d,
    #     cams,
    #     s_model,
    #     d_model,
    #     save_fn_prefix,
    #     global_pose_list,
    #     rel_focal=rel_focal_3d,
    #     bg_color=bg_color,
    # )

    save_fn_prefix = osp.join(save_dir, f"3D_moving")
    # viz_single_2d_video(
    #     H_3d,
    #     W_3d,
    #     cams,
    #     s_model,
    #     d_model,
    #     save_fn_prefix,
    #     global_pose_list,
    #     rel_focal=rel_focal_3d,
    #     bg_color=bg_color,
    # )

    # flow
    save_fn_prefix = osp.join(save_dir, f"training_moving_flow")
    # viz_single_2d_flow_video(
    #     H,
    #     W,
    #     cams,
    #     s_model,
    #     d_model,
    #     save_fn_prefix,
    #     training_pose_list,
    #     bg_color=bg_color,
    # )
    # node
    save_fn_prefix = osp.join(save_dir, f"training_moving_node")
    # viz_single_2d_node_video(
    #     H,
    #     W,
    #     cams,
    #     s_model,
    #     d_model,
    #     save_fn_prefix,
    #     training_pose_list,
    #     bg_color=bg_color,
    # )
    # rgb
    save_fn_prefix = osp.join(save_dir, f"training_moving")
    # viz_single_2d_video(
    #     H,
    #     W,
    #     cams,
    #     s_model,
    #     d_model,
    #     save_fn_prefix,
    #     training_pose_list,
    #     bg_color=bg_color,
    # )

    # * #############################################################################
    # key_time_step = cams.T // 2
    q = torch.tensor( # w, x, y, z
        [
            0.0014498578768367928,
            -0.005260779557546396,
            0.9640436120785223,
            -0.265688042864521
        ],
    )
    rotation_mat = q2R(q)
    camera_position = torch.tensor(
        [
            -0.02440260125266045,
            2.235835112227348,
            3.7484688351199478
        ],
    )

    R_inv = rotation_mat.T
    t_inv = -R_inv @ camera_position

    fixed_pose = torch.eye(4)
    fixed_pose[:3, :3] = R_inv
    fixed_pose[:3, 3] = t_inv
    fixed_pose = fixed_pose.to(cams.T_cw(0).device)

    for key_time_step in key_steps:
        # fixed_pose_list = [cams.T_cw(key_time_step) for _ in range(cams.T)]
        fixed_pose_list = [fixed_pose for _ in range(cams.T)]
        round_pose_list = get_move_around_cam_T_cw(
            s_model,
            d_model,
            cams,
            H,
            W,
            np.deg2rad(move_angle_deg),
            total_steps=cams.T,  # cams.T
            center_id=key_time_step,
        )

        # # viz flow
        # save_fn_prefix = osp.join(save_dir, f"{key_time_step}_fixed_moving_flow")
        # viz_single_2d_flow_video(
        #     H, W, cams, s_model, d_model, save_fn_prefix, fixed_pose_list
        # )
        # save_fn_prefix = osp.join(save_dir, f"{key_time_step}_round_moving_flow")
        # viz_single_2d_flow_video(
        #     H, W, cams, s_model, d_model, save_fn_prefix, round_pose_list
        # )
        # # Viz node
        # save_fn_prefix = osp.join(save_dir, f"{key_time_step}_round_moving_node")
        # viz_single_2d_node_video(
        #     H, W, cams, s_model, d_model, save_fn_prefix, round_pose_list
        # )
        # save_fn_prefix = osp.join(save_dir, f"{key_time_step}_round_freezing_node")
        # viz_single_2d_node_video(
        #     H,
        #     W,
        #     cams,
        #     s_model,
        #     d_model,
        #     save_fn_prefix,
        #     round_pose_list,
        #     model_t=key_time_step,
        #     bg_color=bg_color,
        # )
        # save_fn_prefix = osp.join(save_dir, f"{key_time_step}_fixed_moving_node")
        # viz_single_2d_node_video(
        #     H, W, cams, s_model, d_model, save_fn_prefix, fixed_pose_list
        # )
        # Viz rgb
        save_fn_prefix = osp.join(save_dir, f"{key_time_step}_round_moving")
        # viz_single_2d_video(
        #     H,
        #     W,
        #     cams,
        #     s_model,
        #     d_model,
        #     save_fn_prefix,
        #     round_pose_list,
        #     bg_color=bg_color,
        #     bg_flag=False,
        # )
        save_fn_prefix = osp.join(save_dir, f"{key_time_step}_round_freezing")
        # viz_single_2d_video(
        #     H,
        #     W,
        #     cams,
        #     s_model,
        #     d_model,
        #     save_fn_prefix,
        #     round_pose_list,
        #     model_t=key_time_step,
        #     bg_color=bg_color,
        #     bg_flag=False,
        # )
        save_fn_prefix = osp.join(save_dir, f"{key_time_step}_fixed_moving")
        viz_single_2d_video(
            H,
            W,
            cams,
            s_model,
            d_model,
            save_fn_prefix,
            fixed_pose_list,
            bg_color=bg_color,
            bg_flag=False,
        )

    return


if __name__ == "__main__":

    import argparse

    args = argparse.ArgumentParser()
    args.add_argument("--cfg", "-c", type=str, required=True)
    args.add_argument("--logdir", "-r", type=str, required=True)
    args.add_argument("--savedir", "-s", type=str, required=True)
    args.add_argument("--N", "-n", type=int, default=20)
    args.add_argument("--move_angle_deg", "-m", type=float, default=270.0)
    args = args.parse_args()

    viz_main(
        args.savedir,
        args.logdir,
        args.cfg,
        N=args.N,
        move_angle_deg=args.move_angle_deg,
    )
