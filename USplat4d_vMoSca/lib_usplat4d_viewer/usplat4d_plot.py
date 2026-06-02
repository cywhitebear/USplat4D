import torch
import numpy as np
import math
import plotly.graph_objects as go
import plotly.express as px
from plotly.express import colors as _c
from eval_utils.campose_alignment import align_ate_c2b_use_a2b
from lib_usplat4d_viewer.usplat4d_utils import test_camera_distance, test_camera_angle
import matplotlib.pyplot as plt


def plot_camera_trajectories(pose_seqs, scale=0.1, ax=None, cmap="tab10", colors=None):
    """
    Plot one or more camera trajectories, each in its own color.

    Args:
        pose_seqs: list of (N_i x 4 x 4) arrays or torch.Tensors,
                   each is a camera-to-world sequence.
        scale:     float, length of the little axis arrows.
        ax:        an existing Matplotlib 3D axis (optional).
        cmap:      name of a Matplotlib colormap for automatically
                   choosing colors if `colors` is None.
                   [blue, orange, green, red]
        colors:    list of colors (any Matplotlib‐acceptable format).
                   If given, length must == len(pose_seqs).

    Returns:
        The Matplotlib 3D axis with your plot.
    """
    # convert tensors->numpy and ensure list
    seqs = []
    for P in pose_seqs:
        if not isinstance(P, np.ndarray):
            # assume torch
            P = P.detach().cpu().numpy()
        seqs.append(P)

    n = len(seqs)
    # pick or generate colors
    if colors is None:
        cmap = plt.get_cmap(cmap)
        colors = [cmap(i) for i in range(n)]
    elif len(colors) != n:
        raise ValueError(f"`colors` length ({len(colors)}) != number of sequences ({n})")

    # prep axis
    if ax is None:
        fig = plt.figure(figsize=(8,8))
        ax  = fig.add_subplot(111, projection='3d')

    # for each sequence
    for seq, col in zip(seqs, colors):
        Rs = seq[:, :3, :3]
        ts = seq[:, :3, 3]

        # scatter all camera centers of this seq
        ax.scatter(ts[:,0], ts[:,1], ts[:,2], c=[col], marker='o', s=20, label=None)

        # draw a little triad at each camera
        for R, t in zip(Rs, ts):
            # camera axes in world frame
            x_axis = R.T @ np.array([1,0,0])
            y_axis = R.T @ np.array([0,1,0])
            z_axis = R.T @ np.array([0,0,1])
            for vec in (x_axis, y_axis, z_axis):
                ax.plot(
                    [t[0], t[0]+scale*vec[0]],
                    [t[1], t[1]+scale*vec[1]],
                    [t[2], t[2]+scale*vec[2]],
                    color=col,
                    linewidth=1
                )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect((1,1,1))
    return ax

import numpy as np

def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres look like spheres,
    cubes look like cubes, etc.

    Call after plotting your data.
    """
    # Grab current limits
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    # Compute ranges
    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]
    max_range = max(x_range, y_range, z_range)

    # Compute mid‐points
    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)

    # Set all axes to cover the same span
    half = max_range / 2
    ax.set_xlim3d(x_mid - half, x_mid + half)
    ax.set_ylim3d(y_mid - half, y_mid + half)
    ax.set_zlim3d(z_mid - half, z_mid + half)



def add_camera_frustum(
    fig,         # your go.Figure
    t,           # 3-vector camera center
    R,           # 3×3 rotation (camera-to-world)
    scale=0.1,   # overall size of both cone and plane
    color="orange",  # trajectory color
    plane_size=(0.05, 0.03),  # (width, height) of image-plane rectangle
    plane_dist=0.2         # how far in front of the camera to place that plane
):
    # 1) Cone pointing along camera’s local +Z
    forward = R.T @ np.array([0,0,1])
    fig.add_trace(go.Cone(
        x=[t[0]], y=[t[1]], z=[t[2]],
        u=[forward[0]], v=[forward[1]], w=[forward[2]],
        sizemode="absolute",
        sizeref=scale * 0.5,
        anchor="tip",
        showscale=False,
        colorscale=[[0, color], [1, color]],
        name="",
        showlegend=False
    ))

    # 2) little “image-plane” rectangle at z=plane_dist
    w, h = plane_size
    corners_cam = np.array([
        [-w/2, -h/2, plane_dist],
        [ w/2, -h/2, plane_dist],
        [ w/2,  h/2, plane_dist],
        [-w/2,  h/2, plane_dist],
    ])
    corners_world = (R.T @ corners_cam.T).T + t[None,:]

    fig.add_trace(go.Mesh3d(
        x=corners_world[:,0],
        y=corners_world[:,1],
        z=corners_world[:,2],
        i=[0,1,2, 0,2,3],
        j=[1,2,3, 2,3,0],
        k=[2,3,0, 3,0,1],
        color=color,
        opacity=0.5,
        showscale=False,
        name="",
        showlegend=False
    ))

def plot_trajectories_plotly(c2w_poses_list, scale=0.1, colors=None):
    """
    Plot one or more camera‐to‐world trajectories in a single 3D figure,
    using a single color per trajectory, and represent each camera as a cone
    + a small rectangular image‐plane.
    """
    if colors is None:
        colors = px.colors.qualitative.Plotly

    fig = go.Figure()
    all_ts = []

    for idx, c2w in enumerate(c2w_poses_list):
        if not isinstance(c2w, np.ndarray):
            # assume torch
            c2w = c2w.detach().cpu().numpy()
        Rs = c2w[:, :3, :3]
        ts = c2w[:, :3, 3]
        all_ts.append(ts)

        traj_color = colors[idx % len(colors)]

        # draw trajectory path
        fig.add_trace(go.Scatter3d(
            x=ts[:,0], y=ts[:,1], z=ts[:,2],
            mode='markers+lines',
            marker=dict(size=3, color=traj_color),
            line=dict(color=traj_color, width=2),
            name=f"traj {idx}"
        ))

        # draw each camera as cone+plane
        for R, t in zip(Rs, ts):
            add_camera_frustum(
                fig, t, R,
                scale=scale,
                color=traj_color,
                plane_size=(scale*0.5, scale*0.3),
                plane_dist=scale*0.8
            )

    # global bounds & cubic aspect
    all_ts = np.vstack(all_ts)
    mins, maxs = all_ts.min(axis=0), all_ts.max(axis=0)
    span = maxs - mins
    pad = 0.05 * span.max()
    xr = [mins[0]-pad, maxs[0]+pad]
    yr = [mins[1]-pad, maxs[1]+pad]
    zr = [mins[2]-pad, maxs[2]+pad]

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=xr, autorange=False),
            yaxis=dict(range=yr, autorange=False),
            zaxis=dict(range=zr, autorange=False),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    return fig

def plot_camera_frustums(
        c2w_poses_list, 
        object_center=None, # center of object to be visualized
        depth=0.1,       # distance of image‐plane from camera
        fov_deg=30.0,    # vertical field of view
        aspect=4/3,      # w/h of image plane
        colors=None, 
        dataset_name="iPhone"
    ):
    """
    c2w_poses_list: list of (Nx4×4) camera‐to‐world matrices
    depth:    how far out to put the image plane
    fov_deg:  vertical field of view in degrees
    aspect:   width/height ratio of the plane
    """
    if colors is None:
        colors = _c.qualitative.Plotly

    fig = go.Figure()
    verts = []  # for collecting all points so we can set a nice cubical camera

    for i, c2w_poses in enumerate(c2w_poses_list): # c2w is camera_poses, ex. shape (n, 4, 4)
        if not isinstance(c2w_poses, np.ndarray):
            c2w_poses = c2w_poses.detach().cpu().numpy() 
        
        for c2w_pose in c2w_poses:
            # extract rotation and translation
            R = c2w_pose[:3, :3]
            t = c2w_pose[:3, 3]

            # compute half‐height & half‐width of plane at distance 'depth'
            h2 = np.tan(np.deg2rad(fov_deg/2)) * depth
            w2 = h2 * aspect

            # define the four corners of the image plane in camera coordinates
            plane_cam = np.array([
                [+w2, +h2, depth],
                [-w2, +h2, depth],
                [-w2, -h2, depth],
                [+w2, -h2, depth],
            ])  # shape (4,3)

            # transform into world coords
            plane_world = (R @ plane_cam.T).T + t[None,:]

            color = colors[i % len(colors)]

            # draw the four pyramid edges (camera center → each corner)
            for corner in plane_world:
                fig.add_trace(go.Scatter3d(
                    x=[t[0], corner[0]],
                    y=[t[1], corner[1]],
                    z=[t[2], corner[2]],
                    mode='lines',
                    line=dict(color=color, width=0.5),
                    showlegend=False
                ))
 
            # draw the rectangle edges of the plane
            for j in range(4):
                k = (j+1) % 4
                fig.add_trace(go.Scatter3d(
                    x=[plane_world[j,0], plane_world[k,0]],
                    y=[plane_world[j,1], plane_world[k,1]],
                    z=[plane_world[j,2], plane_world[k,2]],
                    mode='lines',
                    line=dict(color=color, width=0.5),
                    showlegend=False
                ))

            # add a text label at the camera center
            fig.add_trace(go.Scatter3d(
                x=[t[0]], y=[t[1]], z=[t[2]],
                mode='text',
                text=[str(i)],
                textposition='top center',
                showlegend=False
            ))

            verts.append(t)
            verts.append(plane_world.min(axis=0))
            verts.append(plane_world.max(axis=0))

    # compute global axis limits so the aspect is equal
    all_pts = np.vstack(verts)
    # add object center to the list of points
    if object_center is not None:
        if isinstance(object_center, torch.Tensor):
            object_center = object_center.detach().cpu().numpy()  # Convert tensor to numpy
        all_pts = np.vstack([all_pts, object_center])
    mins, maxs = all_pts.min(axis=0), all_pts.max(axis=0)
    span = maxs - mins
    pad = 0.1 * span.max()
    xr = [mins[0]-pad, maxs[0]+pad]
    yr = [mins[1]-pad, maxs[1]+pad]
    zr = [mins[2]-pad, maxs[2]+pad]

    if object_center is not None:
        if isinstance(object_center, torch.Tensor):
            object_center = object_center.detach().cpu().numpy()  # Convert tensor to numpy
        # Draw a large sphere centered at object_center
        sphere_radius = 0.05 * (span.max())  # Adjust 0.2 as needed for visual size

        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x = object_center[0] + sphere_radius * np.outer(np.cos(u), np.sin(v))
        y = object_center[1] + sphere_radius * np.outer(np.sin(u), np.sin(v))
        z = object_center[2] + sphere_radius * np.outer(np.ones_like(u), np.cos(v))

        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            opacity=0.5,  # semi-transparent
            colorscale='Viridis',
            showscale=False,
        ))

    # if dataset_name == "iPhone":
    #     tick_unit = 0.5 
    # else:
    #     tick_unit = 0.1 * 19

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=xr, autorange=False, tick0=0.0),
            yaxis=dict(range=yr, autorange=False, tick0=0.0),
            zaxis=dict(range=zr, autorange=False, tick0=0.0),
            aspectmode='cube', 
            camera=dict(
                projection=dict(type='orthographic'),
            ),
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )
    return fig

def tfm_pose_sim3_and_se3(aligned_test_camera_path, solved_cam_path, train_camera_path):
    """
    Convert a 4x4 transformation matrix to a 6D vector (SE3).
    :param aligned_test_camera_T_wi_i: The camera pose from the aligned test model.
    :return: A 6D vector representing the SE3 transformation.
    """
    # Extract rotation and translation
    aligned_test_camera_T_wi = torch.load(aligned_test_camera_path)
    solved_cam_T_wi = torch.load(solved_cam_path)
    train_camera_T_wi = torch.load(train_camera_path)

    new_SE3_aligned_camera = align_ate_c2b_use_a2b(
        traj_a=aligned_test_camera_T_wi,
        traj_b=solved_cam_T_wi.detach().cpu(),
        traj_c=aligned_test_camera_T_wi
    )
    
    fig = plot_camera_frustums(
        [aligned_test_camera_T_wi, solved_cam_T_wi, new_SE3_aligned_camera],
        depth=0.1,
        fov_deg=30.0,
        aspect=4/3,
        colors=["blue", "red", "green"], 
        dataset_name="Davis"
    )
    fig.write_html("traj_se3.html")

def main():
    # backpack
    # solved_cam_T_wi = torch.tensor([
    #     [ 9.9999e-01,  6.5029e-04, -3.2435e-03, -8.4832e-04],
    #     [ 6.5222e-04, -1.0000e+00,  5.9420e-04, -6.8465e-03],
    #     [-3.2431e-03, -5.9631e-04, -9.9999e-01,  1.9007e-03],
    #     [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]
    # ], device='cuda:0')

    # aligned_test_camera_T_wi = torch.tensor([
    #     [ 9.9992e-01,  7.4238e-03, -1.0082e-02, -6.7736e-04],
    #     [ 7.8025e-03, -9.9924e-01,  3.8060e-02,  4.0515e-03],
    #     [-9.7923e-03, -3.8136e-02, -9.9922e-01,  6.7043e-04],
    #     [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]
    # ], device='cuda:0')

    solved_cam_T_wi= torch.tensor([[ 9.9990e-01, -1.4276e-02, -3.1325e-04, -6.5773e-03],
        [ 1.4273e-02,  9.9986e-01, -8.3347e-03,  3.9866e-03],
        [ 4.3220e-04,  8.3294e-03,  9.9997e-01, -3.3767e-03],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]
    ], device='cuda:0')
    aligned_test_camera_T_wi = torch.tensor([[ 4.5901e-01,  5.9657e-01, -6.5835e-01,  1.0165e-02],
        [ 8.3455e-01, -5.4367e-01,  8.9209e-02,  5.9640e-04],
        [-3.0470e-01, -5.9037e-01, -7.4741e-01,  1.1850e-02],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]
    ], device='cuda:0')
    test_camera_distance(solved_cam_T_wi, aligned_test_camera_T_wi)

    # camel
    # solved_cam_T_wi = torch.tensor([
    #     [ 0.9999, -0.0153, -0.0071, -0.0073],
    #     [ 0.0153,  0.9999, -0.0071,  0.0040],
    #     [ 0.0072,  0.0070,  1.0000, -0.0039],
    #     [ 0.0000,  0.0000,  0.0000,  1.0000]
    # ], device='cuda:0')

    # aligned_test_camera_T_wi = torch.tensor([
    #     [ 0.4544,  0.5975, -0.6607, -0.0053],
    #     [ 0.8352, -0.5438,  0.0826,  0.0031],
    #     [-0.3100, -0.5893, -0.7461, -0.0013],
    #     [ 0.0000,  0.0000,  0.0000,  1.0000]
    # ], device='cuda:0')


if __name__ == "__main__":
    # main()
    tfm_pose_sim3_and_se3(
        aligned_test_camera_path="aligned_test_camera_T_wi.pth",
        solved_cam_path="solved_cam_T_wi.pth",
        train_camera_path="train_camera_T_wi.pth"
    )