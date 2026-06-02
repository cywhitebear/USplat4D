import os, sys, os.path as osp
import torch

sys.path.append(osp.dirname(osp.abspath(__file__)))
# sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), ".."))
from sh_utils import eval_sh
import torch.nn.functional as F
from gsplat.rendering import rasterization

# get from env variable
try:
    GS_BACKEND = os.environ["GS_BACKEND"]
except:
    # GS_BACKEND = "native"
    print(f"No GS_BACKEND env var specified, for now use native_add3 backend")
    GS_BACKEND = "native_add3"
GS_BACKEND = GS_BACKEND.lower()
print(f"GS_BACKEND: {GS_BACKEND.lower()}")

if GS_BACKEND == "native":
    from gauspl_renderer_native import render_cam_pcl
elif GS_BACKEND == "gof":
    from gauspl_renderer_gof import render_cam_pcl
elif GS_BACKEND == "native_add3":
    from gauspl_renderer_native_add3 import render_cam_pcl
else:
    raise ValueError(f"Unknown GS_BACKEND: {GS_BACKEND.lower()}")
from sh_utils import RGB2SH, SH2RGB


def render(
    gs_param,
    H,
    W,
    K,
    T_cw,
    bg_color=[1.0, 1.0, 1.0],
    scale_factor=1.0,
    opa_replace=None,
    bg_cache_dict=None,
    colors_precomp=None,
    add_buffer=None,
):
    # * Core render interface
    # prepare gs5 param in world system
    if torch.is_tensor(gs_param[0]):  # direct 5 tuple
        mu, fr, s, o, sph = gs_param
    else:
        mu, fr, s, o, sph = gs_param[0]
        for i in range(1, len(gs_param)):
            mu = torch.cat([mu, gs_param[i][0]], 0)
            fr = torch.cat([fr, gs_param[i][1]], 0)
            s = torch.cat([s, gs_param[i][2]], 0)
            o = torch.cat([o, gs_param[i][3]], 0)
            sph = torch.cat([sph, gs_param[i][4]], 0)
    if opa_replace is not None:
        assert isinstance(opa_replace, float)
        o = torch.ones_like(o) * opa_replace
    s = s * scale_factor

    # cvt to cam frame
    assert T_cw.ndim == 2 and T_cw.shape[1] == T_cw.shape[0] == 4
    R_cw, t_cw = T_cw[:3, :3], T_cw[:3, 3]
    mu_cam = torch.einsum("ij,nj->ni", R_cw, mu) + t_cw[None]
    fr_cam = torch.einsum("ij,njk->nik", R_cw, fr)

    # render
    render_dict = render_cam_pcl(
        mu_cam,
        fr_cam,
        s,
        o,
        sph,
        H,
        W,
        CAM_K=K,
        bg_color=bg_color,
        colors_precomp=colors_precomp,
        add_buffer=add_buffer,
    )

    if bg_cache_dict is not None:
        render_dict = fast_bg_compose_render(bg_cache_dict, render_dict, bg_color)
    return render_dict

def render_gsplat(
    gs_param,
    H,
    W,
    K,
    T_cw,
    bg_color=[1.0, 1.0, 1.0],
    scale_factor=1.0,
    opa_replace=None,
    bg_cache_dict=None,
    colors_precomp=None,
    add_buffer=None,
    n_fg=None,
):
    # * Core render interface
    # prepare gs5 param in world system
    if torch.is_tensor(gs_param[0]):  # direct 5 tuple
        mu, fr, s, o, sph = gs_param
    else:
        mu, fr, s, o, sph = gs_param[0]
        for i in range(1, len(gs_param)):
            mu = torch.cat([mu, gs_param[i][0]], 0)
            fr = torch.cat([fr, gs_param[i][1]], 0)
            s = torch.cat([s, gs_param[i][2]], 0)
            o = torch.cat([o, gs_param[i][3]], 0)
            sph = torch.cat([sph, gs_param[i][4]], 0)
    if opa_replace is not None:
        assert isinstance(opa_replace, float)
        o = torch.ones_like(o) * opa_replace
    s = s * scale_factor

    # cvt to cam frame
    assert T_cw.ndim == 2 and T_cw.shape[1] == T_cw.shape[0] == 4
    R_cw, t_cw = T_cw[:3, :3], T_cw[:3, 3]
    mu_cam = torch.einsum("ij,nj->ni", R_cw, mu) + t_cw[None]
    fr_cam = torch.einsum("ij,njk->nik", R_cw, fr)

    render_dict={}

    # # render
    # render_dict = render_cam_pcl(
    #     mu_cam,
    #     fr_cam,
    #     s,
    #     o,
    #     sph,
    #     H,
    #     W,
    #     CAM_K=K,
    #     bg_color=bg_color,
    #     colors_precomp=colors_precomp,
    #     add_buffer=add_buffer,
    # )
    # dict_keys(['rgb', 'buf', 'dep', 'alpha', 'viewspace_points', 'visibility_filter', 'radii'])
    # 'rgb': [3,H,W]
    # 'buf': [3,H,W]
    # 'depth': [1,H,W]
    # 'alpha': [1,H,W]

    # gsplat version
    means=mu
    quats=rotation_matrix_to_quaternion(fr) # [N,4]
    scales=s
    opacities=o.squeeze(-1)
    # sh to color
    color_feat=sph
    dir_cam = F.normalize(mu_cam, dim=-1)
    # P_w = Frame @ P_local
    dir_local = torch.einsum("nji,nj->ni", fr_cam, dir_cam)  # note the transpose
    dir_local = F.normalize(
        dir_local, dim=-1
    )  # If frame is not SO(3) but Affinity, have to normalize
    N = len(color_feat)
    shs_view = color_feat.reshape(N, -1, 3)  # N, Deg, Channels
    _deg = shs_view.shape[1]
    if _deg == 1:
        active_sph_order = 0
    elif _deg == 4:
        active_sph_order = 1
    elif _deg == 9:
        active_sph_order = 2
    elif _deg == 16:
        active_sph_order = 3
    else:
        raise ValueError(f"Unexpected SH degree: {_deg}")
    sh2rgb = eval_sh(active_sph_order, shs_view.permute(0, 2, 1), dir_local)

    if colors_precomp is None:
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
    else:
        assert (
            colors_precomp.shape == sh2rgb.shape
        ), f"{colors_precomp.shape} != {sh2rgb.shape}"
    # ! compute spherical color
    # colors_precomp = color_feat
    colors_override=colors_precomp
    bg_color2=torch.tensor(bg_color,device=colors_precomp.device).unsqueeze(0).repeat(1,1) # [1, 3]
    w2cs=T_cw.unsqueeze(0) # [1, 4, 4]
    Ks=K.unsqueeze(0) # [1, 3, 3]
    # render_colors, alphas, info, contribs, h_data = rasterization(
    render_colors, alphas, info, contribs= rasterization(
        # render_colors, alphas, info = rasterization(
            means=means, # t_w
            quats=quats, # q_w [wxyz]
            scales=scales,
            opacities=opacities,
            colors=colors_override,
            backgrounds=bg_color2, # [C,3]
            viewmats=w2cs,  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=W,
            height=H,
            packed=False,
            # render_mode="RGB+ED+UCD",
            render_mode="RGB+ED",
        )
    # render_colors [C,H,W,8]
    render_dict['rgb']=render_colors[0,:,:,:3].permute(2,0,1) # [3,H,W] # check if the rgb is the same.
    render_dict['dep']=render_colors[0,:,:,3].unsqueeze(-1).permute(2,0,1) # [1,H,W]
    render_dict['alpha']=alphas[0,:,:,:].permute(2,0,1) # [1,H,W]
    render_dict["visibility_filter"]=None
    render_dict["viewspace_points"]=None
    render_dict["radii"]=None
    # alphas # [C,H,W,1]

    out_dict={}
    out_dict["img"] = render_colors[:,:,:,:3] # [C,H,W,3]
    out_dict["acc"] = alphas
    if n_fg is not None:
        out_dict["contribs"] = contribs[-n_fg*3:] # my code # TODO: for memory
        out_dict["means"] = means[-n_fg:]
        out_dict["quats"] = quats[-n_fg:]
    if bg_cache_dict is not None:
        render_dict = fast_bg_compose_render(bg_cache_dict, render_dict, bg_color)
    return render_dict, out_dict


def fast_bg_compose_render(bg_cache_dict, render_dict, bg_color=[1.0, 1.0, 1.0]):
    assert GS_BACKEND == "native", "GOF does not support this now"

    # manually compose the fg
    # ! warning, be careful when use the visibility masks .etc, watch the len
    fg_rgb, bg_rgb = render_dict["rgb"], bg_cache_dict["rgb"]
    fg_alpha, bg_alpha = render_dict["alpha"], bg_cache_dict["alpha"]
    fg_dep, bg_dep = render_dict["dep"], bg_cache_dict["dep"]
    _fg_alp = torch.clamp(fg_alpha, 1e-8, 1.0)
    _bg_alp = torch.clamp(bg_alpha, 1e-8, 1.0)
    fg_dep_corr = fg_dep / _fg_alp
    bg_dep_corr = bg_dep / _bg_alp
    fg_in_front = (fg_dep_corr < bg_dep_corr).float()
    # compose alpha
    alpha_fg_front_compose = fg_alpha + (1.0 - fg_alpha) * bg_alpha
    alpha_fg_behind_compose = bg_alpha + (1.0 - bg_alpha) * fg_alpha
    alpha_composed = alpha_fg_front_compose * fg_in_front + alpha_fg_behind_compose * (
        1.0 - fg_in_front
)
    alpha_composed = torch.clamp(alpha_composed, 0.0, 1.0)
    # compose rgb
    bg_color = torch.as_tensor(bg_color, device=fg_rgb.device, dtype=fg_rgb.dtype)
    rgb_fg_front_compose = (
        fg_rgb * fg_alpha
        + bg_rgb * (1.0 - fg_alpha) * bg_alpha
        + (1.0 - fg_alpha) * (1.0 - bg_alpha) * bg_color[:, None, None]
    )
    rgb_fg_behind_compose = (
        bg_rgb * bg_alpha
        + fg_rgb * (1.0 - bg_alpha) * fg_alpha
        + (1.0 - bg_alpha) * (1.0 - fg_alpha) * bg_color[:, None, None]
    )
    rgb_composed = rgb_fg_front_compose * fg_in_front + rgb_fg_behind_compose * (
        1.0 - fg_in_front
    )
    # compose dep
    dep_fg_front_compose = (
        fg_dep_corr * fg_alpha + bg_dep_corr * (1.0 - fg_alpha) * bg_alpha
    )
    dep_fg_behind_compose = (
        bg_dep_corr * bg_alpha + fg_dep_corr * (1.0 - bg_alpha) * fg_alpha
    )
    dep_composed = dep_fg_front_compose * fg_in_front + dep_fg_behind_compose * (
        1.0 - fg_in_front
    )
    return {
        "rgb": rgb_composed,
        "dep": dep_composed,
        "alpha": alpha_composed,
        "visibility_filter": render_dict["visibility_filter"],
        "viewspace_points": render_dict["viewspace_points"],
        "radii": render_dict["radii"],
        "dyn_rgb": fg_rgb,
        "dyn_dep": fg_dep,
        "dyn_alpha": fg_alpha,
    }

def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrices [N, 3, 3] to quaternions [N, 4] (w, x, y, z).
    
    Args:
        R (torch.Tensor): shape (N, 3, 3)
        
    Returns:
        torch.Tensor: shape (N, 4)
    """
    assert R.shape[-2:] == (3, 3)

    N = R.shape[0]

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    qw = torch.empty(N, device=R.device)
    qx = torch.empty(N, device=R.device)
    qy = torch.empty(N, device=R.device)
    qz = torch.empty(N, device=R.device)

    mask = trace > 0

    # Case 1: trace > 0
    s = torch.sqrt(trace[mask] + 1.0) * 2  # s=4*qw
    qw[mask] = 0.25 * s
    qx[mask] = (R[mask, 2, 1] - R[mask, 1, 2]) / s
    qy[mask] = (R[mask, 0, 2] - R[mask, 2, 0]) / s
    qz[mask] = (R[mask, 1, 0] - R[mask, 0, 1]) / s

    # Case 2: trace <= 0
    mask0 = (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2]) & (~mask)
    mask1 = (R[:, 1, 1] > R[:, 2, 2]) & (~mask) & (~mask0)
    mask2 = (~mask) & (~mask0) & (~mask1)

    s0 = torch.sqrt(1.0 + R[mask0, 0, 0] - R[mask0, 1, 1] - R[mask0, 2, 2]) * 2  # s=4*qx
    qw[mask0] = (R[mask0, 2, 1] - R[mask0, 1, 2]) / s0
    qx[mask0] = 0.25 * s0
    qy[mask0] = (R[mask0, 0, 1] + R[mask0, 1, 0]) / s0
    qz[mask0] = (R[mask0, 0, 2] + R[mask0, 2, 0]) / s0

    s1 = torch.sqrt(1.0 + R[mask1, 1, 1] - R[mask1, 0, 0] - R[mask1, 2, 2]) * 2  # s=4*qy
    qw[mask1] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s1
    qx[mask1] = (R[mask1, 0, 1] + R[mask1, 1, 0]) / s1
    qy[mask1] = 0.25 * s1
    qz[mask1] = (R[mask1, 1, 2] + R[mask1, 2, 1]) / s1

    s2 = torch.sqrt(1.0 + R[mask2, 2, 2] - R[mask2, 0, 0] - R[mask2, 1, 1]) * 2  # s=4*qz
    qw[mask2] = (R[mask2, 1, 0] - R[mask2, 0, 1]) / s2
    qx[mask2] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s2
    qy[mask2] = (R[mask2, 1, 2] + R[mask2, 2, 1]) / s2
    qz[mask2] = 0.25 * s2

    quat = torch.stack([qw, qx, qy, qz], dim=1)  # (N, 4)
    return quat


# def matrix_to_quaternion(T: torch.Tensor) -> torch.Tensor:
#     """
#     Convert a [4,4] transformation matrix to a quaternion (wxyz).

#     Args:
#         T (torch.Tensor): shape (4,4) transformation matrix.

#     Returns:
#         torch.Tensor: shape (4,) quaternion in (w, x, y, z) order.
#     """
#     R = T[:3, :3]  # Extract rotation part

#     # Compute quaternion manually
#     trace = R.trace()
#     if trace > 0:
#         s = torch.sqrt(trace + 1.0) * 2  # s = 4 * qw
#         qw = 0.25 * s
#         qx = (R[2,1] - R[1,2]) / s
#         qy = (R[0,2] - R[2,0]) / s
#         qz = (R[1,0] - R[0,1]) / s
#     elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
#         s = torch.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2  # s = 4 * qx
#         qw = (R[2,1] - R[1,2]) / s
#         qx = 0.25 * s
#         qy = (R[0,1] + R[1,0]) / s
#         qz = (R[0,2] + R[2,0]) / s
#     elif R[1,1] > R[2,2]:
#         s = torch.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2  # s = 4 * qy
#         qw = (R[0,2] - R[2,0]) / s
#         qx = (R[0,1] + R[1,0]) / s
#         qy = 0.25 * s
#         qz = (R[1,2] + R[2,1]) / s
#     else:
#         s = torch.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2  # s = 4 * qz
#         qw = (R[1,0] - R[0,1]) / s
#         qx = (R[0,2] + R[2,0]) / s
#         qy = (R[1,2] + R[2,1]) / s
#         qz = 0.25 * s

#     quat = torch.stack([qw, qx, qy, qz])
#     return quat
