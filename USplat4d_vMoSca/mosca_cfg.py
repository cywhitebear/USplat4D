import os.path as osp
import torch

from omegaconf import OmegaConf
from lib_moca.camera import MonocularCameras
from lib_mosca.dynamic_gs import DynSCFGaussian
from lib_mosca.static_gs import StaticGaussian
from lib_render.render_helper import GS_BACKEND
from lib_ugraph.ugraph_gs import UGraphGaussian

import logging


def load_model_cfg(cfg, saved_dir, device="cuda", is_eval=True, use_ugraph=True):

    # get cfg
    if saved_dir.endswith("/"):
        saved_dir = saved_dir[:-1]
    if isinstance(cfg, str):
        cfg = OmegaConf.load(cfg)
        # OmegaConf.set_readonly(cfg, True)
    dataset_mode = getattr(cfg, "mode", "iphone")
    print(f"Dataset mode: {dataset_mode}")
    max_sph_order = getattr(cfg, "max_sph_order", 1)
    logging.info(f"Dataset mode: {dataset_mode}")

    ######################################################################
    ######################################################################

    cams = MonocularCameras.load_from_ckpt(
        torch.load(osp.join(saved_dir, "photometric_cam.pth"))
    ).to(device)
    try:
        s_model = StaticGaussian.load_from_ckpt(
            torch.load(
                osp.join(saved_dir, f"photometric_s_model_{GS_BACKEND.lower()}.pth")
            ),
            device=device,
        )
    except:
        s_model = None
        logging.info("Static Gaussians disabled - skipping static model creation")
    d_model = DynSCFGaussian.load_from_ckpt(
        torch.load(
            osp.join(saved_dir, f"photometric_d_model_{GS_BACKEND.lower()}.pth")
        ),
        device=device,
    )

    cams.to(device)
    d_model.to(device)
    try:
        s_model.to(device)
    except:
        pass

    if use_ugraph:
        print("Start loading ugraph model...")
        state_dict = torch.load(osp.join(saved_dir, f"photometric_ugraph_model_{GS_BACKEND.lower()}.pth"))
        ugraph_model: UGraphGaussian = UGraphGaussian.init_from_state_dict(state_dict, None, device)
        ugraph_model.to(device)
        d_model.set_ugraph(ugraph_model)
        print("Done loading ugraph model.")

    if is_eval:
        cams.eval()
        d_model.eval()
        try:
            s_model.eval()
        except:
            pass
        if use_ugraph:
            ugraph_model.eval()

    return cfg, d_model, s_model, cams