"""4-way live comparison viewer (baseline / observability / render_contrib / usplat4d).

One viser server, one shared client camera + playback time. Each frame renders all
loaded variants from the SAME viewpoint and tiles them into a 2x2 grid so the
reconstructions can be compared side by side while orbiting.

Usage (run by user; visualization is user-launched):
    python -m lib_usplat4d_viewer.compare_4way_viewer \
        --cfg_fn ./profile/iphone/iphone_fit.yaml \
        --spin_root /media/ee904/DATA1/Yun/Datasets/MoSca/iphone/spin \
        --port 8899

Auto-discovers the 4 spin variants' photometric checkpoints; any missing variant is
simply skipped (grid shrinks). Override any path with --<name>_dir / --<name>_pth.
"""
import argparse
import glob
import os
import os.path as osp
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib_usplat4d_viewer.usplat4d_viser import get_server
from lib_usplat4d_viewer.usplat4d_viewer import DynamicViewer
from lib_render.render_helper import render
from lib_mosca.dynamic_gs import DynSCFGaussian
from lib_moca.camera import MonocularCameras

try:
    import cv2

    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


def _newest(pattern):
    hits = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return hits[0] if hits else None


def discover_variants(spin_root):
    """Return ordered list of (label, d_model_pth, cam_pth) for the 4 variants."""
    base_log = osp.join(spin_root, "logs", "iphone_fit_native_add3")
    logs2 = osp.join(spin_root, "logs2")
    out = []

    # baseline: June photometric (root, no suffix)
    d = osp.join(base_log, "photometric_d_model_native_add3.pth")
    c = osp.join(base_log, "photometric_cam.pth")
    out.append(("baseline", d, c))

    # ours (mode1, logs2/<exp>_<ts>/): newest per exp_name
    for label, exp in [
        ("observability", "iphone_fit_uar"),
        ("render_contrib", "iphone_fit_uar_ru"),
    ]:
        run_dir = _newest(osp.join(logs2, f"{exp}_native_add3_*"))
        if run_dir is None:
            out.append((label, None, None))
            continue
        out.append(
            (
                label,
                osp.join(run_dir, "photometric_d_model_native_add3.pth"),
                osp.join(run_dir, "photometric_cam.pth"),
            )
        )

    # usplat4d (mode3 ugraph, root, _ugraph suffix)
    out.append(
        (
            "usplat4d",
            osp.join(base_log, "photometric_d_model_native_add3_ugraph.pth"),
            osp.join(base_log, "photometric_cam_ugraph.pth"),
        )
    )
    return out


class Compare4WayViewer:
    def __init__(self, cfg_fn, spin_root, device, port=8899, overrides=None):
        self.device = device
        self.models = []  # list of (label, d_model, cams)
        overrides = overrides or {}

        for label, d_pth, c_pth in discover_variants(spin_root):
            d_pth = overrides.get(f"{label}_pth", d_pth)
            c_pth = overrides.get(f"{label}_cam", c_pth)
            if not (d_pth and c_pth and osp.exists(d_pth) and osp.exists(c_pth)):
                print(f"[skip] {label}: missing ckpt ({d_pth})")
                continue
            try:
                d_model = DynSCFGaussian.load_from_ckpt(
                    torch.load(d_pth, map_location="cpu", weights_only=False),
                    device=device,
                ).to(device)
                d_model.eval()
                cams = MonocularCameras.load_from_ckpt(
                    torch.load(c_pth, map_location="cpu", weights_only=False)
                ).to(device)
                self.models.append((label, d_model, cams))
                print(f"[ok]   {label}: T={d_model.T} M={d_model.scf.M}")
            except Exception as e:
                print(f"[fail] {label}: {e}")

        if not self.models:
            raise RuntimeError("No variant checkpoints loaded.")

        self.num_frames = min(int(m[1].T) for m in self.models)
        ref_cams = self.models[0][2]
        # pivot from first model mid-frame
        t0 = self.num_frames // 2
        center = self.models[0][1](t0)[0][::10].mean(dim=0).detach().cpu().numpy()

        server = get_server(port=port)
        self.viewer = DynamicViewer(
            server,
            self.render_fn,
            self.num_frames,
            spin_root,
            mode="rendering",
            look_at_tuple=(float(center[0]), float(center[1]), float(center[2])),
            cams=ref_cams,
        )
        print(f"4-way viewer on port {port}: {[m[0] for m in self.models]}")

    def _label(self, img, text):
        if not _HAS_CV2:
            return img
        img = np.ascontiguousarray(img)
        cv2.rectangle(img, (0, 0), (len(text) * 11 + 12, 26), (0, 0, 0), -1)
        cv2.putText(
            img, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
        )
        return img

    @torch.inference_mode()
    def _render_one(self, d_model, ph, pw, K, w2c, t):
        gs5 = [list(d_model(t))]
        out = render(gs5, ph, pw, K=K, T_cw=w2c, bg_color=[1.0, 1.0, 1.0])
        img = torch.clamp(out["rgb"].permute(1, 2, 0), 0.0, 1.0)
        return (img.cpu().numpy() * 255.0).astype(np.uint8)

    @torch.inference_mode()
    def render_fn(self, camera_state, img_wh):
        W, H = img_wh
        # 2x2 grid -> each panel is half size
        pw, ph = W // 2, H // 2
        focal = 0.5 * ph / np.tan(0.5 * camera_state.fov).item()
        K = torch.tensor(
            [[focal, 0.0, pw / 2.0], [0.0, focal, ph / 2.0], [0.0, 0.0, 1.0]],
            device=self.device,
        )
        w2c = torch.linalg.inv(
            torch.from_numpy(camera_state.c2w.astype(np.float32)).to(self.device)
        )
        t = (
            int(self.viewer._playback_guis[0].value)
            if not getattr(self.viewer, "_canonical_checkbox", None)
            or not self.viewer._canonical_checkbox.value
            else None
        )

        panels = []
        for label, d_model, _cams in self.models:
            tt = None if t is None else min(int(t), int(d_model.T) - 1)
            try:
                p = self._render_one(d_model, ph, pw, K, w2c, tt)
            except Exception as e:
                p = np.full((ph, pw, 3), 200, np.uint8)
                print(f"[render fail] {label}: {e}")
            panels.append(self._label(p, label))

        while len(panels) < 4:  # pad to 2x2
            panels.append(np.full((ph, pw, 3), 255, np.uint8))

        top = np.concatenate([panels[0], panels[1]], axis=1)
        bot = np.concatenate([panels[2], panels[3]], axis=1)
        grid = np.concatenate([top, bot], axis=0)
        # ensure exact img_wh
        if grid.shape[0] != H or grid.shape[1] != W:
            grid = grid[:H, :W]
        return grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg_fn", type=str, default="./profile/iphone/iphone_fit.yaml")
    ap.add_argument("--spin_root", type=str, required=True)
    ap.add_argument("--port", type=int, default=8899)
    # optional per-variant overrides
    for nm in ["baseline", "observability", "render_contrib", "usplat4d"]:
        ap.add_argument(f"--{nm}_pth", type=str, default=None)
        ap.add_argument(f"--{nm}_cam", type=str, default=None)
    args = ap.parse_args()

    overrides = {}
    for nm in ["baseline", "observability", "render_contrib", "usplat4d"]:
        if getattr(args, f"{nm}_pth"):
            overrides[f"{nm}_pth"] = getattr(args, f"{nm}_pth")
        if getattr(args, f"{nm}_cam"):
            overrides[f"{nm}_cam"] = getattr(args, f"{nm}_cam")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Compare4WayViewer(args.cfg_fn, args.spin_root, device, port=args.port, overrides=overrides)
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
