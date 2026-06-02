import argparse
import os
import sys
import torch
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib_usplat4d_viewer.usplat4d_renderer import Renderer

def run_rendering():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_fn", type=str)
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument("--use_ugraph", action="store_true")
    parser.add_argument("--pth_save_dir", type=str, required=False) # if not provided, will use the same as work_dir
    parser.add_argument("--images_dir", type=str, required=False) # if provided, show images on camera frustums
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    renderer = Renderer(args.cfg_fn, device, args.work_dir, port=args.port, bg_flag=False, fg_flag=True, pth_save_dir=args.pth_save_dir, use_ugraph=args.use_ugraph, images_dir=args.images_dir)
    while True:
        time.sleep(1.0)

if __name__ == "__main__":
    run_rendering()