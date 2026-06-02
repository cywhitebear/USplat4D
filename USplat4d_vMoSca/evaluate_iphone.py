import os, os.path as osp
import pandas as pd
from tqdm import tqdm
import datetime
import shutil

def evaluate_iphone():
    scene_name_list = [
        "apple",
        "block",
        "paper-windmill",
        # "space-out",
        "spin",
        "teddy",
        # "wheel",
    ]

    root = "./data/iphone/"
    backup_root = "./data/metrics_collected"
    backup_flag = True
    prefix = "tto_"


    for name in ["iphone_fit_native", "iphone_fit_colfree_native", "iphone_fit_focalonly_native"]:
        print("_".join(root.split("/")))
        backup_dir = osp.join(
            backup_root,
            prefix
            + name
            + "_".join(root.split("/"))
            + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        )
        os.makedirs(backup_dir, exist_ok=True)

        results = {}
        pck_results = {}
        for scene_name in tqdm(scene_name_list):
            # find the latest log
            log_dirs = os.listdir(osp.join(root, scene_name, "logs"))
            log_dirs = [d for d in log_dirs if name in d]

            log_dirs.sort()
            print(log_dirs)
            log_dir = log_dirs[-1]
            fn = osp.join(root, scene_name, "logs", log_dir, f"{prefix}dycheck_metrics.xlsx")
            if not osp.exists(fn):
                print("file not found", fn)
                psnr = 0
                ssim = 0
                lpips = 1000000
            else:
                # read the first line of xls
                df = pd.read_excel(fn).to_numpy()

                psnr = float(df[0][-3])
                ssim = float(df[0][-2])
                lpips = float(df[0][-1])
                print(df[0])
            print(scene_name)
            print(fn)
            print(psnr, ssim, lpips)
            results[f"{scene_name}-psnr"] = psnr
            results[f"{scene_name}-ssim"] = ssim
            results[f"{scene_name}-lpips"] = lpips

            pck_fn = osp.join(root, scene_name, "logs", log_dir, f"pck5.txt")
            with open(pck_fn, "r") as f:
                pck = float(f.read().split(":")[-1].strip("\n"))
            pck_results[f"{scene_name}-pck"] = pck

            if backup_flag:
                # copy the log_dir into the backup
                shutil.copytree(
                    osp.join(root, scene_name, "logs", log_dir, f"{prefix}test"),
                    osp.join(
                        backup_dir,
                        prefix + scene_name + f"{log_dir}_test" + "_".join(root.split("/")),
                    ),
                )
        df = pd.DataFrame([results], index=[0])
        df.to_excel(
            osp.join(backup_dir, f"{prefix}{name}_iphone_collected_masked_metrics.xlsx")
        )

        mean_psnr = sum([results[f"{scene}-psnr"] for scene in scene_name_list]) / len(scene_name_list)
        mean_ssim = sum([results[f"{scene}-ssim"] for scene in scene_name_list]) / len(scene_name_list)
        mean_lpips = sum([results[f"{scene}-lpips"] for scene in scene_name_list]) / len(scene_name_list)

        print("mean psnr", mean_psnr)
        print("mean ssim", mean_ssim)
        print("mean lpips", mean_lpips)

        df = pd.DataFrame([pck_results], index=[0])
        df.to_excel(osp.join(backup_dir, f"{prefix}{name}_iphone_collected_pck.xlsx"))
        mean_pck = sum(pck_results.values()) / len(pck_results)
        print("mean pck", mean_pck)

if __name__ == "__main__":
    evaluate_iphone()