# USplat4D: Uncertainty Matters in Dynamic Gaussian Splatting for Monocular 4D Reconstruction

[![Project Page](https://img.shields.io/badge/Project%20Page-Visit-blue)](https://tamu-visual-ai.github.io/usplat4d/)
[![Paper](https://img.shields.io/badge/Latest%20Paper-Read-orange)](https://arxiv.org/abs/2510.12768)
[![Contact Author](https://img.shields.io/badge/Contact%20Author-Email-green)](mailto:fengzh_g@tamu.edu)


Code release of **USplat4D** - **U**ncertainty Matters in Dynamic Gaussian **Splat**ting for Monocular **4D** Reconstruction. We support the initialization from 4DGS including [MoSca](https://www.cis.upenn.edu/~leijh/projects/mosca/).

## Updates
- 06/02/2026 Code Release of USplat4D supporting MoSca's 4DGS output. We also released the unofficial implementation of FG/BG mask support for MoSca.

## Contents

| Path | What it is |
|---|---|
| [`USplat4d_vMoSca/`](./USplat4d_vMoSca) | Main USplat4D pipeline (preprocessing + reconstruction on MoSca output) |
| [`MoSca_mask/`](./MoSca_mask) | MoSca fork with FG-BG mask support (unofficial). Shares the same conda env. |
| [`install_usplat4d_env.sh`](./install_usplat4d_env.sh) | One-shot environment installer |
| [`requirements.txt`](./requirements.txt) | Pip dependencies (consumed by the installer) |
| [`info.txt`](./info.txt) | Step-by-step install notes (verbose; useful for debugging) |

## Install

Requires: Linux, conda, NVIDIA driver with CUDA ≥ 12.1, gcc available on PATH. Verified on H100 (sm_90).

```bash
git clone <this-repo> usplat4d_release && cd usplat4d_release
bash install_usplat4d_env.sh
conda activate usplat4d
```

The script handles everything: creates a Python 3.10 env, installs PyTorch 2.1.2 + CUDA 12.1, pins `mkl=2023.1.0` (to avoid the `iJIT_NotifyEvent` ABI bug), installs `xformers` / `pytorch3d` / `fvcore` / PyG wheels, the four local CUDA extensions in `MoSca_mask/lib_render/`, plus optional `jax` (for DyCheck eval) and `nvdiffrast`.

GPU architecture is auto-detected via `nvidia-smi`. Override defaults as env vars:

```bash
# Optionally, overwrite default values
# ENV_NAME=usplat4d 
# TORCH_CUDA_ARCH_LIST=9.0       # e.g. for H100; leave unset to auto-detect
# INSTALL_NVDIFFRAST=0            # skip nvdiffrast (default: 1)
# INSTALL_GSPLAT=1 GSPLAT_DIR=/path/to/gsplat    # editable install of gsplat fork

# install conda env
bash install_usplat4d_env.sh

# install gsplat with uncertainty estimation if you have not installed it.
# USplat4D relies on uncertainty estimation, so make sure you install this gsplat version for uncertainty estimation.
cd external/gsplat
git checkout contribs1.5.3
pip install -e . --no-build-isolation
```

See [`info.txt`](./info.txt) for the full reasoning behind each step and known gotchas (CUDA toolkit channel pinning, dangling `.so` symlinks, gcc version constraint, PEP 517 build isolation).

Follow [`./MoSca_mask/readme.md`](./MoSca_mask/readme.md) to download the weights for 2D foundation models.

## Download Dataset
Download dycheck dataset (or iphone dataset) [here]() and Davis dataset [here](). And make sure you have this folder structure:
```bash
<Dataset_folder>
├── <instance_1_folder>
│   ├── images
│   └── mask
├── <instance_2_folder>
│   ├── images
│   └── mask
└── ...
```
We also provide the initialization version [here](https://huggingface.co/datasets/percool777/usplat4d), in which already finishes the initialization by MoSca in Quick Start below and you can directly run Step 2.


## Quick Start

After `conda activate usplat4d`:

```bash
# 1) Step 1: Run initialization (by MoSca).
cd MoSca_mask
bash run_mosca_all_davis.sh # Run Davis instances (Make sure you use the correct folder path in .sh)
# bash run_mosca_all_iphone.sh # Run iphone instances (Make sure you use the correct folder path in .sh)

# 2) Step 2: Run USplat4D.
cd ../USplat4d_vMoSca
bash run_davis_mask_runall.sh # Run Davis instances (Make sure you use the correct folder path in .sh)
# bash run_iphone_runall.sh # Run iphone instances (Make sure you use the correct folder path in .sh)
```

## Interactive Viewer
Run USplat4D by:
```bash
cd USplat4d_vMoSca
ini_folder_path=/data/dataset/davis480_mosca_trained_masked
usplat4d_folder_path=/data/dataset/davis480_mosca_graph_model_masked
seq_name=train
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python lib_usplat4d_viewer/usplat4d_run_rendering.py \
    --cfg_fn profile/demo/demo_fit.yaml \
    --work_dir ${ini_folder_path}/${seq_name}/logs/demo_fit_native_add3 \
    --use_ugraph \
    --pth_save_dir ${usplat4d_folder_path}/${seq_name}/dr0.001_thr0.5_vmax_contrib/saved_ugraph_model/step_1599
```

Run MoSca for comparison by:
```bash
cd USplat4d_vMoSca
seq_name=train
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python lib_usplat4d_viewer/usplat4d_run_rendering.py \
    --cfg_fn profile/demo/demo_fit.yaml \
    --work_dir /data/dataset/davis480_mosca_trained_masked/${seq_name}/logs/demo_fit_native_add3
```

## Evaluation
To evaluate iphone dataset of the USplat4D model, run:
```bash
cd USplat4d_vMoSca
bash run_iphone_evaluate.sh
```
## Acknowledgements
We thank the authors of [Shape of Motion](https://shape-of-motion.github.io) and [MoSca](https://jiahuilei.com/projects/mosca/) for their great works and sharing the code and results. We also thank the authors of [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [gsplat](https://github.com/nerfstudio-project/gsplat) for their great contributions on the Gaussian Splatting implementation.

## License

MIT for the original code written by the authors of USplat4D (mostly under `lib_ugraph` and `lib_usplat4d_prep`) and MoSca (mostly under `lib_moca/`, `lib_mosca/`). Third-party components under `lib_prior/` and `lib_render/` (foundation models, GS rasterizers) retain their respective upstream licenses — see [`MoSca_mask/readme.md`](./MoSca_mask/readme.md) for details.

## Citation

If you use this code, please cite USplat4D and MoSca:

```bibtex
@inproceedings{guo2026uncertainty,
  title={Uncertainty Matters in Dynamic Gaussian Splatting for Monocular 4D Reconstruction},
  author={Fengzhi Guo and Chih-Chuan Hsu and Sihao Ding and Cheng Zhang},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=m3rZ7Fdlst}
}
```
```bibtex
@article{lei2024mosca,
  title  = {MoSca: Dynamic Gaussian Fusion from Casual Videos via 4D Motion Scaffolds},
  author = {Lei, Jiahui and Weng, Yijia and Harley, Adam and Guibas, Leonidas and Daniilidis, Kostas},
  journal= {arXiv preprint arXiv:2405.17421},
  year   = {2024}
}
```
