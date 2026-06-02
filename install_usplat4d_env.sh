#!/usr/bin/env bash
# =============================================================================
# install_usplat4d_env.sh
#
# Reproduces the install procedure documented in ./info.txt.
# Verified on: NVIDIA H100 (sm_90), Ubuntu, conda 24+.
#
# Usage:
#   bash install_usplat4d_env.sh
#
# Environment overrides (set before invoking):
#   ENV_NAME            conda env name                       (default: usplat4d)
#   PY_VERSION          Python version                       (default: 3.10.16)
#   TORCH_CUDA_ARCH_LIST   e.g. "9.0" for H100, "8.6" for 3090, "8.9" for 4090
#                          (default: auto-detected via nvidia-smi)
#   MAX_JOBS            parallel compile jobs                (default: 8)
#   REPO_ROOT           usplat4d_release directory  (default: dir of this script)
#   GSPLAT_DIR          gsplat-with-uncertainty checkout    (default: unset; skipped)
#
#   INSTALL_PYTORCH3D   build pytorch3d from source         (default: 1)
#   INSTALL_JAX         install jax+jaxlib+jax-ecosystem    (default: 1)
#   INSTALL_MMCV        install mmcv-full                   (default: 0)
#   INSTALL_NVDIFFRAST  install nvdiffrast from GitHub      (default: 1)
#   INSTALL_GSPLAT      pip install -e $GSPLAT_DIR          (default: 0)
# =============================================================================
# NOTE: we use `set -eo pipefail` (no `-u`). conda's gxx_linux-64 / cuda
# deactivate scripts reference unset vars like CONDA_BACKUP_CXX without
# guarding, which would otherwise abort the script on every `conda activate`.
set -eo pipefail

# ----------------------- Config (overridable via env) ------------------------
ENV_NAME="${ENV_NAME:-usplat4d}"
PY_VERSION="${PY_VERSION:-3.10.16}"
MAX_JOBS="${MAX_JOBS:-8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
GSPLAT_DIR="${GSPLAT_DIR:-}"

INSTALL_PYTORCH3D="${INSTALL_PYTORCH3D:-1}"
INSTALL_JAX="${INSTALL_JAX:-1}"
# INSTALL_MMCV="${INSTALL_MMCV:-0}"
INSTALL_NVDIFFRAST="${INSTALL_NVDIFFRAST:-1}"
INSTALL_GSPLAT="${INSTALL_GSPLAT:-0}"

# ----------------------- Pretty logging --------------------------------------
log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[err ]\033[0m %s\n' "$*"; exit 1; }

# ----------------------- Pre-flight ------------------------------------------
log "Pre-flight checks"

command -v conda >/dev/null || die "conda not found in PATH"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found (no NVIDIA driver?)"
[ -f "$REPO_ROOT/requirements.txt" ] || die "requirements.txt missing at $REPO_ROOT"
[ -d "$REPO_ROOT/MoSca_mask/lib_render" ] || die "MoSca_mask/lib_render missing at $REPO_ROOT"

# Auto-detect GPU compute capability if not provided.
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
    DETECTED_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ' ')"
    if [ -z "$DETECTED_CAP" ]; then
        warn "Could not auto-detect compute_cap; falling back to TORCH_CUDA_ARCH_LIST=9.0"
        TORCH_CUDA_ARCH_LIST="9.0"
    else
        TORCH_CUDA_ARCH_LIST="$DETECTED_CAP"
        log "Detected GPU compute capability: $TORCH_CUDA_ARCH_LIST"
    fi
fi
export TORCH_CUDA_ARCH_LIST

# Load conda's shell functions so `conda activate` works inside a script.
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# ----------------------- 1) Create env ---------------------------------------
log "[1/11] Create conda env '$ENV_NAME' (Python $PY_VERSION)"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    warn "Env '$ENV_NAME' already exists; reusing it (delete first with 'conda env remove -n $ENV_NAME' if you want a clean slate)"
else
    conda create -n "$ENV_NAME" "python=$PY_VERSION" -y
fi
conda activate "$ENV_NAME"
echo "Active python: $(which python)"
echo "Active pip:    $(which pip)"

# ----------------------- 2) Torch + CUDA 12.1 --------------------------------
log "[2/11] Install PyTorch 2.1.2 + CUDA 12.1"
conda install -y \
    pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 \
    -c pytorch -c nvidia

# ----------------------- 3) mkl ABI fix --------------------------------------
log "[3/11] Pin mkl=2023.1.0 to avoid the iJIT_NotifyEvent ABI break"
conda install -y mkl=2023.1.0

# ----------------------- 4) Other conda deps ---------------------------------
log "[4/11] xformers / fvcore / iopath"
conda install -y -c xformers   xformers=0.0.23.post1
conda install -y -c conda-forge fvcore iopath

# ----------------------- 5) Pip packages -------------------------------------
log "[5/11] Pip packages from requirements.txt"
pip install -r "$REPO_ROOT/requirements.txt"
pip install numpy==1.26.4   # re-pin in case a transitive dep bumped it

log "[5/11] PyG wheels (custom index)"
pip install \
    pyg-lib==0.4.0 \
    torch-scatter==2.1.2 \
    torch-sparse==0.6.18 \
    torch-cluster==1.6.3 \
    torch-spline-conv==1.2.2 \
    -f https://data.pyg.org/whl/torch-2.1.2+cu121.html
pip install torch-geometric==2.6.1
pip install pytorch-lightning==2.1.4

# ----------------------- 6) Build toolchain ----------------------------------
log "[6/11] Pin cuda-nvcc 12.1 via the versioned label channel"
conda install -y -c nvidia/label/cuda-12.1.0 \
    cuda-nvcc cuda-cudart-dev cuda-nvrtc-dev cuda-libraries-dev

log "[6/11] gcc 11 (CUDA 12.1 nvcc rejects gcc>=13)"
conda install -y -c conda-forge gcc_linux-64=11 gxx_linux-64=11 sysroot_linux-64=2.17

pip install ninja

# ----------------------- 6b) Repair dangling cuda symlinks -------------------
log "[6b/11] Repair dangling cuda .so symlinks (if any)"
pushd "$CONDA_PREFIX/lib" >/dev/null
for f in libcudart libcurand libnvrtc libnvJitLink libnvjpeg libcufile libcufile_rdma; do
    if [ -L "${f}.so" ] && [ ! -e "${f}.so" ]; then
        target="$(ls ${f}.so.[0-9]* 2>/dev/null \
                  | grep -E "^${f}\.so\.[0-9]+$" | head -n1)"
        if [ -n "$target" ]; then
            ln -sf "$target" "${f}.so"
            echo "fixed ${f}.so -> $target"
        else
            warn "no .so.<major> for ${f}; left dangling"
        fi
    fi
done
# libnvrtc-builtins uses .so.12.1 as its highest symlink, not .so.12
if [ -L "libnvrtc-builtins.so" ] && [ ! -e "libnvrtc-builtins.so" ]; then
    builtins_target="$(ls libnvrtc-builtins.so.[0-9]* 2>/dev/null \
                       | grep -E "^libnvrtc-builtins\.so\.[0-9]+(\.[0-9]+)?$" | head -n1)"
    [ -n "$builtins_target" ] && ln -sf "$builtins_target" "libnvrtc-builtins.so" \
        && echo "fixed libnvrtc-builtins.so -> $builtins_target"
fi
# Final sanity check
bad=0
for f in "$CONDA_PREFIX"/lib/lib*.so; do
    if [ -L "$f" ] && [ ! -e "$f" ]; then
        warn "STILL BROKEN: $f -> $(readlink "$f")"
        bad=1
    fi
done
[ $bad -eq 0 ] && echo "All cuda .so symlinks resolve."
popd >/dev/null

# ----------------------- 7) Local CUDA extensions ----------------------------
log "[7/11] Build local CUDA extensions in MoSca_mask/lib_render/  (arch=$TORCH_CUDA_ARCH_LIST)"
export CUDA_HOME="$CONDA_PREFIX"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-cc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
export FORCE_CUDA=1
export MAX_JOBS

# pytorch3d from source (needs the build toolchain set up above).
if [ "$INSTALL_PYTORCH3D" = "1" ]; then
    log "[7/11] pytorch3d 0.7.8 from source (no cu121 conda build exists)"
    # Clone full repo (NOT --depth=1 / --filter=blob:none) into a stable path
    # so the install is robust to transient network blips. pip's default
    # `git+...@<tag>` does a partial clone + lazy blob fetch, which can time
    # out mid-checkout. We avoid that footgun by checking out manually.
    # NOTE: pytorch3d tags 0.7.8 as V0.7.8 (uppercase V); every other 0.7.x
    # uses lowercase v. https://github.com/facebookresearch/pytorch3d/tags
    # PT3D_SRC="${PT3D_SRC:-/tmp/pytorch3d_v078_src}"
    # if [ ! -d "$PT3D_SRC/.git" ]; then
    #     rm -rf "$PT3D_SRC"
    #     git clone --branch V0.7.8 --depth 1 \
    #         https://github.com/facebookresearch/pytorch3d.git "$PT3D_SRC"
    # else
    #     log "Reusing existing pytorch3d clone at $PT3D_SRC"
    #     ( cd "$PT3D_SRC" && git fetch --depth 1 origin tag V0.7.8 && git checkout V0.7.8 )
    # fi
    # pip install --no-build-isolation --no-cache-dir "$PT3D_SRC"
    pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8"
else
    warn "Skipping pytorch3d (INSTALL_PYTORCH3D=0)"
fi

pushd "$REPO_ROOT" >/dev/null
pip install --no-build-isolation --no-cache-dir MoSca_mask/lib_render/simple-knn
pip install --no-build-isolation --no-cache-dir MoSca_mask/lib_render/diff-gaussian-rasterization-alphadep-add3
pip install --no-build-isolation --no-cache-dir MoSca_mask/lib_render/diff-gaussian-rasterization-alphadep
pip install --no-build-isolation --no-cache-dir MoSca_mask/lib_render/gof-diff-gaussian-rasterization
popd >/dev/null

# ----------------------- 8) Sanity check -------------------------------------
log "[8/11] Sanity check (import torch first so libc10.so is on the loader path)"
python - <<'PY'
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())
import simple_knn._C, diff_gaussian_rasterization, diff_gaussian_rasterization_add3, gof_diff_gaussian_rasterization
print('all four GS extensions import ok')
if torch.cuda.is_available():
    x = torch.rand(1000, 3, device='cuda')
    from simple_knn._C import distCUDA2
    out = distCUDA2(x)
    print('distCUDA2 cuda test:', tuple(out.shape), 'mean=%.4f' % out.mean().item())
PY

# ----------------------- 9) Optional: JAX ------------------------------------
# jax 0.4.14 is what DyCheck pins. Three gotchas, all baked into the pins below:
#   (a) jax 0.4.14 still uses scipy.linalg.tril at import time -> pin scipy<1.13
#       (deprecated in 1.10, removed in 1.13). We install the cap *first* so
#       that pip downgrades the requirements.txt scipy==1.15.2 cleanly.
#   (b) optax 0.1.4 references jnp.DeviceArray (removed in jax 0.4.x).
#       Bump to optax 0.1.7 which uses jnp.ndarray and still supports jax 0.4.14.
#   (c) The plain jaxlib==0.4.14+cuda12.cudnn89 wheel doesn't bundle cuDNN /
#       cuSPARSE, and conda's CUDA 12.1 ships an older cuSPARSE (12.0.2) that
#       shadows jax's expectations on H100, leaving JAX on CPU.
#       Using the jax[cuda12_pip] extra installs nvidia-cudnn-cu12 +
#       nvidia-cusparse-cu12 + friends as pip wheels, isolated from conda CUDA.
#       JAX then loads its own bundled libs and GPU backend works on H100.
if [ "$INSTALL_JAX" = "1" ]; then
    log "[9/11] JAX 0.4.14 with bundled CUDA (cuda12_pip) for DyCheck evaluation"
    pip install "scipy<1.13"
    pip install "jax[cuda12_pip]==0.4.14" jaxlib==0.4.14+cuda12.cudnn89 \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    pip install \
        chex==0.1.82 dm-haiku==0.0.10 dm-tree==0.1.8 flax==0.7.1 \
        jaxline==0.0.5 jaxtyping==0.3.2 jmp==0.0.4 \
        ml-collections==1.1.0 optax==0.1.7 orbax-checkpoint==0.4.4 \
        tensorstore==0.1.74
else
    warn "Skipping JAX (INSTALL_JAX=0)"
fi

# ----------------------- 10) Optional: mmcv-full / nvdiffrast ---------------
# if [ "$INSTALL_MMCV" = "1" ]; then
#     log "[10/11] mmcv-full 1.7.2"
#     pip install mmcv-full==1.7.2 \
#         -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.2/index.html
# else
#     warn "Skipping mmcv-full (INSTALL_MMCV=0)"
# fi

if [ "$INSTALL_NVDIFFRAST" = "1" ]; then
    # log "[10/11] nvdiffrast v0.3.3 from GitHub (not on PyPI)"
    # NVDR_SRC="${NVDR_SRC:-/tmp/nvdiffrast_v033_src}"
    # if [ ! -d "$NVDR_SRC/.git" ]; then
    #     rm -rf "$NVDR_SRC"
    #     git clone --branch v0.3.3 --depth 1 \
    #         https://github.com/NVlabs/nvdiffrast.git "$NVDR_SRC"
    # fi
    # pip install --no-build-isolation --no-cache-dir "$NVDR_SRC"
    log "[10/11] nvdiffrast v0.3.3"
    pip install --no-build-isolation \
        "git+https://github.com/NVlabs/nvdiffrast.git@v0.3.3"
else
    warn "Skipping nvdiffrast (INSTALL_NVDIFFRAST=0)"
fi

# ----------------------- 11) Optional: gsplat (editable) --------------------
if [ "$INSTALL_GSPLAT" = "1" ]; then
    [ -n "$GSPLAT_DIR" ] || die "INSTALL_GSPLAT=1 but GSPLAT_DIR is not set"
    [ -d "$GSPLAT_DIR" ] || die "GSPLAT_DIR='$GSPLAT_DIR' does not exist"
    log "[11/11] gsplat (editable) from $GSPLAT_DIR"
    pushd "$GSPLAT_DIR" >/dev/null
    pip install -e . --no-build-isolation -v
    popd >/dev/null
else
    warn "Skipping gsplat (INSTALL_GSPLAT=0 or GSPLAT_DIR unset)"
fi

# ----------------------- Done -----------------------------------------------
log "Install complete. Env: $ENV_NAME"
echo "Activate with:  conda activate $ENV_NAME"
