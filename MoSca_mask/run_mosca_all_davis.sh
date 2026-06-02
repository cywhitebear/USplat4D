#!/usr/bin/env bash
set -euo pipefail

GPU_ID=0

root_dir=/data/dataset/davis480_mosca_trained_masked

log_pre="${root_dir}/.log_pre"
log_recon="${root_dir}/.log_recon"
mkdir -p "$log_pre" "$log_recon"

# Explicit DAVIS sequence list
seq_names=(train)
# seq_names=(train car-roundabout breakdance-flare helicopter schoolgirls crossing)

# Build absolute paths from sequence names
base_dirs=()
for name in "${seq_names[@]}"; do
  base_dirs+=("${root_dir}/${name}")
done

total=${#base_dirs[@]}
# for name in "${seq_names[@]}"; do
#   echo "name=$name"
# done

for dir in "${base_dirs[@]}"; do
  echo "dir=$dir"
done

i=0

for dir in "${base_dirs[@]}"; do
  ws="${dir%/}"
  i=$((i+1))
  echo "[$i/$total] ws=$ws"

  pre_log="${log_pre}/$(echo "$ws" | sed 's#^/##; s#/#_#g').log"
  recon_log="${log_recon}/$(echo "$ws" | sed 's#^/##; s#/#_#g').log"
  
  echo "precompute: $pre_log"
  start_pre=$(date +%s)
  CUDA_VISIBLE_DEVICES=$GPU_ID python mosca_precompute.py --cfg ./profile/nvidia/nvidia_prep.yaml --ws "$ws" >"$pre_log" 2>&1
  end_pre=$(date +%s)
  dur_pre=$((end_pre - start_pre))
  echo "precompute: ${dur_pre}s" | tee -a "$pre_log"

  echo "reconstruct: $recon_log"
  start_rec=$(date +%s)
  CUDA_VISIBLE_DEVICES=$GPU_ID python mosca_reconstruct.py --cfg ./profile/demo/demo_fit.yaml --ws "$ws" --use_gt_masks --gt_mask_path mask >"$recon_log" 2>&1
  end_rec=$(date +%s)
  dur_rec=$((end_rec - start_rec))
  echo "reconstruct: ${dur_rec}s" | tee -a "$recon_log"
done