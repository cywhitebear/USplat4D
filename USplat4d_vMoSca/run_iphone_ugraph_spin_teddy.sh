#!/bin/bash
# Re-run run_ugraph for spin and teddy only (wheel already running separately)
# Fix: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Requires Sunshine stopped first
gpu_id=0
PYTHON=/home/ee904/miniconda3/envs/usplat4d/bin/python

dataset_name=iphone
seq_names=(spin teddy)

depth_ratio=0.1
threshold_min_set=0.5
version_key_edges=minmax_distance_contrib
log_subfolder=iphone_fit_native_add3
extra_save_str=

ini_folder_path=/media/ee904/DATA1/Yun/Datasets/MoSca/iphone
usplat4d_folder_path=/media/ee904/DATA1/Yun/Datasets/usplat4d_iphone_graph

cd /home/ee904/Yun/usplat4d/USplat4d_vMoSca

for seq_name in "${seq_names[@]}"; do
    echo "======== run_ugraph: ${seq_name} ========"
    dir_name_saved_propressinging_model=dr${depth_ratio}_thr${threshold_min_set}_v${version_key_edges}${extra_save_str}
    dir_name_saving_this_run_model=saved_ugraph_model${extra_save_str}

    CUDA_VISIBLE_DEVICES=$gpu_id \
    $PYTHON -u mosca_reconstruct.py \
        --cfg ./profile/iphone/iphone_fit.yaml \
        --ws ${ini_folder_path}/${seq_name} \
        --run_ugraph \
        --graph_dir ${usplat4d_folder_path}/${seq_name}/${dir_name_saved_propressinging_model}/ \
        --log_dir ${ini_folder_path}/${seq_name}/logs/${log_subfolder} \
        --dir_name_saving_this_run_model $dir_name_saving_this_run_model \
        --disable_cam_pose_training
    echo "======== run_ugraph DONE: ${seq_name} ========"
done

echo "ALL DONE (spin+teddy)"
