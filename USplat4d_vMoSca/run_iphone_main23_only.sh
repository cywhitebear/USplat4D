#!/bin/bash
# Run main23 + mosca_reconstruct for spin/teddy/wheel
# Skips main1 (pkl files already exist from previous run)
gpu_id=0
PYTHON=/home/ee904/miniconda3/envs/usplat4d/bin/python

dataset_name=iphone
seq_names=(spin teddy wheel)

depth_ratio=0.1
threshold_min_set=0.5
version_key_edges=minmax_distance_contrib
log_subfolder=iphone_fit_native_add3
extra_save_str=

ini_folder_path=/media/ee904/DATA1/Yun/Datasets/MoSca/iphone
usplat4d_folder_path=/media/ee904/DATA1/Yun/Datasets/usplat4d_iphone_graph

cd /home/ee904/Yun/usplat4d/USplat4d_vMoSca/lib_usplat4d_prep

for seq_name in "${seq_names[@]}"; do
    echo "======== main23: ${seq_name} ========"
    CUDA_VISIBLE_DEVICES=$gpu_id $PYTHON -u usplat4d_prep.py \
        --seq_name $seq_name \
        --func_name main23 \
        --dataset_name $dataset_name \
        --depth_ratio $depth_ratio \
        --threshold_min_set $threshold_min_set \
        --version_key_edges $version_key_edges \
        --log_subfolder $log_subfolder \
        --ini_folder_path $ini_folder_path \
        --usplat4d_folder_path $usplat4d_folder_path
    echo "======== main23 DONE: ${seq_name} ========"
done

cd /home/ee904/Yun/usplat4d/USplat4d_vMoSca

for seq_name in "${seq_names[@]}"; do
    echo "======== run_ugraph: ${seq_name} ========"
    dir_name_saved_propressinging_model=dr${depth_ratio}_thr${threshold_min_set}_v${version_key_edges}${extra_save_str}
    dir_name_saving_this_run_model=saved_ugraph_model${extra_save_str}

    CUDA_VISIBLE_DEVICES=$gpu_id $PYTHON -u mosca_reconstruct.py \
        --cfg ./profile/iphone/iphone_fit.yaml \
        --ws /media/ee904/DATA1/Yun/Datasets/MoSca/iphone/${seq_name} \
        --run_ugraph \
        --graph_dir /media/ee904/DATA1/Yun/Datasets/usplat4d_iphone_graph/${seq_name}/${dir_name_saved_propressinging_model}/ \
        --log_dir /media/ee904/DATA1/Yun/Datasets/MoSca/iphone/${seq_name}/logs/${log_subfolder} \
        --dir_name_saving_this_run_model $dir_name_saving_this_run_model \
        --disable_cam_pose_training
    echo "======== run_ugraph DONE: ${seq_name} ========"
done

echo "ALL DONE"
