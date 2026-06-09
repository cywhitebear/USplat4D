#!/bin/bash
gpu_id=0

dataset_name=iphone

seq_names=(paper-windmill) # Try any instance
# seq_names=(
#         'spin'
#         'teddy'
#         'wheel'
#         'apple'
#         'block'
#         'paper-windmill'
#         'space-out'
#     ) # Run all instances

depth_ratio=0.1 # 0.001 backpack_ # 0.1 default
threshold_min_set=0.5 # 0.1 paper-windmill # 0.5 others
version_key_edges=minmax_distance_contrib # choices: max_contrib minmax_distance_contrib max_native

# extra_save_str=test
log_subfolder=iphone_fit_native_add3

ini_folder_path=/media/ee904/DATA1/Yun/Datasets/MoSca/iphone
usplat4d_folder_path=/media/ee904/DATA1/Yun/Datasets/usplat4d_iphone_graph

cd lib_usplat4d_prep

# Loop and print each name
for seq_name in "${seq_names[@]}"
do
    echo "${seq_name}"
done

# Loop over each sequence
for seq_name in "${seq_names[@]}"
do
    echo "Starting training for ${seq_name}"

    CUDA_VISIBLE_DEVICES=$gpu_id python usplat4d_prep.py \
        --seq_name $seq_name \
        --func_name main1 \
        --dataset_name $dataset_name \
        --depth_ratio $depth_ratio \
        --threshold_min_set $threshold_min_set \
        --version_key_edges $version_key_edges \
        --log_subfolder $log_subfolder \
        --ini_folder_path $ini_folder_path \
        --usplat4d_folder_path $usplat4d_folder_path
      #   --extra_save_str $extra_save_str # comment if no extra_save_str
    CUDA_VISIBLE_DEVICES=$gpu_id python usplat4d_prep.py \
        --seq_name $seq_name \
        --func_name main23 \
        --dataset_name $dataset_name \
        --depth_ratio $depth_ratio \
        --threshold_min_set $threshold_min_set \
        --version_key_edges $version_key_edges \
        --log_subfolder $log_subfolder \
        --ini_folder_path $ini_folder_path \
        --usplat4d_folder_path $usplat4d_folder_path

    echo "Finished training for ${seq_name}"
    echo "*****************************************"
    echo "*****************************************"
done

cd ..

# Loop and print each name
for seq_name in "${seq_names[@]}"
do
    echo "${seq_name}"
done

# # choose preprocessing folder
dir_name_saving_this_run_model=saved_ugraph_model${extra_save_str} 

# Loop over each sequence
for seq_name in "${seq_names[@]}"
do
    echo "Starting training for ${seq_name}"

    dir_name_saved_propressinging_model=dr${depth_ratio}_thr${threshold_min_set}_v${version_key_edges}${extra_save_str} # _ab_key_selection _ab_key_selection2

    CUDA_VISIBLE_DEVICES=$gpu_id python mosca_reconstruct.py \
        --cfg ./profile/iphone/iphone_fit.yaml \
        --ws /media/ee904/DATA1/Yun/Datasets/MoSca/iphone/${seq_name} \
        --run_ugraph \
        --graph_dir /media/ee904/DATA1/Yun/Datasets/usplat4d_iphone_graph/${seq_name}/${dir_name_saved_propressinging_model}/ \
        --log_dir /media/ee904/DATA1/Yun/Datasets/MoSca/iphone/${seq_name}/logs/${log_subfolder} \
        --dir_name_saving_this_run_model $dir_name_saving_this_run_model \
        --disable_cam_pose_training

    echo "Finished training for ${seq_name}"
    echo "*****************************************"
    echo "*****************************************"
done

