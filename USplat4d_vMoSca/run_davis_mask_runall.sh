#!/bin/bash
gpu_id=0

dataset_name=davis_mask
        
seq_names=(train)
# seq_names=(train car-roundabout breakdance-flare helicopter schoolgirls crossing)

threshold_min_set=0.5 # 3.0 # 0.1 # 0.1 paper-windmill # 0.5 others
version_key_edges=max_contrib # choices: max_contrib minmax_distance_contrib max_native
depth_ratio=0.001 # 0.001 backpack_ # 0.1 default
# extra_save_str=test
log_subfolder=demo_fit_native_add3

ini_folder_path=/data/dataset/davis480_mosca_trained_masked
usplat4d_folder_path=/data/dataset/davis480_mosca_graph_model_masked

# Function to get current timestamp
get_timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

# Function to calculate elapsed time
calculate_elapsed_time() {
    local start_time=$1
    local end_time=$2
    local elapsed=$((end_time - start_time))
    local hours=$((elapsed / 3600))
    local minutes=$(((elapsed % 3600) / 60))
    local seconds=$((elapsed % 60))
    printf "%02d:%02d:%02d" $hours $minutes $seconds
}

# Function to write config header to log file
write_config_header() {
    local log_file=$1
    local script_name=$2
    echo "==========================================" > "$log_file"
    echo "Configuration Parameters" >> "$log_file"
    echo "==========================================" >> "$log_file"
    echo "Dataset: $dataset_name" >> "$log_file"
    echo "Threshold Min Set: $threshold_min_set" >> "$log_file"
    echo "Version Key Edges: $version_key_edges" >> "$log_file"
    echo "Depth Ratio: $depth_ratio" >> "$log_file"
    echo "Log Subfolder: $log_subfolder" >> "$log_file"
    echo "Script: $script_name" >> "$log_file"
    echo "Start Time: $(get_timestamp)" >> "$log_file"
    echo "==========================================" >> "$log_file"
    echo "" >> "$log_file"
}

cd lib_usplat4d_prep

# Loop and print each name
for seq_name in "${seq_names[@]}"
do
    echo "${seq_name}"
done

# Create log directories for this sequence
base_log_dir="/data/dataset/davis480_mosca_graph_model_masked"
mkdir -p "${base_log_dir}/.log_main1"
mkdir -p "${base_log_dir}/.log_main23"

# Loop over each sequence
for seq_name in "${seq_names[@]}"
do
    echo "Starting training for ${seq_name}"
    echo "Start time: $(get_timestamp)"
    
    # First Python script - main1
    echo "Running usplat4d_prep.py main1 for ${seq_name}"
    log_file1="${base_log_dir}/.log_main1/${seq_name}.log"
    write_config_header "$log_file1" "usplat4d_prep.py main1"
    start_time1=$(date +%s)
    CUDA_VISIBLE_DEVICES=$gpu_id python usplat4d_prep.py \
        --seq_name $seq_name \
        --func_name main1 \
        --dataset_name $dataset_name \
        --depth_ratio $depth_ratio \
        --threshold_min_set $threshold_min_set \
        --version_key_edges $version_key_edges \
        --log_subfolder $log_subfolder \
        --ini_folder_path $ini_folder_path \
        --usplat4d_folder_path $usplat4d_folder_path \
        2>&1 | tee -a "$log_file1"
    end_time1=$(date +%s)
    elapsed1=$(calculate_elapsed_time $start_time1 $end_time1)
    echo "main1 completed for ${seq_name} in ${elapsed1}"
    echo "main1 runtime: ${elapsed1}" >> "$log_file1"
    
    # Second Python script - main23
    echo "Running usplat4d_prep.py main23 for ${seq_name}"
    log_file2="${base_log_dir}/.log_main23/${seq_name}.log"
    write_config_header "$log_file2" "usplat4d_prep.py main23"
    start_time2=$(date +%s)
    CUDA_VISIBLE_DEVICES=$gpu_id python usplat4d_prep.py \
        --seq_name $seq_name \
        --func_name main23 \
        --dataset_name $dataset_name \
        --depth_ratio $depth_ratio \
        --threshold_min_set $threshold_min_set \
        --version_key_edges $version_key_edges \
        --log_subfolder $log_subfolder \
        --ini_folder_path $ini_folder_path \
        --usplat4d_folder_path $usplat4d_folder_path \
        2>&1 | tee -a "$log_file2"
    end_time2=$(date +%s)
    elapsed2=$(calculate_elapsed_time $start_time2 $end_time2)
    echo "main23 completed for ${seq_name} in ${elapsed2}"
    echo "main23 runtime: ${elapsed2}" >> "$log_file2"

    echo "Finished preprocessing for ${seq_name}"
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

# Create log directory for reconstruction
base_log_dir="/data/dataset/davis480_mosca_graph_model_masked"
mkdir -p "${base_log_dir}/.log_recon"
# Loop over each sequence
for seq_name in "${seq_names[@]}"
do
    echo "Starting reconstruction for ${seq_name}"
    echo "Start time: $(get_timestamp)"
    
    dir_name_saved_propressinging_model=dr${depth_ratio}_thr${threshold_min_set}_v${version_key_edges}${extra_save_str} # _ab_key_selection _ab_key_selection2

    # Third Python script - mosca_reconstruct.py
    echo "Running mosca_reconstruct.py for ${seq_name}"
    log_file3="${base_log_dir}/.log_recon/${seq_name}.log"
    write_config_header "$log_file3" "mosca_reconstruct.py"
    start_time3=$(date +%s)
    CUDA_VISIBLE_DEVICES=$gpu_id python mosca_reconstruct.py     \
        --cfg ./profile/demo/demo_fit.yaml     \
        --ws /data/dataset/davis480_mosca_trained_masked/${seq_name}     \
        --run_ugraph \
        --graph_dir /data/dataset/davis480_mosca_graph_model_masked/${seq_name}/${dir_name_saved_propressinging_model}/     \
        --log_dir /data/dataset/davis480_mosca_trained_masked/${seq_name}/logs/${log_subfolder} \
        --dir_name_saving_this_run_model $dir_name_saving_this_run_model \
        --use_gt_masks \
        --gt_mask_path mask \
        2>&1 | tee -a "$log_file3"
    end_time3=$(date +%s)
    elapsed3=$(calculate_elapsed_time $start_time3 $end_time3)
    echo "mosca_reconstruct.py completed for ${seq_name} in ${elapsed3}"
    echo "mosca_reconstruct.py runtime: ${elapsed3}" >> "$log_file3"

    echo "Finished reconstruction for ${seq_name}"
    echo "*****************************************"
    echo "*****************************************"
done

# Print final summary
echo "=========================================="
echo "ALL SEQUENCES COMPLETED"
echo "Final completion time: $(get_timestamp)"
echo "=========================================="