# mosca or ugraph can be evaluated by the same script
# mosca_evauate.py should use main_ugraph()

# seq_names=('paper-windmill') # evaluate one instance
seq_names=('apple' 'spin' 'teddy' 'block' 'wheel' 'space-out' 'paper-windmill') # evaluate all instances
# Loop and print each name
for seq_name in "${seq_names[@]}"
do
    echo "${seq_name}"
done

# Loop over each sequence
for seq_name in "${seq_names[@]}"
do
    echo "Starting training for ${seq_name}"
    
    CUDA_VISIBLE_DEVICES=0 python mosca_evaluate.py  \
        --cfg ./profile/iphone/iphone_fit.yaml    \
        --ws /data/dataset/iphone_mosca_trained/$seq_name/   \
        --pth_save_dir /data/dataset/iphone_mosca_graph_model/$seq_name/dr0.1_thr0.5_vminmax_distance_contrib/saved_ugraph_model/step_1599

    echo "Finished training for ${seq_name}"
    echo "*****************************************"
    echo "*****************************************"
done