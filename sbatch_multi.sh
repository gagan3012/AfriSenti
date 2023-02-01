models=(afro-xlmr-base-lm)
subset=(yo twi ts sw pt pcm ma kr ig ha dz am multilingual or tg)
for model in "${models[@]}"
do
    for sub in "${subset[@]}"
    do
        run_name="pred$model$sub"
        echo $run_name
        sbatch --job-name=$run_name run_predict_test.sh $model $sub
    done
done
# for model in "${t5_models[@]}"
# do
#     for sub in "${subset[@]}"
#     do
#         run_name="$model$sub"
#         echo $run_name
#         sbatch --job-name=$run_name run_t5.sh $model $sub 
#     done
# done