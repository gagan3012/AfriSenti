# models=('xlm-roberta-base' 'xlm-roberta-large' 'bert-base-multilingual-cased' 'afro-xlmr-large' 'afro-xlmr-base' 'afriberta_large' afriberta_base)
t5_models=(afriteva_base afriteva_large)
models=(afro-xlmr-base-lm)
# t5_models=(afriteva_large)
subset=(yo twi ts sw pt pcm ma kr ig ha dz am multilingual)
# subset=(kr ig ha)
for model in "${models[@]}"
do
    for sub in "${subset[@]}"
    do
        run_name="$model$sub"
        echo $run_name
        sbatch --job-name=$run_name run_textclass.sh $model $sub
    done
done
# for model in "${t5_models[@]}"
# do
#     for sub in "${subset[@]}"
#     do
#         run_name="$model$sub"
#         echo $run_name
#         sbatch --job-name=$run_name run_predict_t5.sh $model $sub
#     done
# done

