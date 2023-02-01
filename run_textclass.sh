#!/bin/bash
#SBATCH --time=1:59:59
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --account=rrg-mageed
#SBATCH --job-name=train_image_v2
#SBATCH --output=/lustre07/scratch/gagan30/arocr/logs/%x.out
#SBATCH --error=/lustre07/scratch/gagan30/arocr/logs/%x.err
#SBATCH --mail-user=gbhatia880@gmail.com
#SBATCH --mail-type=ALL

module load python/3.8 scipy-stack gcc arrow cuda cudnn httpproxy 

source ~/ENV38_default/bin/activate

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

model_name=$1
data_subset=$2 
if [ $data_subset == "multilingual" ]; then
  data_dir="SubtaskB"
else
  data_dir="SubtaskA"
fi

CUDA_VISIBLE_DEVICES=0 python run_textclass.py \
  --model_name_or_path ../../../models/$model_name \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size 16 \
  --num_train_epochs 15 \
  --max_seq_length 128 \
  --data_dir ../$data_dir/ \
  --data_subset $data_subset \
  --output_dir ../../../results/afrisenti/$model_name/$data_subset \
  --load_best_model_at_end True \
  --metric_for_best_model f1 \
  --greater_is_better True \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --logging_steps 1 \
  --save_total_limit 1 \
  --seed 42 \
  --overwrite_output_dir \
  --gradient_accumulation_steps 16 \
  --gradient_checkpointing True \
  --fp16 \
  # --hyperparameter_search True \

python run_predict.py \
  --model_path ../../../results/afrisenti/$model_name/$data_subset \
  --file_name ../$data_dir/dev/${data_subset}_dev.tsv \
  --gold_file ../$data_dir/dev_gold/${data_subset}_dev_gold_label.tsv \
  --lang_code $data_subset \

python run_predict.py \
  --file_name ../$data_dir/test/${data_subset}_test_participants.tsv \
  --lang_code $data_subset \
  --model_path ../../../results/afrisenti/$model_name/$data_subsets \
  --model_name $model_name \
