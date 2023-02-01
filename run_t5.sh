#!/bin/bash
#SBATCH --time=2:59:59
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
# export TRANSFORMERS_OFFLINE=1

model_name=$1
data_subset=$2
if [ $data_subset == "multilingual" ]; then
  data_dir="SubtaskB"
else
  data_dir="SubtaskA"
fi

CUDA_VISIBLE_DEVICES=0 python run_summarization.py \
  --model_name_or_path ../../../models/$model_name \
  --do_train \
  --do_eval \
  --do_predict \
  --per_device_train_batch_size 8 \
  --num_train_epochs 50 \
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
  --fp16 \
  --text_column tweet \
  --summary_column label \
  --source_prefix "afrisenti: " \
  --predict_with_generate \
