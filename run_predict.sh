#!/bin/bash
#SBATCH --time=2:59:59
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --account=def-mageed
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
  data_subsets="multilingual"
elif [ $data_subset == "or" ]; then
  data_dir="SubtaskC"
  data_subsets="multilingual"
elif [ $data_subset == "tg" ]; then
  data_dir="SubtaskC"
  data_subsets="multilingual"
else
  data_dir="SubtaskA"
  data_subsets=$2
fi

python run_predict.py \
  --file_name ../$data_dir/dev/${data_subset}_dev.tsv \
  --gold_file ../$data_dir/dev_gold/${data_subset}_dev_gold_label.tsv \
  --lang_code $data_subset \
  --model_path ../../../results/afrisenti/$model_name/$data_subsets \