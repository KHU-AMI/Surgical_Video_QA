#!/bin/bash
#SBATCH -J STAGE1_1019K
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=50G
#SBATCH -p batch
#SBATCH -w augi5
#SBATCH -t 336:00:00
#SBATCH -o logs/%j.out

source /data/kayoung/init.sh
conda activate vtimellm

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

MODEL_VERSION=vicuna-v1-5-7b
gpu_vis="0,1,2,3,4,5,6" # per_device_train_batch_size * gradient_accumulation_steps * n_gpus = 128
MASTER_PORT=29570


deepspeed --include localhost:$gpu_vis --master_port $MASTER_PORT vtimellm/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./checkpoints/vicuna-7b-v1.5 \
    --version plain \
    --data_path /data/kayoung/repos/surgvu/dataset_final.json \
    --feat_folder /data2/local_datasets/kayoung_data/surgvu_stage1_feat_final \
    --tune_mm_mlp_adapter True \
    --output_dir ./checkpoints/vtimellm-$MODEL_VERSION-stage1_KAYOUNG_464K \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \