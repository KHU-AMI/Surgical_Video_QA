#!/bin/bash
#SBATCH -J STAGE2_1019K_1294K
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
export WANDB_DISABLED=true 
export TORCHDYNAMO_DISABLE=1
###########################
#4BIT 원본 파일
###########################
MODEL_VERSION=vicuna-v1-5-7b
gpu_vis="0,1,2,3,4,5,6"
MASTER_PORT=29570
##/data/kayoung/repos/projects/SurgVU/VTimeLLM/FINAL_converted.json원본 데이터셋
deepspeed --include localhost:$gpu_vis --master_port $MASTER_PORT vtimellm/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --lora_enable True \
    --model_name_or_path ./checkpoints/vicuna-7b-v1.5 \
    --version v1 \
    --data_path /data/kayoung/repos/surgvu/FINAL/KAYOUNG_2/PART_5_REFINED/KAYOUNG_FINAL_TRAIN_259K_PART5.json \
    --feat_folder /data2/local_datasets/YT_data/surgvu24_videos_only_features_endovit_only_cls/surgvu24 \
    --pretrain_mm_mlp_adapter /data/kayoung/repos/projects/SurgVU/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage1_KAYOUNG_280K/checkpoint-10230/mm_projector.bin \
    --output_dir /data/kayoung/repos/projects/SurgVU/VTimeLLM/checkpoints/KAYOUNG_FINAL_TRAIN_259K_PART1_REFINED \
    --bf16 True \
    --bits 4 \
    --quant_type nf4 \
    --double_quant True \
    --num_train_epochs 10 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --save_strategy "epoch" \
    --save_total_limit 30 \
    --learning_rate 1e-4 \
    --freeze_mm_mlp_adapter True \
    --lora_r 64 \
    --lora_alpha 128 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 50 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to none
