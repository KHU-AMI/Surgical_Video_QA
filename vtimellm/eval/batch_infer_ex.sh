#!/bin/bash
set -e

VAL_JSON="/data/kayoung/repos/projects/SurgVU/VTimeLLM/FINAL_converted_eval.json"
FEAT_DIR="/data2/local_datasets/YT_data/surgvu24_videos_only_features_endovit_only_cls/surgvu24"

MODEL_BASE="/data/kayoung/repos/projects/SurgVU/VTimeLLM/checkpoints/vicuna-7b-v1.5"
ADAPTER="/data/kayoung/repos/projects/SurgVU/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage1_1/mm_projector.bin"
STAGE2="/data/kayoung/repos/projects/SurgVU/VTimeLLM/checkpoints/0904_checkpoints/checkpoint-382_epoch2"

OUT="pred_test_n30.jsonl"

python -m vtimellm.eval.batch_infer \
  --val_path "$VAL_JSON" \
  --feat_folder "$FEAT_DIR" \
  --model_base "$MODEL_BASE" \
  --pretrain_mm_mlp_adapter "$ADAPTER" \
  --stage2 "$STAGE2" \
  --stage3 '' \
  --out_path "$OUT" \
  --temperature 0.05 \
  --max_new_tokens 512 \
  --max_frames 30 \
  --gpu_mem 16GiB \
  --cpu_mem 64GiB
