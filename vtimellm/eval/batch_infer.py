#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VTimeLLM Stage2 배치 추론 스크립트 (4bit 양자화 지원)

python -m vtimellm.eval.batch_infer \
  --val_path /data/kayoung/repos/projects/SurgVU/VTimeLLM/FINAL_converted_eval.json\
  --feat_folder /data2/local_datasets/YT_data/surgvu24_videos_only_features_endovit_only_cls/surgvu24 \
  --model_base /data/kayoung/repos/projects/SurgVU/VTimeLLM/checkpoints/vicuna-7b-v1.5 \
  --pretrain_mm_mlp_adapter /data/kayoung/repos/projects/SurgVU/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage1_1/mm_projector.bin \
  --stage2   /data2/local_datasets/YT_data/0909_checkpoints_F30/checkpoint-4560 \
  --stage3 '' \
  --out_path pred_015_F30.jsonl \
  --temperature 0.05 --max_new_tokens 512 \
  --load_4bit --bnb_4bit_quant_type nf4 --bnb_4bit_compute_dtype auto --device_map cuda:0

"""

import os
import sys
import json
import argparse
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from vtimellm.constants import IMAGE_TOKEN_INDEX
from vtimellm.conversation import conv_templates, SeparatorStyle
from vtimellm.model.builder import load_pretrained_model
from vtimellm.utils import disable_torch_init
from vtimellm.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **k: x  # fallback


def load_array_auto(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".npy", ".npz"]:
        arr = np.load(path, allow_pickle=True)
        if hasattr(arr, "files"):
            arr = arr[arr.files[0]]
        return np.asarray(arr)
    elif ext in [".pkl", ".pickle"]:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            for k in ["feat", "feats", "features", "image_features", "data"]:
                if k in obj:
                    obj = obj[k]; break
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()
        return np.asarray(obj)
    else:
        raise ValueError(f"Unsupported feature file: {path}")

def _to_tensor_and_squeeze(x: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(np.asarray(x))
    if t.ndim == 3 and t.shape[1] == 1:
        t = t.squeeze(1)
    return t

def build_features_like_training(
    feat_folder: str,
    feat_paths: List[str],
    start_time: float,
    stop_time: float,
    dtype: torch.dtype = torch.float16,
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    st = int(float(start_time))
    ed = int(float(stop_time))
    paths = [os.path.join(feat_folder, p.strip()) for p in feat_paths]

    if len(paths) != 1:
        p1, p2 = paths[0], paths[1]
        a1 = _to_tensor_and_squeeze(load_array_auto(p1))
        a2 = _to_tensor_and_squeeze(load_array_auto(p2))
        a1 = a1[st:, :]
        a2 = a2[:ed, :]
        image = torch.cat([a1, a2], dim=0)
        new_st, new_ed = 0, image.shape[0]
        if (new_ed - new_st) > 30:
            idx = np.linspace(new_st, new_ed - 1, num=30, dtype=int)
            image = image[idx, :]
        else:
            image = image[new_st:new_ed, :]
    else:
        a = _to_tensor_and_squeeze(load_array_auto(paths[0]))
        
        if (ed - st) > 30:
            idx = np.linspace(st, ed - 1, num=30, dtype=int)
            image = a[idx, :]
        else:
            image = a[st:ed, :]

    return image.to(dtype=dtype, device=device)  # (T<=100, D)

def run_inference(model, tokenizer, image_feats: torch.Tensor, query: str,
                  temperature: float, max_new_tokens: int, device: torch.device) -> str:
    if not query.startswith("<video>"):
        query = "<video>\n" + query

    conv = conv_templates["v1"].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            images=image_feats[None, ...],  # (1, T, D)
            do_sample=False, ######원래 True
            temperature=float(temperature),
            num_beams=1,
            max_new_tokens=int(max_new_tokens),
            use_cache=True,
            # stopping_criteria=[stopping_criteria],
        )

    input_token_len = input_ids.shape[1]
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0].strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    return outputs.strip()

def load_validation(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if path.lower().endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                items.append(json.loads(line))
    else:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
            if isinstance(obj, list):
                items = obj
            else:
                raise ValueError("Validation JSON must be a list of items.")
    return items

def _resolve_compute_dtype_for_feats(args) -> torch.dtype:
    if not getattr(args, "load_4bit", False):
        return torch.float16
    comp = getattr(args, "bnb_4bit_compute_dtype", "auto")
    if comp == "bf16":
        return torch.bfloat16
    if comp == "fp32":
        return torch.float32
    if comp == "fp16":
        return torch.float16
    try:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16

def _infer_device_from_model(model) -> torch.device:
    # 모델이 4bit + device_map으로 로드된 경우에도 실제 파라미터의 디바이스를 찾아 사용
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    # fallback
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    ap = argparse.ArgumentParser(description="VTimeLLM Stage2 Batch Inference (4bit ready)")
    ap.add_argument("--model_base", type=str, required=True)
    ap.add_argument("--pretrain_mm_mlp_adapter", type=str, required=True)
    ap.add_argument("--stage2", type=str, required=True)
    ap.add_argument("--stage3", type=str, default="")
    ap.add_argument("--val_path", type=str, required=True, help="validation .json or .jsonl")
    ap.add_argument("--feat_folder", type=str, required=True, help="features root folder")
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--out_path", type=str, default="pred_01epoch.jsonl")
    # 4bit
    ap.add_argument("--load_4bit", action="store_true")
    ap.add_argument("--bnb_4bit_quant_type", type=str, default="nf4", choices=["nf4", "fp4"])
    ap.add_argument("--bnb_4bit_compute_dtype", type=str, default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--no_bnb_4bit_double_quant", action="store_true")
    ap.add_argument("--device_map", type=str, default="auto")
    return ap.parse_args()

def main():
    args = parse_args()
    disable_torch_init()

    tokenizer, model, context_len = load_pretrained_model(
        args,
        stage2=args.stage2,
        stage3=(args.stage3 if args.stage3 else None),
    )

    # 4bit가 아닐 때만 명시적 half/cuda 이동
    if not getattr(args, "load_4bit", False):
        model = model.cuda().to(torch.float16)
    model.eval()

    # 모델 디바이스 동적 추론
    model_device = _infer_device_from_model(model)

    data = load_validation(args.val_path)

    features_cache: Dict[str, torch.Tensor] = {}
    feat_dtype = _resolve_compute_dtype_for_feats(args)

    out_f = open(args.out_path, "w", encoding="utf-8")

    for idx, item in enumerate(tqdm(data, desc="infer")):
        if "feat_path" not in item or "start_time" not in item or "stop_time" not in item:
            print(f"[warn] missing keys at index {idx}, skip.")
            continue

        feat_paths = item["feat_path"]
        if not isinstance(feat_paths, list) or not feat_paths:
            print(f"[warn] invalid feat_path at index {idx}, skip.")
            continue

        st = item.get("start_time", 0.0)
        ed = item.get("stop_time", 0.0)

        st_i, ed_i = int(float(st)), int(float(ed))
        key = f"{'|'.join(feat_paths)}|{st_i}|{ed_i}"

        if key not in features_cache:
            features_cache[key] = build_features_like_training(
                feat_folder=args.feat_folder,
                feat_paths=feat_paths,
                start_time=st,
                stop_time=ed,
                dtype=feat_dtype,
                device=model_device,  # ★ 모델 디바이스와 일치
            )

        feats = features_cache[key]

        conversations = item.get("conversations", [])
        human_questions: List[Tuple[int, str]] = []
        for ci, c in enumerate(conversations):
            if isinstance(c, dict) and c.get("from") == "human":
                q = c.get("value", "")
                human_questions.append((ci, q))

        if not human_questions:
            print(f"[warn] no human question at index {idx}, skip.")
            continue

        for ci, q in human_questions:
            try:
                pred = run_inference(
                    model=model,
                    tokenizer=tokenizer,
                    image_feats=feats,
                    query=q,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    device=model_device,  # ★ input_ids도 같은 디바이스
                )

                ref_answer = None
                if ci + 1 < len(conversations):
                    nxt = conversations[ci + 1]
                    if isinstance(nxt, dict) and nxt.get("from") == "gpt":
                        ref_answer = nxt.get("value")

                rec = {
                    "index": idx,
                    "id": item.get("id"),
                    "feat_path": feat_paths,
                    "start_time": st,
                    "stop_time": ed,
                    "question_index": ci,
                    "question": q,
                    "prediction": pred,
                    "reference": ref_answer,
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            except Exception as e:
                rec = {
                    "index": idx,
                    "id": item.get("id"),
                    "error": f"{type(e).__name__}: {e}",
                    "question_index": ci,
                    "question": q,
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    out_f.close()
    print(f"[info] saved -> {args.out_path}")

if __name__ == "__main__":
    main()
