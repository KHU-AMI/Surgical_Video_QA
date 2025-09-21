#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VTimeLLM Stage2 배치 추론 스크립트

- 입력: validation JSON(리스트 형태) 또는 JSONL
- 각 항목의 feat_path/start_time/stop_time을 이용해 훈련과 동일한 전처리로 (<=100, D) 피처 생성
- 같은 (feat_paths, start, stop) 구간은 캐싱하여 여러 질문에 재사용
- conversations 중 'from' == 'human' 인 모든 질문에 대해 생성
- 출력: JSONL (한 줄당 한 개의 (샘플, 질문) 결과)
# augi5 : /data2/local_datasets/YT_data/surgvu24_videos_only_features_endovit_only_cls/surgvu24
# augi2 : /data2/local_datasets/SurgVU/surgvu24_videos_only_features_endovit_only_cls/surgvu24
예시:
    python -m vtimellm.eval.batch_infer \
      --val_path /data/kayoung/repos/projects/SurgVU/VTimeLLM/FINAL_converted_eval.json \
      --feat_folder /data2/local_datasets/YT_data/surgvu24_videos_only_features_endovit_only_cls/surgvu24 \
      --model_base /data/kayoung/repos/projects/SurgVU/VTimeLLM/checkpoints/vicuna-7b-v1.5 \
      --pretrain_mm_mlp_adapter /data/kayoung/repos/projects/SurgVU/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage1_1/mm_projector.bin \
      --stage2 /data/kayoung/repos/projects/SurgVU/VTimeLLM/checkpoints/0904_checkpoints/checkpoint-382_epoch2 \
      --stage3 '' \
      --out_path pred_epoch_002.jsonl \
      --temperature 0.05 --max_new_tokens 512
"""

import os
import sys
import json
import argparse
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

# VTimeLLM modules
from vtimellm.constants import IMAGE_TOKEN_INDEX
from vtimellm.conversation import conv_templates, SeparatorStyle
from vtimellm.model.builder import load_pretrained_model
from vtimellm.utils import disable_torch_init
from vtimellm.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **k: x  # fallback


# -------------------------------
# 로딩 유틸: npy/npz/pkl 지원
# -------------------------------
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
    if t.ndim == 3 and t.shape[1] == 1:  # (N,1,D) 저장된 경우
        t = t.squeeze(1)
    return t


# ------------------------------------------------------
# 훈련과 동일 전처리: 두 파트 이어붙이기 + 슬라이스 + 100프레임
# ------------------------------------------------------
def build_features_like_training(
    feat_folder: str,
    feat_paths: List[str],
    start_time: float,
    stop_time: float,
) -> torch.Tensor:
    st = int(float(start_time))
    ed = int(float(stop_time))
    paths = [os.path.join(feat_folder, p.strip()) for p in feat_paths]

    if len(paths) != 1:
        p1, p2 = paths[0], paths[1]
        a1 = _to_tensor_and_squeeze(load_array_auto(p1))  # (N1, D)
        a2 = _to_tensor_and_squeeze(load_array_auto(p2))  # (N2, D)
        a1 = a1[st:, :]
        a2 = a2[:ed, :]
        image = torch.cat([a1, a2], dim=0)
        new_st, new_ed = 0, image.shape[0]
        if (new_ed - new_st) > 100:
            idx = np.linspace(new_st, new_ed - 1, num=100, dtype=int)
            image = image[idx, :]
        else:
            image = image[new_st:new_ed, :]
    else:
        a = _to_tensor_and_squeeze(load_array_auto(paths[0]))
        if (ed - st) > 100:
            idx = np.linspace(st, ed - 1, num=100, dtype=int)
            image = a[idx, :]
        else:
            image = a[st:ed, :]

    return image.to(torch.float16).to("cuda")  # (T<=100, D)


# -------------------------------
# 생성
# -------------------------------
def run_inference(model, tokenizer, image_feats: torch.Tensor, query: str,
                  temperature: float, max_new_tokens: int) -> str:
    if not query.startswith("<video>"):
        query = "<video>\n" + query

    conv = conv_templates["v1"].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to("cuda")

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            images=image_feats[None, ...],  # (1, T, D)
            do_sample=True,
            temperature=float(temperature),
            num_beams=1,
            max_new_tokens=int(max_new_tokens),
            use_cache=True,
            # 필요 시 중단 기준 활성화:
            # stopping_criteria=[stopping_criteria],
        )

    input_token_len = input_ids.shape[1]
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0].strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    return outputs.strip()


# -------------------------------
# 데이터 로딩
# -------------------------------
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


def parse_args():
    ap = argparse.ArgumentParser(description="VTimeLLM Stage2 Batch Inference")
    # 모델/어댑터
    ap.add_argument("--model_base", type=str, required=True)
    ap.add_argument("--pretrain_mm_mlp_adapter", type=str, required=True)
    ap.add_argument("--stage2", type=str, required=True)
    ap.add_argument("--stage3", type=str, default="")

    # 데이터
    ap.add_argument("--val_path", type=str, required=True, help="validation .json or .jsonl")
    ap.add_argument("--feat_folder", type=str, required=True, help="features root folder")

    # 생성 옵션
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--max_new_tokens", type=int, default=512)

    # 출력
    ap.add_argument("--out_path", type=str, default="pred_01epoch.jsonl")

    return ap.parse_args()


def main():
    args = parse_args()
    disable_torch_init()

    # 모델 로드 (builder는 Stage2/3 merge를 수행)
    tokenizer, model, context_len = load_pretrained_model(
        args,
        stage2=args.stage2,
        stage3=(args.stage3 if args.stage3 else None),
    )
    model = model.cuda().to(torch.float16)
    model.eval()

    # 데이터 로드
    data = load_validation(args.val_path)

    # 피처 캐시: 동일 (feat_paths, st, ed) 조합 재사용
    features_cache: Dict[str, torch.Tensor] = {}

    # 출력 스트림
    out_f = open(args.out_path, "w", encoding="utf-8")
    
    ###################################################################
    # 이 부분이 FEATURE를 불러오는 부분입니다 
    ##################################################################

    for idx, item in enumerate(tqdm(data, desc="infer")):
        # 필수 키 점검
        if "feat_path" not in item or "start_time" not in item or "stop_time" not in item:
            # 스킵 또는 에러
            print(f"[warn] missing keys at index {idx}, skip.")
            continue

        feat_paths = item["feat_path"]  # list
        if not isinstance(feat_paths, list) or not feat_paths:
            print(f"[warn] invalid feat_path at index {idx}, skip.")
            continue

        st = item.get("start_time", 0.0)
        ed = item.get("stop_time", 0.0)

        # 캐시 키 구성(훈련과 동일하게 int로 캐스팅해 쓰므로 키도 int)
        st_i, ed_i = int(float(st)), int(float(ed))
        key = f"{'|'.join(feat_paths)}|{st_i}|{ed_i}"

        # 피처 준비
        if key not in features_cache:
            features_cache[key] = build_features_like_training(
                feat_folder=args.feat_folder,
                feat_paths=feat_paths,
                start_time=st,
                stop_time=ed,
            )  # (T<=100, D) half, cuda

        feats = features_cache[key]

        # 질문들 추출(conversations 중 human)
        conversations = item.get("conversations", [])
        human_questions: List[Tuple[int, str]] = []
        for ci, c in enumerate(conversations):
            if isinstance(c, dict) and c.get("from") == "human":
                q = c.get("value", "")
                human_questions.append((ci, q))

        # 질문이 없다면 스킵
        if not human_questions:
            print(f"[warn] no human question at index {idx}, skip.")
            continue

        # 각 질문에 대해 생성
        for ci, q in human_questions:
            try:
                pred = run_inference(
                    model=model,
                    tokenizer=tokenizer,
                    image_feats=feats,
                    query=q,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                )

                # 참조 답(있으면)도 기록
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
