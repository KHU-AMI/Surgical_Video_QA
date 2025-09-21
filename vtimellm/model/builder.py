import os
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from vtimellm.model import *
from peft import PeftModel

def load_lora(model, lora_path):
    non_lora_trainables_path = os.path.join(lora_path, 'non_lora_trainables.bin')
    if os.path.exists(non_lora_trainables_path):
        non_lora_trainables = torch.load(non_lora_trainables_path, map_location='cpu')
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)
    print('Loading LoRA weights...')
    model = PeftModel.from_pretrained(model, lora_path)
    return model

def _resolve_bnb_compute_dtype(args) -> torch.dtype:
    comp = getattr(args, "bnb_4bit_compute_dtype", "auto")
    if comp == "bf16":
        return torch.bfloat16
    if comp == "fp16":
        return torch.float16
    if comp == "fp32":
        return torch.float32
    # auto
    try:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16

def _infer_model_device(model) -> torch.device:
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _move_non_quant_mm_modules_to(device, model):
    """
    4bit 양자화에서는 LLM 블록은 bnb Linear4bit라서 device_map에 맡기고,
    CPU에 남을 수 있는 projector/vision 계열만 안전하게 디바이스 이동.
    """
    core = model.get_model() if hasattr(model, "get_model") else model
    candidate_names = [
        "mm_projector",
        "multi_modal_projector",
        "video_projector",
        "image_projector",
        "vision_projector",
        "vision_resampler",
        "vision_tower",
    ]
    def _module_has_any_param(m):
        return any(p.device.type != "meta" for p in m.parameters(recurse=True))
    for name in candidate_names:
        if hasattr(core, name):
            sub = getattr(core, name)
            if sub is None:
                continue
            try:
                if _module_has_any_param(sub):
                    sub.to(device=device)
            except Exception as e:
                print(f"[warn] failed to move '{name}' to {device}: {type(e).__name__}: {e}")

def _cast_non_quant_mm_modules_dtype(model, dtype: torch.dtype):
    """
    projector/vision 계열 모듈의 파라미터/버퍼 dtype을 모델 계산 dtype(fp16/bf16 등)으로 강제.
    (bnb 4bit 레이어는 이 후보군에 보통 없음. 만약 포함돼도 .to(dtype=)는 내부에서 무시되거나 안전 실패 시 except로 넘어감)
    """
    core = model.get_model() if hasattr(model, "get_model") else model
    candidate_names = [
        "mm_projector",
        "multi_modal_projector",
        "video_projector",
        "image_projector",
        "vision_projector",
        "vision_resampler",
        "vision_tower",
    ]
    for name in candidate_names:
        if hasattr(core, name):
            sub = getattr(core, name)
            if sub is None:
                continue
            try:
                sub.to(dtype=dtype)
            except Exception as e:
                # 일부 모듈/레이어가 dtype 캐스트를 지원하지 않을 수 있으므로 경고만 남김
                print(f"[warn] failed to cast dtype for '{name}' to {dtype}: {type(e).__name__}: {e}")

def load_pretrained_model(args, stage2=None, stage3=None):
    # 기본 dtype은 half; 4bit 시에는 bnb compute dtype과 일관성 유지
    torch_dtype = torch.float16

    model_base = args.model_base
    load_4bit = bool(getattr(args, "load_4bit", False))
    device_map = getattr(args, "device_map", "auto")

    # 4bit에서는 기본값을 cuda:0으로 강제하여 CPU로 샤딩되는 것을 방지
    if load_4bit and (device_map is None or device_map == "auto"):
        print("[info] 4-bit 모드에서 device_map=auto → 'cuda:0'으로 강제합니다.")
        device_map = "cuda:0"

    quantization_config = None
    if load_4bit:
        compute_dtype = _resolve_bnb_compute_dtype(args)
        use_double_quant = not bool(getattr(args, "no_bnb_4bit_double_quant", False))
        quant_type = getattr(args, "bnb_4bit_quant_type", "nf4")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=use_double_quant,
            bnb_4bit_quant_type=quant_type,
        )
        torch_dtype = compute_dtype

    print('Loading VTimeLLM from base model...')
    if 'chatglm' in model_base:
        tokenizer = AutoTokenizer.from_pretrained(model_base, trust_remote_code=True)
        if load_4bit:
            model = VTimeLLMChatGLMForCausalLM.from_pretrained(
                model_base,
                quantization_config=quantization_config,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
        else:
            model = VTimeLLMChatGLMForCausalLM.from_pretrained(
                model_base,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        if load_4bit:
            model = VTimeLLMLlamaForCausalLM.from_pretrained(
                model_base,
                low_cpu_mem_usage=True,
                quantization_config=quantization_config,
                device_map=device_map,
                torch_dtype=torch_dtype,
            )
        else:
            model = VTimeLLMLlamaForCausalLM.from_pretrained(
                model_base,
                low_cpu_mem_usage=True,
                torch_dtype=torch_dtype,
            )
        # 토크나이저 크기 조정 이슈 방지
        token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

    # Stage1 비전 모듈 초기화
    model.get_model().initialize_vision_modules(args)

    # 모델 디바이스/계산 dtype 동기화
    model_device = _infer_model_device(model)
    _move_non_quant_mm_modules_to(model_device, model)
    _cast_non_quant_mm_modules_dtype(model, torch_dtype)

    # LoRA 스테이지 로드
    if stage2 is not None:
        print('Loading stage2 weights...')
        model = load_lora(model, stage2)
        if not load_4bit:
            print('Merging stage2 weights...')
            model = model.merge_and_unload()
        else:
            print('4-bit 모드: stage2는 merge하지 않고 어댑터 상태로 유지합니다.')
        # LoRA 후에도 다시 동기화(혹시 CPU/dtype으로 생긴 파라미터가 있으면 이동/캐스트)
        _move_non_quant_mm_modules_to(model_device, model)
        _cast_non_quant_mm_modules_dtype(model, torch_dtype)

        if stage3 is not None:
            print('Loading stage3 weights...')
            model = load_lora(model, stage3)
            if not load_4bit:
                print('Merging stage3 weights...')
                model = model.merge_and_unload()
            else:
                print('4-bit 모드: stage3도 merge하지 않습니다.')
            _move_non_quant_mm_modules_to(model_device, model)
            _cast_non_quant_mm_modules_dtype(model, torch_dtype)

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, context_len