from dataclasses import dataclass
from typing import Dict, Any, Union
import csv, os

def build_runner(args, use_vllm: bool = True) -> Union['QwenThinkingRunner', 'VLLMRunner']:
    if use_vllm:
        try:
            from vllm_model_utils import VLLMRunner, VLLMConfig
            cfg = VLLMConfig(
                model=args.model,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                seed=args.seed,
                dtype=args.dtype,
                trust_remote_code=args.trust_remote_code,
                enable_reasoning=getattr(args, 'enable_reasoning', True),
                max_model_len=getattr(args, 'max_model_len', 8192),
            )
            return VLLMRunner(cfg)
        except ImportError:
            print("VLLM not available, falling back to transformers...")
            use_vllm = False
    
    if not use_vllm:
        from model_utils import QwenThinkingRunner, QwenGenConfig
        cfg = QwenGenConfig(
            model=args.model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            enable_reasoning=getattr(args, 'enable_reasoning', True),
        )
        return QwenThinkingRunner(cfg)

def save_csv_summary(path: str, metrics: Dict[str, Any]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in metrics.items():
            w.writerow([k, v])
