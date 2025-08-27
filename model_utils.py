from typing import List, Tuple, Optional
import torch, re, random, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass

DEFAULT_SYSTEM = (
    "You are a helpful Python coding assistant. "
    "Think privately, then output ONLY the final code solution.\n"
    "Rules:\n"
    "1) Do NOT include explanations.\n"
    "2) Return Python code in a single fenced block like:\n"
    "```python\n<code here>\n```\n"
    "3) If tests or function signature are given, satisfy them exactly.\n"
)

@dataclass
class QwenGenConfig:
    model: str
    max_new_tokens: int = 4096
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    seed: int = 1234
    dtype: str = "bfloat16"
    trust_remote_code: bool = True

class QwenThinkingRunner:
    def __init__(self, cfg: QwenGenConfig):
        import torch
        torch.manual_seed(cfg.seed)
        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        torch_dtype = dtype_map.get(cfg.dtype, torch.bfloat16)
        
        print(f"Loading model {cfg.model}...")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=cfg.trust_remote_code)
        print(f"Tokenizer loaded. Loading model with dtype={torch_dtype}...")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=cfg.trust_remote_code,
        )
        
        # Print device placement
        print(f"Model loaded. Device placement:")
        for name, param in self.model.named_parameters():
            if hasattr(param, 'device'):
                print(f"  {name}: {param.device}")
                break  # Just show first parameter's device
        
        self.cfg = cfg

    @staticmethod
    def _strip_think_and_extract_code(text: str) -> str:
        # Keep only the portion after the last </think>
        if "</think>" in text:
            text = text.split("</think>")[-1]
        # Extract triple-backtick code block (python or any)
        import re
        m = re.search(r"```(?:python)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        # Fallback: remove markdown lines that aren't code-like
        return text.strip()

    def generate_code(self, prompt: str, n: int = 1, system: Optional[str] = None) -> List[str]:
        messages = [
            {"role": "system", "content": system or DEFAULT_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        text_inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text_inputs, return_tensors="pt").to(self.model.device)

        gen_cfg = dict(
            do_sample=self.cfg.temperature > 0,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            top_k=self.cfg.top_k,
            max_new_tokens=self.cfg.max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        outs = []
        for _ in range(n):
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, **gen_cfg)
            out_text = self.tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            outs.append(self._strip_think_and_extract_code(out_text))
        return outs
