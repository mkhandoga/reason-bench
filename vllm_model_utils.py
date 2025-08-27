from typing import List, Optional
from dataclasses import dataclass
import re

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
class VLLMConfig:
    model: str
    max_new_tokens: int = 4096
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    seed: int = 1234
    dtype: str = "bfloat16"
    trust_remote_code: bool = True
    tensor_parallel_size: int = 1  # Set to number of GPUs
    gpu_memory_utilization: float = 0.9

class VLLMRunner:
    def __init__(self, cfg: VLLMConfig):
        from vllm import LLM, SamplingParams
        
        print(f"Loading model {cfg.model} with VLLM...")
        self.llm = LLM(
            model=cfg.model,
            dtype=cfg.dtype,
            trust_remote_code=cfg.trust_remote_code,
            tensor_parallel_size=cfg.tensor_parallel_size,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            seed=cfg.seed,
        )
        
        self.sampling_params = SamplingParams(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            max_tokens=cfg.max_new_tokens,
            seed=cfg.seed,
        )
        
        self.cfg = cfg
        print("VLLM model loaded successfully!")

    @staticmethod
    def _strip_think_and_extract_code(text: str) -> str:
        # Keep only the portion after the last </think>
        if "</think>" in text:
            text = text.split("</think>")[-1]
        # Extract triple-backtick code block (python or any)
        m = re.search(r"```(?:python)?\s*([\s\S]*?)```", text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        # Fallback: remove markdown lines that aren't code-like
        return text.strip()

    def generate_code(self, prompt: str, n: int = 1, system: Optional[str] = None) -> List[str]:
        # Format as chat messages
        messages = [
            {"role": "system", "content": system or DEFAULT_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        
        # Apply chat template
        formatted_prompt = self.llm.get_tokenizer().apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Update sampling params for this generation
        sampling_params = SamplingParams(
            temperature=self.sampling_params.temperature,
            top_p=self.sampling_params.top_p,
            top_k=self.sampling_params.top_k,
            max_tokens=self.sampling_params.max_tokens,
            n=n,
            seed=self.cfg.seed,
        )
        
        # Generate
        outputs = self.llm.generate([formatted_prompt], sampling_params)
        
        # Extract generated text
        results = []
        for output in outputs[0].outputs:
            generated_text = output.text
            results.append(self._strip_think_and_extract_code(generated_text))
        
        return results