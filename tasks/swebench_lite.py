from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import os, json, random
from datasets import load_dataset
from ._shared import build_runner

SWE_PROMPT_TMPL = '''# Task
You are given a GitHub issue description for a repository.
Write a **unified diff patch** (GNU diff) that fixes the bug or implements the feature.

# Output format
Return ONLY the patch between the markers:
*** Begin Patch
<unified diff here>
*** End Patch

# Constraints
- Keep the patch minimal.
- Preserve style and tests.
- Do not explain or add prose.
- Use correct file paths as in the repo.

# Issue
{issue}
'''

@dataclass
class SWEArgs:
    model: str
    limit: Optional[int]
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    seed: int
    dtype: str
    outpath: str
    trust_remote_code: bool
    use_vllm: bool = True

def _extract_patch(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>")[-1]
    # Find patch between markers
    import re
    m = re.search(r"\*\*\* Begin Patch(.*?)\*\*\* End Patch", text, re.DOTALL | re.IGNORECASE)
    if m:
        return "*** Begin Patch\n" + m.group(1).strip() + "\n*** End Patch"
    # Fallback: return raw (will likely fail)
    return text.strip()

def run_swe_lite(**kwargs):
    args = SWEArgs(**kwargs)
    ds = load_dataset("SWE-bench/SWE-bench_Lite", split="test")
    if args.limit is not None:
        ds = ds.select(range(min(args.limit, len(ds))))
    runner = build_runner(args, use_vllm=args.use_vllm)

    preds: List[Dict[str, Any]] = []
    total = len(ds)
    print(f"Processing {total} instances...")
    
    for i, rec in enumerate(ds):
        print(f"[{i+1}/{total}] Processing {rec['instance_id']}...")
        prompt = SWE_PROMPT_TMPL.format(issue=rec["problem_statement"])
        outs = runner.generate_code(prompt, n=1, system=(
            "You are a precise software patch generator. Think privately, then output ONLY the unified diff."
        ))
        patch = _extract_patch(outs[0])
        preds.append({
            "instance_id": rec["instance_id"],
            "model_name_or_path": args.model,
            "model_patch": patch,
        })
        print(f"  Generated patch ({len(patch)} chars)")

    os.makedirs(os.path.dirname(args.outpath) or ".", exist_ok=True)
    with open(args.outpath, "w") as f:
        json.dump(preds, f, indent=2)
