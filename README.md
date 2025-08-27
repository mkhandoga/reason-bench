# Bench Qwen3 *Thinking* — Code Benchmarks (HumanEval/MBPP via EvalPlus, SWE-bench Lite skeleton)

This mini-framework lets you **reproduce and extend** code benchmarks for
[Qwen/Qwen3-4B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507):

- ✅ **HumanEval / HumanEval+** (via **EvalPlus**)
- ✅ **MBPP / MBPP+** (via **EvalPlus**)
- ✅ **APPS** (via **BigCode Evaluation Harness**, optional helper script provided)
- 🚧 **SWE-bench Lite** (local HF inference → predictions JSON; evaluate with official harness)

It standardizes **generation params** for Qwen *Thinking* models (reasoning tag `</think>`)
and strips the chain-of-thought before returning the **final code**.

> Tip: The Qwen model card recommends **Transformers ≥ 4.51.0** and decoding like
> `temperature=0.6, top_p=0.95, top_k=20, seed=1234` and long outputs for hard tasks.
> We default to these here and expose CLI flags to change them.

---

## Quick start

```bash
# 0) Create env
uv venv .venv && source .venv/bin/activate   # or python -m venv .venv
uv pip install -r requirements.txt            # or: pip install -r requirements.txt

# 1) HumanEval (EvalPlus)
python bench.py humaneval   --model Qwen/Qwen3-4B-Thinking-2507   --n-samples 1 --max-new-tokens 4096 --dtype bfloat16

# 2) MBPP (EvalPlus)
python bench.py mbpp   --model Qwen/Qwen3-4B-Thinking-2507   --n-samples 1 --max-new-tokens 4096 --dtype bfloat16

# 3) Evaluate results are printed and saved under outputs/
#    (EvalPlus also produces JSON summaries.)

# 4) SWE-bench Lite (baseline skeleton)
# Generate predictions with local HF model:
python bench.py swe-lite   --model Qwen/Qwen3-4B-Thinking-2507   --limit 10 --max-new-tokens 2048 --dtype bfloat16

# Then evaluate with the official harness (inside the same env or another machine):
python -m swebench.harness.run_evaluation   --dataset_name SWE-bench/SWE-bench_Lite   --predictions_path outputs/swe_lite_predictions.json   --max_workers 1
```

> **GPU memory:** 4B *Thinking* runs comfortably on a single 16–24 GB GPU with `bfloat16`.
> For very long outputs (e.g., 8k–32k+ tokens), prefer vLLM/TGI or reduce batch size.
> (Quantization is okay but may slightly change scores.)

---

## Why these tools?

- **EvalPlus** gives **HumanEval+ / MBPP+** with rigorous extra tests (less false positives).
- **BigCode Evaluation Harness** covers **APPS** and many other code sets. A helper shell
  script is included to call it with Qwen defaults (generation-only → eval-only mode).
- **SWE-bench** provides an official, dockerized harness. We include a **local inference
  skeleton** that writes predictions in the expected JSON format so you can evaluate with
  their runner.

---

## Results & Repro switches

Key reproducibility flags you’ll likely touch:

- `--n-samples` (for pass@k), `--temperature`, `--top-p`, `--top-k`, `--seed`
- `--max-new-tokens` (increase for hard coding tasks)
- `--dtype` `{float16|bfloat16|float32}`
- `--trust-remote-code` (enabled by default for Qwen)

Outputs land in `outputs/` as JSONL/JSON/CSV.

---

## APPS (optional, via BigCode harness)

We provide a helper script that
- **generates** with the HF model (using harness) and saves `generations.json`
- then **evaluates** those generations in a **CPU-only** docker or locally

```bash
bash scripts/run_apps_bigcode.sh Qwen/Qwen3-4B-Thinking-2507
```

You can edit the script to change `n_samples`, `batch_size`, etc.

---

## Notes on Qwen *Thinking* outputs

Qwen *Thinking* models often emit chain-of-thought ending with `</think>` followed by the
**final answer**. Our wrappers strip anything before `</think>` and then try to extract a
```python code block``` if present. The code we pass to evaluators is the post-`</think>`
section (code block contents if found, otherwise the raw tail text).

---

## Caveats

- **APPS/LiveCodeBench/OJBench**: use their official harnesses for leaderboard-quality
  numbers; we include only a thin helper for APPS.
- **SWE-bench Lite** here is a **single-shot patch generator** baseline—no retrieval,
  no multi-turn tools. Expect low scores; upgrade by plugging in SWE-Agent style retrieval
  and multi-step planning.
- Unit-test execution is sandboxed by EvalPlus; still, run in a container if you need
  extra isolation.

---

## License

Apache-2.0 for this repo. Respect the licenses of datasets and external tools.
