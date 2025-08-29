# Reason-Bench — Code Benchmarks with Master Runner

This framework lets you **benchmark and compare** code generation models across multiple tasks:

- ✅ **HumanEval / HumanEval+** (via **EvalPlus**)
- ✅ **MBPP / MBPP+** (via **EvalPlus**)  
- ✅ **APPS** (via **BigCode Evaluation Harness**, optional helper script provided)
- 🚧 **SWE-bench Lite** (local inference → predictions JSON; evaluate with official harness)

The framework supports **multiple models** including Qwen Thinking models and provides:
- 🔧 **YAML-based configuration** for running multiple models and benchmarks
- 📊 **Automated result collection** with JSON and CSV outputs
- ⚡ **vLLM support** for faster inference
- 🧠 **Reasoning-aware processing** for thinking models (strips `</think>` tags)

> **New:** Use the master `benchmark_runner.py` to run multiple models across multiple benchmarks with a single YAML config file.

---

## Quick start

### Option 1: Master Benchmark Runner (Recommended)

```bash
# 0) Create env
uv venv .venv && source .venv/bin/activate   # or python -m venv .venv
uv pip install -r requirements.txt            # or: pip install -r requirements.txt

# 1) Configure your benchmarks in a YAML file (see config_example.yaml)
cp config_example.yaml my_config.yaml
# Edit my_config.yaml to specify your models and benchmarks

# 2) Run all benchmarks
python benchmark_runner.py run my_config.yaml

# 3) Results are saved to benchmark_results.json and benchmark_summary.csv
```

### Option 2: Individual Benchmark Commands

```bash
# HumanEval (EvalPlus)
python bench.py humaneval --model Qwen/Qwen3-4B-Thinking-2507 --use-vllm --enable-reasoning

# MBPP (EvalPlus) 
python bench.py mbpp --model Qwen/Qwen3-4B-Thinking-2507 --use-vllm --enable-reasoning

# SWE-bench Lite (baseline skeleton)
python bench.py swe-lite --model Qwen/Qwen3-4B-Thinking-2507 --limit 10 --max-new-tokens 2048

# Then evaluate with the official harness:
python -m swebench.harness.run_evaluation --dataset_name SWE-bench/SWE-bench_Lite --predictions_path outputs/swe_lite_predictions.json --max_workers 1
```

> **GPU memory:** Most models run comfortably on 16–24 GB GPUs with `bfloat16` and vLLM.
> The master runner provides progress tracking and handles timeouts automatically.

---

## Configuration Format

The master runner uses YAML configuration files to define which models and benchmarks to run. Here's the structure:

```yaml
# Global defaults (applied to all models unless overridden)
defaults:
  temperature: 0.6
  top_p: 0.95
  top_k: 20
  seed: 1234
  dtype: "bfloat16"
  max_new_tokens: 4096
  n_samples: 1
  use_vllm: true
  enable_reasoning: true

# Models to benchmark
models:
  - name: "Qwen/Qwen3-4B-Thinking-2507"
    # Uses all defaults
  - name: "microsoft/CodeT5p-770M"
    # Override specific settings for this model
    enable_reasoning: false
    max_new_tokens: 2048

# Benchmarks to run
benchmarks:
  - humaneval
  - mbpp
  # - swe_lite

# Output settings
output:
  results_file: "benchmark_results.json"
  summary_file: "benchmark_summary.csv" 
  outdir_base: "outputs"
```

The runner will create a matrix of all models × all benchmarks and run each combination, saving detailed results and a summary table.

---

## Why these tools?

- **EvalPlus** gives **HumanEval+ / MBPP+** with rigorous extra tests (less false positives).
- **BigCode Evaluation Harness** covers **APPS** and many other code sets. A helper shell
  script is included to call it with Qwen defaults (generation-only → eval-only mode).
- **SWE-bench** provides an official, dockerized harness. We include a **local inference
  skeleton** that writes predictions in the expected JSON format so you can evaluate with
  their runner.

---

## Results & Configuration

### Master Runner Output

The benchmark runner produces:
- **`benchmark_results.json`**: Detailed results with all metrics, timing, and configuration
- **`benchmark_summary.csv`**: Summary table with key metrics for easy analysis  
- **Progress display**: Real-time progress bar and results table

### Individual Benchmark Parameters

Key reproducibility flags for individual `bench.py` commands:

- `--n-samples` (for pass@k), `--temperature`, `--top-p`, `--top-k`, `--seed`
- `--max-new-tokens` (increase for hard coding tasks)
- `--dtype` `{float16|bfloat16|float32}`
- `--use-vllm` (faster inference), `--enable-reasoning` (for thinking models)

Individual benchmark outputs land in `outputs/` as JSONL/JSON/CSV.

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

## Notes on Reasoning Models

**Thinking models** (like Qwen *Thinking*) often emit chain-of-thought ending with `</think>` followed by the
**final answer**. When `--enable-reasoning` is used, the framework strips anything before `</think>` and then tries to extract a
```python code block``` if present. The code passed to evaluators is the post-`</think>`
section (code block contents if found, otherwise the raw tail text).

For non-reasoning models, set `enable_reasoning: false` in your config to disable this processing.

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

## Project Structure

- `benchmark_runner.py` — Master script for running multiple benchmarks with YAML config
- `bench.py` — Individual benchmark runner (HumanEval, MBPP, SWE-bench Lite)
- `config_example.yaml` — Example configuration file
- `model_utils.py` — Model loading utilities with HuggingFace Transformers
- `vllm_model_utils.py` — vLLM-based model utilities for faster inference
- `tasks/` — Benchmark-specific task implementations
- `outputs/` — Results directory (created automatically)

---

## License

Apache-2.0 for this repo. Respect the licenses of datasets and external tools.
