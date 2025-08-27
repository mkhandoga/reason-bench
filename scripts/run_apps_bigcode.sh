#!/usr/bin/env bash
# Helper to run APPS with BigCode evaluation harness.
# Usage: bash scripts/run_apps_bigcode.sh Qwen/Qwen3-4B-Thinking-2507
set -euo pipefail
MODEL="${1:-Qwen/Qwen3-4B-Thinking-2507}"

# 1) GENERATION ONLY (GPU)
accelerate launch -m bigcode_eval.main   --model "$MODEL"   --tasks apps   --limit 100   --max_length_generation 2048   --temperature 0.6   --do_sample True   --n_samples 1   --batch_size 1   --precision bf16   --trust_remote_code   --save_generations   --generation_only   --save_generations_path generations_apps.json

# 2) EVAL ONLY (CPU ok)
python -m bigcode_eval.main   --tasks apps   --allow_code_execution   --load_generations_path generations_apps.json   --model "$MODEL"   --n_samples 1
