#!/usr/bin/env python3
import os, sys, json, math, random, pathlib, time
import typer
from rich import print
from typing import Optional
from tasks.humaneval_evalplus import run_humaneval
from tasks.mbpp_evalplus import run_mbpp
from tasks.swebench_lite import run_swe_lite

app = typer.Typer(add_completion=False, no_args_is_help=True)

COMMON_HELP = "Default decoding per Qwen Thinking card: temp=0.6, top_p=0.95, top_k=20, seed=1234."

@app.command(help=f"Run HumanEval/HumanEval+ using EvalPlus. {COMMON_HELP}")
def humaneval(
    model: str = typer.Option(..., "--model", help="HF model id or local path"),
    n_samples: int = typer.Option(1, help="Generations per task (for pass@k)"),
    max_new_tokens: int = typer.Option(4096),
    temperature: float = typer.Option(0.6),
    top_p: float = typer.Option(0.95),
    top_k: int = typer.Option(20),
    seed: int = typer.Option(1234),
    dtype: str = typer.Option("bfloat16", help="float16|bfloat16|float32"),
    outdir: str = typer.Option("outputs/humaneval", help="Where to save generations & reports"),
    trust_remote_code: bool = typer.Option(True),
):
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    result = run_humaneval(
        model=model,
        n_samples=n_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        dtype=dtype,
        outdir=outdir,
        trust_remote_code=trust_remote_code,
    )
    print("[bold green]Done.[/bold green] Results:", result)

@app.command(help=f"Run MBPP/MBPP+ using EvalPlus. {COMMON_HELP}")
def mbpp(
    model: str = typer.Option(..., "--model", help="HF model id or local path"),
    n_samples: int = typer.Option(1, help="Generations per task (for pass@k)"),
    max_new_tokens: int = typer.Option(4096),
    temperature: float = typer.Option(0.6),
    top_p: float = typer.Option(0.95),
    top_k: int = typer.Option(20),
    seed: int = typer.Option(1234),
    dtype: str = typer.Option("bfloat16", help="float16|bfloat16|float32"),
    outdir: str = typer.Option("outputs/mbpp", help="Where to save generations & reports"),
    trust_remote_code: bool = typer.Option(True),
):
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    result = run_mbpp(
        model=model,
        n_samples=n_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        dtype=dtype,
        outdir=outdir,
        trust_remote_code=trust_remote_code,
    )
    print("[bold green]Done.[/bold green] Results:", result)

@app.command(help="Generate baseline predictions for SWE-bench Lite (then evaluate with official harness).")
def swe_lite(
    model: str = typer.Option(..., "--model", help="HF model id or local path"),
    limit: Optional[int] = typer.Option(None, help="Number of instances to run (for quick tests)"),
    max_new_tokens: int = typer.Option(2048),
    temperature: float = typer.Option(0.2, help="Greedy-ish is often better for patches"),
    top_p: float = typer.Option(0.95),
    top_k: int = typer.Option(20),
    seed: int = typer.Option(1234),
    dtype: str = typer.Option("bfloat16", help="float16|bfloat16|float32"),
    outpath: str = typer.Option("outputs/swe_lite_predictions.json", help="Where to save predictions JSON"),
    trust_remote_code: bool = typer.Option(True),
    use_vllm: bool = typer.Option(True, help="Use VLLM for faster inference (fallback to transformers if not available)"),
):
    pathlib.Path(os.path.dirname(outpath) or ".").mkdir(parents=True, exist_ok=True)
    run_swe_lite(
        model=model,
        limit=limit,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        dtype=dtype,
        outpath=outpath,
        trust_remote_code=trust_remote_code,
        use_vllm=use_vllm,
    )
    print(f"[bold green]Wrote[/bold green] {outpath}. Evaluate with swebench harness (see README).")

if __name__ == "__main__":
    app()
