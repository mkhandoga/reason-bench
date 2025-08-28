from typing import Dict, Any
import os, json, pathlib, subprocess, sys
from dataclasses import dataclass
from evalplus.data import get_mbpp_plus, write_jsonl
from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.console import Console
from ._shared import build_runner, save_csv_summary

@dataclass
class MBArgs:
    model: str
    n_samples: int
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    seed: int
    dtype: str
    outdir: str
    trust_remote_code: bool
    debug: bool = False
    use_vllm: bool = True
    enable_reasoning: bool = True
    max_model_len: int = 8192

def run_mbpp(**kwargs) -> Dict[str, Any]:
    args = MBArgs(**kwargs)
    runner = build_runner(args, use_vllm=args.use_vllm)
    problems = get_mbpp_plus()  # dict: task_id -> {"prompt": "...", "test_imports":..., "test_list":...}
    
    console = Console()
    
    gens = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        main_task = progress.add_task("Processing problems", total=len(problems))
        
        for task_id, prob in problems.items():
            progress.update(main_task, description=f"Processing {task_id}")
            
            if args.debug:
                console.print(f"\n[bold blue]Task {task_id}:[/bold blue]")
                console.print(f"[dim]Prompt:[/dim] {prob['prompt'][:100]}...")
            
            samples = runner.generate_code(prob["prompt"], n=args.n_samples)
            
            if args.debug:
                for i, sample in enumerate(samples):
                    console.print(f"[dim]Response {i+1}:[/dim] {sample[:200]}...")
            
            for s in samples:
                gens.append({"task_id": task_id, "completion": s})
            
            progress.advance(main_task)

    samples_path = os.path.join(args.outdir, "mbpp_samples.jsonl")
    os.makedirs(args.outdir, exist_ok=True)
    write_jsonl(samples_path, gens)

    # Evaluate MBPP and MBPP+
    cmd1 = [sys.executable, "-m", "evalplus.evaluate", "--dataset", "mbpp", "--samples", samples_path]
    cmd2 = [sys.executable, "-m", "evalplus.evaluate", "--dataset", "mbppplus", "--samples", samples_path]

    print("Running:", " ".join(cmd1))
    r1 = subprocess.run(cmd1, capture_output=True, text=True)
    print(r1.stdout)
    print(r1.stderr, file=sys.stderr)

    print("Running:", " ".join(cmd2))
    r2 = subprocess.run(cmd2, capture_output=True, text=True)
    print(r2.stdout)
    print(r2.stderr, file=sys.stderr)

    # Save terse CSV
    metrics = {}
    for line in (r1.stdout + "\n" + r2.stdout).splitlines():
        if "pass@1" in line and ":" in line:
            k, v = line.split(":", 1)
            metrics[k.strip()] = v.strip()
    save_csv_summary(os.path.join(args.outdir, "mbpp_summary.csv"), metrics)
    return metrics
