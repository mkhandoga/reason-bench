#!/usr/bin/env python3
"""
Master benchmark runner that executes multiple models across multiple benchmarks
based on a YAML configuration file.
"""
import os
import sys
import yaml
import json
import csv
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()

class BenchmarkRunner:
    def __init__(self, config_path: str):
        """Initialize the benchmark runner with a YAML config file."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.results = []
        self.start_time = datetime.now()
        
        # Ensure output directories exist
        self.setup_output_dirs()
    
    def setup_output_dirs(self):
        """Create necessary output directories."""
        base_dir = Path(self.config.get('output', {}).get('outdir_base', 'outputs'))
        base_dir.mkdir(exist_ok=True)
        
        for benchmark in self.config['benchmarks']:
            (base_dir / benchmark).mkdir(exist_ok=True)
    
    def merge_config(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge model-specific config with defaults."""
        defaults = self.config.get('defaults', {})
        merged = defaults.copy()
        
        # Remove 'name' from model_config for merging
        model_params = {k: v for k, v in model_config.items() if k != 'name'}
        merged.update(model_params)
        
        return merged
    
    def build_command(self, benchmark: str, model_name: str, config: Dict[str, Any]) -> List[str]:
        """Build the command to run a specific benchmark."""
        cmd = [sys.executable, 'bench.py', benchmark, '--model', model_name]
        
        # Add all configuration parameters as command line arguments
        for key, value in config.items():
            if key == 'name':
                continue
            
            # Convert parameter names to CLI format
            cli_key = f"--{key.replace('_', '-')}"
            
            if isinstance(value, bool):
                if value:
                    cmd.append(cli_key)
                # else:
                #     cmd.append(f"--no-{key.replace('_', '-')}")
            else:
                cmd.extend([cli_key, str(value)])
        
        return cmd
    
    def run_single_benchmark(self, model_name: str, benchmark: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single model on a single benchmark."""
        console.print(f"Running {benchmark} on {model_name}...")
        
        # Build command
        cmd = self.build_command(benchmark, model_name, config)
        
        # Set output directory
        outdir_base = self.config.get('output', {}).get('outdir_base', 'outputs')
        safe_model_name = model_name.replace('/', '_').replace('\\', '_')
        outdir = f"{outdir_base}/{benchmark}_{safe_model_name}"
        
        # Add outdir to command
        cmd.extend(['--outdir', outdir])
        
        start_time = time.time()
        
        try:
            # Run the benchmark
            if config.get('debug', False):
                console.print(f"[dim]Command:[/dim] {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )
            
            duration = time.time() - start_time
            
            if result.returncode == 0:
                # Parse results from stdout or result files
                metrics = self.parse_benchmark_results(outdir, benchmark)
                status = "success"
                error = None
            else:
                metrics = {}
                status = "failed"
                error = result.stderr
                console.print(f"[red]Failed:[/red] {model_name} on {benchmark}")
                console.print(f"[dim]Error:[/dim] {error[:200]}...")
        
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            metrics = {}
            status = "timeout"
            error = "Benchmark timed out after 1 hour"
            console.print(f"[yellow]Timeout:[/yellow] {model_name} on {benchmark}")
        
        except Exception as e:
            duration = time.time() - start_time
            metrics = {}
            status = "error"
            error = str(e)
            console.print(f"[red]Error:[/red] {model_name} on {benchmark}: {e}")
        
        return {
            'model': model_name,
            'benchmark': benchmark,
            'status': status,
            'duration_seconds': duration,
            'metrics': metrics,
            'error': error,
            'config': config,
            'timestamp': datetime.now().isoformat(),
            'outdir': outdir
        }
    
    def parse_benchmark_results(self, outdir: str, benchmark: str) -> Dict[str, Any]:
        """Parse benchmark results from output files."""
        metrics = {}
        
        # Try to read CSV summary file
        summary_file = Path(outdir) / f"{benchmark}_summary.csv"
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        metrics[row['metric']] = row['value']
            except Exception as e:
                console.print(f"[yellow]Warning:[/yellow] Could not parse {summary_file}: {e}")
        
        return metrics
    
    def run_all_benchmarks(self) -> List[Dict[str, Any]]:
        """Run all configured benchmarks on all models."""
        models = self.config['models']
        benchmarks = self.config['benchmarks']
        
        total_runs = len(models) * len(benchmarks)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            
            main_task = progress.add_task("Running benchmarks", total=total_runs)
            
            for model_config in models:
                model_name = model_config['name']
                merged_config = self.merge_config(model_config)
                
                for benchmark in benchmarks:
                    progress.update(
                        main_task, 
                        description=f"Running {benchmark} on {model_name.split('/')[-1]}"
                    )
                    
                    result = self.run_single_benchmark(model_name, benchmark, merged_config)
                    self.results.append(result)
                    
                    progress.advance(main_task)
        
        return self.results
    
    def save_results(self):
        """Save results to output files."""
        output_config = self.config.get('output', {})
        
        # Save detailed JSON results
        results_file = output_config.get('results_file', 'benchmark_results.json')
        with open(results_file, 'w') as f:
            json.dump({
                'config': self.config,
                'run_info': {
                    'start_time': self.start_time.isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'total_runs': len(self.results),
                },
                'results': self.results
            }, f, indent=2)
        
        console.print(f"[green]Results saved to:[/green] {results_file}")
        
        # Save summary CSV
        summary_file = output_config.get('summary_file', 'benchmark_summary.csv')
        self.save_summary_csv(summary_file)
        
        console.print(f"[green]Summary saved to:[/green] {summary_file}")
    
    def save_summary_csv(self, filename: str):
        """Save a summary CSV with key metrics."""
        if not self.results:
            return
        
        # Collect all unique metric names
        all_metrics = set()
        for result in self.results:
            all_metrics.update(result['metrics'].keys())
        
        fieldnames = [
            'model', 'benchmark', 'status', 'duration_seconds'
        ] + sorted(all_metrics)
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                row = {
                    'model': result['model'],
                    'benchmark': result['benchmark'],
                    'status': result['status'],
                    'duration_seconds': f"{result['duration_seconds']:.1f}",
                }
                
                # Add metrics
                for metric in all_metrics:
                    row[metric] = result['metrics'].get(metric, '')
                
                writer.writerow(row)
    
    def print_summary_table(self):
        """Print a nice summary table of results."""
        if not self.results:
            return
        
        # Group results by model
        models = {}
        for result in self.results:
            model = result['model']
            if model not in models:
                models[model] = {}
            models[model][result['benchmark']] = result
        
        table = Table(title="Benchmark Results Summary")
        table.add_column("Model", style="cyan")
        
        # Add benchmark columns
        benchmarks = self.config['benchmarks']
        for benchmark in benchmarks:
            table.add_column(benchmark.upper(), justify="center")
        
        table.add_column("Total Time", justify="right", style="magenta")
        
        for model, model_results in models.items():
            row = [model.split('/')[-1]]  # Use short model name
            
            total_time = 0
            for benchmark in benchmarks:
                if benchmark in model_results:
                    result = model_results[benchmark]
                    total_time += result['duration_seconds']
                    
                    if result['status'] == 'success':
                        # Try to show pass@1 score if available
                        metrics = result['metrics']
                        pass_at_1 = None
                        for key, value in metrics.items():
                            if 'pass@1' in key.lower():
                                try:
                                    pass_at_1 = float(value)
                                    break
                                except:
                                    pass
                        
                        if pass_at_1 is not None:
                            row.append(f"✅ {pass_at_1:.1%}")
                        else:
                            row.append("✅")
                    elif result['status'] == 'failed':
                        row.append("❌")
                    elif result['status'] == 'timeout':
                        row.append("⏰")
                    else:
                        row.append("❓")
                else:
                    row.append("-")
            
            row.append(f"{total_time/60:.1f}m")
            table.add_row(*row)
        
        console.print()
        console.print(table)

@app.command()
def run(
    config_file: str = typer.Argument(..., help="Path to YAML configuration file"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be run without executing"),
):
    """Run benchmarks based on a YAML configuration file."""
    
    if not os.path.exists(config_file):
        console.print(f"[red]Error:[/red] Configuration file '{config_file}' not found")
        raise typer.Exit(1)
    
    try:
        runner = BenchmarkRunner(config_file)
    except Exception as e:
        console.print(f"[red]Error loading config:[/red] {e}")
        raise typer.Exit(1)
    
    # Show configuration summary
    console.print(Panel.fit(
        f"[bold]Benchmark Configuration[/bold]\n\n"
        f"Models: {len(runner.config['models'])}\n"
        f"Benchmarks: {len(runner.config['benchmarks'])}\n"
        f"Total runs: {len(runner.config['models']) * len(runner.config['benchmarks'])}\n"
        f"Config file: {config_file}",
        title="🚀 Benchmark Runner"
    ))
    
    if dry_run:
        console.print("[yellow]DRY RUN MODE - No benchmarks will be executed[/yellow]\n")
        
        for model_config in runner.config['models']:
            model_name = model_config['name']
            merged_config = runner.merge_config(model_config)
            
            console.print(f"[cyan]Model:[/cyan] {model_name}")
            for benchmark in runner.config['benchmarks']:
                cmd = runner.build_command(benchmark, model_name, merged_config)
                console.print(f"  [dim]{benchmark}:[/dim] {' '.join(cmd)}")
            console.print()
        return
    
    # Confirm before running
    if not typer.confirm("Continue with benchmark execution?"):
        console.print("Cancelled.")
        raise typer.Exit(0)
    
    # Run benchmarks
    console.print("[bold green]Starting benchmark execution...[/bold green]\n")
    
    try:
        runner.run_all_benchmarks()
        runner.save_results()
        runner.print_summary_table()
        
        console.print(f"\n[bold green]✅ Benchmark run completed![/bold green]")
        console.print(f"Total time: {(datetime.now() - runner.start_time).total_seconds() / 60:.1f} minutes")
        
    except KeyboardInterrupt:
        console.print(f"\n[yellow]❌ Benchmark run interrupted by user[/yellow]")
        if runner.results:
            console.print("Saving partial results...")
            runner.save_results()
        raise typer.Exit(1)
    
    except Exception as e:
        console.print(f"\n[red]❌ Benchmark run failed:[/red] {e}")
        if runner.results:
            console.print("Saving partial results...")
            runner.save_results()
        raise typer.Exit(1)

if __name__ == "__main__":
    app()