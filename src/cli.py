"""
Token Squeeze — CLI Entry Point
================================
Command-line interface for running experiments and utilities.

Usage:
    python -m src.cli fertility --data-dir data/
    python -m src.cli drift --data-dir data/ --model llama3-8b
    python -m src.cli expand --strategy bte --target-lang it
    python -m src.cli spar --model llama3-8b --lang it
    python -m src.cli run-all
    python -m src.cli generate-seeds --num 100
"""

import logging
import sys
from pathlib import Path

import click
import yaml

# ──────────────────────────────────────────────────────────
# Logging Setup
# ──────────────────────────────────────────────────────────

def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s │ %(name)-20s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("token_squeeze.log"),
        ],
    )


# ──────────────────────────────────────────────────────────
# CLI Group
# ──────────────────────────────────────────────────────────

@click.group()
@click.option("--config", default="configs/default.yaml", help="Config file path")
@click.option("--verbose", is_flag=True, help="Enable debug logging")
@click.pass_context
def cli(ctx, config: str, verbose: bool):
    """🔬 Token Squeeze Hypothesis — Experimental CLI"""
    setup_logging(verbose)
    ctx.ensure_object(dict)
    with open(config) as f:
        ctx.obj["config"] = yaml.safe_load(f)
    ctx.obj["config_path"] = config


# ──────────────────────────────────────────────────────────
# Command: generate-seeds
# ──────────────────────────────────────────────────────────

@cli.command()
@click.option("--num", default=100, help="Number of conversations to generate")
@click.option("--output", default="data/seeds", help="Output directory")
@click.pass_context
def generate_seeds(ctx, num: int, output: str):
    """Generate seed conversations for Poly-DriftBench."""
    from src.data_gen.seed_generator import generate_seeds_with_llm

    config = ctx.obj["config"]
    min_turns = config["dataset"]["min_turns"]
    max_turns = config["dataset"]["max_turns"]

    generate_seeds_with_llm(
        output_dir=output,
        num_conversations=num,
        min_turns=min_turns,
        max_turns=max_turns,
    )


# ──────────────────────────────────────────────────────────
# Command: produce (Multi-Agent Data Factory)
# ──────────────────────────────────────────────────────────

@cli.command()
@click.option("--num", default=10, help="Number of conversations to produce")
@click.option("--output", default="data", help="Output data directory")
@click.option("--model", default="gpt-4o", help="LLM model for agents")
@click.option("--no-translate", is_flag=True, help="Skip translation phase")
@click.option("--parallel", default=3, help="Parallel conversations (default 3)")
@click.pass_context
def produce(ctx, num: int, output: str, model: str, no_translate: bool, parallel: int):
    """🏭 Run the multi-agent data factory to produce Poly-DriftBench."""
    from src.data_gen.pipeline import DataFactory

    config = ctx.obj["config"]
    factory = DataFactory(config=config, model=model)

    factory.produce_dataset(
        output_dir=output,
        num_conversations=num,
        min_turns=config["dataset"]["min_turns"],
        max_turns=config["dataset"]["max_turns"],
        translate=not no_translate,
        parallel_conversations=parallel,
    )


# ──────────────────────────────────────────────────────────
# Command: fertility
# ──────────────────────────────────────────────────────────

@cli.command()
@click.option("--data-dir", default="data", help="Data directory")
@click.pass_context
def fertility(ctx, data_dir: str):
    """Run Experiment 1: Token Fertility Profiling."""
    from src.experiments.runner import ExperimentRunner

    runner = ExperimentRunner(ctx.obj["config_path"])
    df = runner.run_experiment_1(data_dir)
    if not df.empty:
        click.echo("\n📊 Fertility Results:")
        click.echo(df.to_string(index=False))


# ──────────────────────────────────────────────────────────
# Command: drift
# ──────────────────────────────────────────────────────────

@cli.command()
@click.option("--data-dir", default="data", help="Data directory")
@click.option("--num-runs", default=None, type=int, help="Number of runs")
@click.pass_context
def drift(ctx, data_dir: str, num_runs: int):
    """Run Experiment 2: Baseline Drift Measurement."""
    from src.experiments.runner import ExperimentRunner

    runner = ExperimentRunner(ctx.obj["config_path"])
    runner.run_experiment_2(data_dir, num_runs=num_runs)


# ──────────────────────────────────────────────────────────
# Command: expand
# ──────────────────────────────────────────────────────────

@cli.command()
@click.option("--strategy", type=click.Choice(["bte", "cpi", "cri", "all"]), default="all")
@click.option("--target-lang", default="it", help="Target language for token matching")
@click.option("--data-dir", default="data", help="Data directory")
@click.pass_context
def expand(ctx, strategy: str, target_lang: str, data_dir: str):
    """Run paraphrastic expansion on English dataset."""
    click.echo(f"🔄 Expanding English to match {target_lang} token counts")
    click.echo(f"   Strategy: {strategy}")
    # TODO: Wire up expansion pipeline
    click.echo("   (Not yet implemented — run Experiment 1 first)")


# ──────────────────────────────────────────────────────────
# Command: spar
# ──────────────────────────────────────────────────────────

@cli.command()
@click.option("--data-dir", default="data", help="Data directory")
@click.pass_context
def spar(ctx, data_dir: str):
    """Run Experiment 5: SPAR Attention Analysis."""
    from src.experiments.runner import ExperimentRunner

    runner = ExperimentRunner(ctx.obj["config_path"])
    runner.run_experiment_5(data_dir)


# ──────────────────────────────────────────────────────────
# Command: run-all
# ──────────────────────────────────────────────────────────

@cli.command()
@click.option("--data-dir", default="data", help="Data directory")
@click.pass_context
def run_all(ctx, data_dir: str):
    """Run the full experimental pipeline."""
    from src.experiments.runner import ExperimentRunner

    runner = ExperimentRunner(ctx.obj["config_path"])
    runner.run_all(data_dir)


# ──────────────────────────────────────────────────────────
# Command: sample
# ──────────────────────────────────────────────────────────

@cli.command()
@click.pass_context
def sample(ctx):
    """Generate and evaluate a sample conversation (no GPU/API needed)."""
    from src.data_gen.seed_generator import create_sample_conversation
    from src.evaluation.ddm import DDMEvaluator

    click.echo("🧪 Running sample evaluation...")

    conv = create_sample_conversation()
    evaluator = DDMEvaluator(ctx.obj["config"], language="en")

    # Extract assistant responses
    responses = [
        m["content"] for m in conv["messages"] if m["role"] == "assistant"
    ]

    result = evaluator.evaluate_conversation(
        responses=responses,
        conversation_id=conv["id"],
        language="en",
        model_name="sample",
    )

    click.echo(f"\n📋 Sample Conversation: {conv['id']}")
    click.echo(f"   Domain: {conv['domain']}")
    click.echo(f"   Turns evaluated: {result.total_turns}")
    click.echo(f"   ── Core Metrics ──")
    click.echo(f"   Mean DDM:           {result.mean_ddm:.3f} ± {result.std_ddm:.3f}")
    click.echo(f"   AUC:                {result.auc:.3f}")
    click.echo(f"   Drift Onset (DOP):  {result.drift_onset_point}")
    click.echo(f"   Sustained DOP (k=3):{result.sustained_dop}")
    click.echo(f"   Half-Life (τ½):     {result.half_life}")
    click.echo(f"   Total Collapse:     {result.total_collapse_point}")
    click.echo(f"   Recovery Rate:      {result.recovery_rate:.2%}")
    click.echo(f"   ── Per-Level Decay ──")
    for level, decay in result.per_level_decay.items():
        click.echo(
            f"   {level:15s}: mean={decay.mean_score:.3f}  "
            f"AUC={decay.auc:.3f}  onset={decay.decay_onset}"
        )
    click.echo(f"   ── Turn-by-Turn ──")

    for tr in result.turn_results:
        status = "✅" if tr.all_pass else "⚠️"
        click.echo(
            f"   {status} Turn {tr.turn_number}: DDM={tr.ddm_score:.3f} "
            f"[L1={tr.l1_score:.1f} L2={tr.l2_score:.2f} "
            f"L3={tr.l3_score:.1f} L4={tr.l4_score:.1f}]"
        )


if __name__ == "__main__":
    cli()
