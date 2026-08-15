import typer
from manifold.inference.generate import generate
from manifold.models.loader import load_model
from manifold.utils.config_loader import load_config
from manifold.experiments.experiment import Experiment
from manifold.experiments.runner import run_exp, run_patching_exp

from rich.table import Table
from rich.console import Console

console = Console()

app = typer.Typer(help="MI with pytorch hooks")


@app.command("gen")
def gen_cmd(prompt:str= typer.Option(...,"-p","--prompt", help = "Prompt to generate from"))-> None:
    if not prompt.strip():
        raise typer.BadParameter("Prompt can't be empty")
    
    cfg = load_config()

    loaded_model = load_model(cfg)

    result = generate(loaded_model, prompt, cfg)
    typer.echo(result)

@app.command("run_lens")
def run_lens():
    cfg = load_config()
    loaded_model = load_model(cfg)

    experiment = Experiment(prompt="The capital of France is called", target=" Paris")
    
    results = run_exp(loaded_model, experiment, cfg)
    metrics = results["metrics"]
    artifacts_path= results["artifacts_path"]

    console.print(f"[bold]Model:[/bold] {cfg['model']['name']}")
    console.print(f"[bold]Prompt:[/bold] {experiment.prompt}")
    console.print(f"[bold]Target:[/bold] {experiment.target}")
    console.print()

    prob_table = Table(title="Correct Token Probability")
    prob_table.add_column("Layer")
    prob_table.add_column("Probability")
    
    for layer, prob in metrics["correct_token_probability"].items():
        prob_table.add_row(str(layer),f"{prob:.8f}")
    
    console.print(prob_table)

    for layer, preds in metrics["top_k_predictions"].items():

        table= Table(title=f"Layer {layer}")
        table.add_column("Rank", justify="center")
        table.add_column("Token")
        table.add_column("Probability", justify="right")

        for rank, pred in enumerate(preds, start=1):
            table.add_row(str(rank), repr(pred["token"]), f"{pred['probability']:.6f}")

        console.print(table)
    console.print(f"artifacts saved:{artifacts_path}")


@app.command("patch")
def patch_cmd():
    cfg= load_config()
    loaded_model= load_model(cfg)

    experiment= Experiment(target=" Na",clean_prompt="The chemical symbol for sodium is", corrupted_prompt="The chemical symbol for potassium is",)

    results= run_patching_exp(loaded_model, experiment, cfg)

    recovery_score= results["attention_recovery"]

    console.print(f"[bold]Model:[/bold] {cfg['model']['name']}")
    console.print(f"[bold]Clean:[/bold] {experiment.clean_prompt}")
    console.print(f"[bold]Corrupted:[/bold] {experiment.corrupted_prompt}")
    console.print(f"[bold]Target:[/bold] {experiment.target}")

    table= Table(title="Attention recovery")
    table.add_column("Layer", justify="center")
    table.add_column("Recovery", justify="right")

    for layer, result in recovery_score.items():
        table.add_row(str(layer), f"{result['recovery']:.6f}")

    console.print(table)
