import typer
from manifold.inference.generate import generate
from manifold.models.loader import load_model
from manifold.utils.config_loader import load_config


app = typer.Typer(help="MI with pytorch hooks")


@app.command("gen")
def gen_cmd(prompt:str= typer.Option(...,"-p","--prompt", help = "Prompt to generate from"))-> None:
    if not prompt.strip():
        raise typer.BadParameter("Prompt can't be empty")
    
    cfg = load_config()

    loaded_model = load_model(cfg)

    result = generate(loaded_model, prompt, cfg)
    typer.echo(result)
