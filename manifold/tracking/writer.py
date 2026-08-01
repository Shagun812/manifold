from manifold.utils.config_loader import load_config
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
import json

def save_logit_lens(experiment,model_name:str, metrics)-> dict:

    cfg = load_config()
    artifact_root=Path(cfg["artifacts"]["root"])
    logit_lens_dir= artifact_root/"logit_lens" 

    experiment_data = {**asdict(experiment), "model": model_name}

    if not logit_lens_dir.exists():
        logit_lens_dir.mkdir(parents=True, exist_ok=True)
    now= datetime.now()

    #formatting
    timestamp=now.strftime("%Y-%m-%d_%H-%M-%S")
    filename1= f"experiment_{timestamp}.json"
    filename2= f"metrics_{timestamp}.json"

    filepath1= logit_lens_dir/filename1
    filepath2= logit_lens_dir/filename2


    with open(filepath1, "w", encoding="utf-8") as f:
        json.dump(experiment_data, f, indent=4, ensure_ascii=False)

    with open(filepath2, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    
    return {"experiment": filepath1, "metrics": filename2}