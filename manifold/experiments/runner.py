from manifold.methods.logit_lens import run_logit_lens
from manifold.analysis.metrics import compute_metrics
from manifold.tracking.writer import save_logit_lens
from manifold.methods.activation_patching import run_activation_patching

def run_exp(loaded_model, experiment, cfg):
    if cfg["methods"]["logit_lens"]:
        logit_lens_res = run_logit_lens(loaded_model, experiment)
        metrics = compute_metrics(logit_lens_res, loaded_model, experiment)
        model_name= loaded_model["name"]

        saved_files= save_logit_lens(experiment,model_name, metrics)
        
        return {
            "experiment": experiment,
            "logit_lens": logit_lens_res,
            "metrics": metrics,
            "artifacts_path": saved_files
        }
    raise ValueError("Logit Lens is disabled in configs/default.yaml.")

def run_patching_exp(loaded_model, experiment, cfg):

    if not cfg["methods"]["activation_patching"]:
        raise ValueError("Activation patching is disabled in the configuration")

    return run_activation_patching(loaded_model, experiment)

        
        