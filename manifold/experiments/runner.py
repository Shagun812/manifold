from manifold.methods.logit_lens import run_logit_lens
from manifold.analysis.metrics import compute_metrics

def run_exp(loaded_model, experiment):
    logit_lens_res = run_logit_lens(loaded_model, experiment)
    metrics = compute_metrics(logit_lens_res, loaded_model, experiment)
    return {
        "experiment": experiment,
        "logit_lens": logit_lens_res,
        "metrics": metrics
    }