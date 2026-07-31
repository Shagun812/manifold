from manifold.hooks.manager import HookManager
from manifold.hooks.cache import ActivationCache
from manifold.architectures.gpt_neox_adapter import GPTNeoXAdapter
import torch

def cache_hook(cache: ActivationCache, layer_name: str):

        def hook(module, inputs, output):

            if isinstance(output, tuple):
                output = output[0]
            
            cache.store(layer_name, output)

            return output
        
        return hook

def project_logits(residual, adapter)-> torch.Tensor:
        with torch.inference_mode():
            residual= adapter.final_layer_norm(residual)
            logits= adapter.lm_head(residual)
            next_token_logits = logits[:, -1, :]

        return next_token_logits

def run_logit_lens(loaded_model:dict, experiment)-> dict:

    model= loaded_model["model"]
    tokenizer= loaded_model["tokenizer"]
    device= loaded_model["device"]

    hook_manager= HookManager()
    activation_cache= ActivationCache()
    adapter= GPTNeoXAdapter(model)


    tokenized_inputs =tokenizer(experiment.prompt, return_tensors = "pt").to(device)

    
    try:
        #Register Hooks
        for layer_idx, module in enumerate(adapter.residual_stream):
            hook = cache_hook(activation_cache, layer_idx)
            hook_manager.register(module, hook)
        
        #Forward pass
        with torch.inference_mode():
            model(**tokenized_inputs)

    finally:
        #Remove hooks
        hook_manager.remove()

    layer_logits = {}

    for layer_name, residual in activation_cache.get_all().items():

        layer_logits[layer_name] = project_logits(residual, adapter)

    return {
        "prompt": experiment.prompt,
        "layer_logits": layer_logits
    }