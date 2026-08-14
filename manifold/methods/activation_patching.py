from manifold.hooks.manager import HookManager
from manifold.hooks.cache import ActivationCache
from manifold.hooks.patch import replace_hook
from manifold.architectures.gpt_neox_adapter import GPTNeoXAdapter
import torch


def cache_hook(cache: ActivationCache, activation_name: str):

        def hook(module, inputs, output):

            if isinstance(output, tuple):
                output = output[0]
            
            cache.store(activation_name, output)

            return output
        
        return hook


def cache_clean_activations(model, adapter, clean_inputs) -> ActivationCache:
    hook_manager= HookManager()
    activation_cache= ActivationCache()

    try:
        for layer_idx, module in enumerate(adapter.attention_output):
            activation_name= f"attention_output_{layer_idx}"
            hook= cache_hook(activation_cache, activation_name)
            hook_manager.register(module, hook)
        
        with torch.inference_mode():
            model(**clean_inputs)
    finally:
        hook_manager.remove()
    
    return activation_cache

def get_logits(model, inputs):

    with torch.inference_mode():
        outputs=model(**inputs)

    return outputs.logits[:,-1,:]


def run_single_patch(model, corrupted_inputs, module, clean_cache, activation_name):

    hook_manager= HookManager()
    
    hook_manager.register(module, replace_hook(clean_cache,activation_name))

    try:
        with torch.inference_mode():
            outputs= model(**corrupted_inputs)

    finally:
        hook_manager.remove()

    return outputs.logits[:,-1,:]


def compute_patch_recovery(clean_logits, corrupted_logits, patched_logits, target_token_id):

    clean_logit=clean_logits[target_token_id]
    corrupted_logit=corrupted_logits[target_token_id]
    patched_logit=patched_logits[target_token_id]

    denominator= clean_logit-corrupted_logit

    if torch.abs(denominator) < 1e-8:
        return 0.0
    recovery=(patched_logit-corrupted_logit)/ denominator

    return recovery.item()


def patch_attention(loaded_model, experiment):

    model=loaded_model["model"]
    adapter= GPTNeoXAdapter(model)
    tokenizer= loaded_model["tokenizer"]
    device = loaded_model["device"]

    tokenized_clean_inputs= tokenizer(experiment.clean_prompt, return_tensors="pt").to(device)
    tokenized_corrupted_inputs= tokenizer(experiment.corrupted_prompt, return_tensors="pt").to(device)

    clean_logits=get_logits(model, tokenized_clean_inputs)
    corrupted_logits=get_logits(model, tokenized_corrupted_inputs)

    clean_cache= cache_clean_activations(model, adapter,tokenized_clean_inputs)

    target_ids= tokenizer.encode(experiment.target, add_special_tokens=False)

    if len(target_ids) != 1:
            raise ValueError("Activation patching currently supports only single-token targets")
    
    target_token_id= target_ids[0]

    if (tokenized_clean_inputs["input_ids"].shape != tokenized_corrupted_inputs["input_ids"].shape):
        raise ValueError("Clean and corrupted prompts must have the same tokenized shape for activation patching")

    results={}

    for layer_idx, module in enumerate(adapter.attention_output):

        activation_name= f"attention_output_{layer_idx}"

        patched_logits= run_single_patch(model=model, corrupted_inputs=tokenized_corrupted_inputs, module=module, clean_cache=clean_cache, activation_name= activation_name)

        recovery= compute_patch_recovery(clean_logits.squeeze(0), corrupted_logits.squeeze(0), patched_logits.squeeze(0), target_token_id)

        results[layer_idx]={"recovery": recovery}
    
    return results


def run_activation_patching(loaded_model: dict, experiment):

    attention_recovery= patch_attention(loaded_model, experiment)
    return {
        "clean_prompt": experiment.clean_prompt,
        "corrupted_prompt": experiment.corrupted_prompt,
        "target": experiment.target,
        "attention_recovery": attention_recovery
    }