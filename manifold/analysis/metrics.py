
import torch

def compute_correct_token_probability(logits: torch.Tensor, target_token_id: int)-> float:
    prob= torch.softmax(logits, dim=-1)
    return prob[0,target_token_id].item()

def compute_top_k_pred(logits: torch.Tensor, tokenizer, k:int):
    logits = logits.squeeze()
    probs= torch.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, k=k)

    preds = []

    for token_id, probability in zip(top_ids.tolist(), top_probs.tolist()):
        token = tokenizer.decode([token_id])

        preds.append({"token": token, "probability":probability})
    return preds

def compute_metrics(logit_lens_res: dict, loaded_model:dict, experiment,k:int=3)-> dict:

    tokenizer = loaded_model["tokenizer"]
    layer_logits = logit_lens_res["layer_logits"]

    target_ids = tokenizer.encode(experiment.target, add_special_tokens=False)
    if len(target_ids)!=1:
        raise ValueError("Metrics support only single-token targets")

    target_token_id= target_ids[0]
    
    correct_token_probs ={}
    top_k_preds={}

    for layer, logits in layer_logits.items():
        correct_token_probs[layer] =(compute_correct_token_probability(logits, target_token_id))
 
        top_k_preds[layer]= (compute_top_k_pred(logits, tokenizer,k))

    return {
        "correct_token_probability": correct_token_probs,
        "top_k_predictions": top_k_preds
    }