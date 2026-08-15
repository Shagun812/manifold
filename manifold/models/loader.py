from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Cache of already loaded model
MODEL_CACHE = {}

def get_device(device: str)-> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    else:
        return device

def load_model(cfg):

    model_name = cfg["model"]["name"]

    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name] 
    
    dtype_map={
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16
    }

    dtype= dtype_map[cfg["inference"]["dtype"]]
    

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)

    model.eval()

    device = get_device(cfg["inference"]["device"])
    model.to(device)

    loaded_model = {
        "name": model_name,
        "model": model,
        "tokenizer": tokenizer,
        "device": device
    }
    MODEL_CACHE[model_name] = loaded_model


    return loaded_model