import torch

def generate(loaded_model: dict, prompt: str, config )-> str:

    model = loaded_model["model"]
    tokenizer = loaded_model["tokenizer"]
    device = loaded_model["device"]

    cfg = config["generation"]

    inputs = tokenizer(prompt, return_tensors = "pt").to(device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens = cfg["max_new_tokens"],
            temperature = cfg["temperature"],
            do_sample = cfg["do_sample"],
            top_p = cfg["top_p"]

        )

    input_length = inputs["input_ids"].shape[1]

    generated_ids = output[0, input_length:]

    response = tokenizer.decode(generated_ids, skip_special_tokens = True)

    return response