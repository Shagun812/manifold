# Manifold

**Manifold** is a lightweight mechanistic interpretability framework for studying how **model compression affects the internal representations and causal behavior of transformer models**.

The project is built from scratch using PyTorch hooks and Hugging Face Transformers, with an emphasis on keeping the framework small, transparent, and easy to extend.

> **Status:  Ongoing Research / Development**

---

## What is Manifold?

Manifold investigates what happens inside a transformer when its weights are compressed.

The current research direction is:

**FP32 model → INT8 quantization → compare internal behavior**

Rather than looking only at whether quantization changes accuracy, Manifold uses mechanistic interpretability methods to investigate whether the model's internal representations and causal mechanisms remain stable after compression.

The current experiments use **EleutherAI/Pythia-160M**.

---

## Current Capabilities

### Logit Lens

Manifold can inspect how predictions develop across transformer layers.

Currently it measures:

- Correct-token probability at each layer
- Top-k token predictions at each layer

### Activation Patching

Manifold can perform layer-wise activation patching using clean and corrupted prompts.

Currently it:

- caches clean attention outputs
- patches them into corrupted runs
- measures target-token logit recovery
- reports recovery for each transformer layer

### Experiment Tracking

Experiments produce concise JSON artifacts containing the experiment setup and results.

Example:

```text
artifacts/
├── logit_lens/
│   └── experiment_<timestamp>.json
└── activation_patching/
    └── experiment_<timestamp>.json
```

## CLI Commands


Run logit lens: 
```text
manifold run_lens
```
Run activation patching:
```text
manifold patch
```
Generate Text:
```text
manifold gen -p "Your prompt here"
```

## Currently Working On
 - INT8 post-training quantization
 - FP32 vs INT8 experimental pipeline
 - Quantized-model validation

## Next

Once quantization is implemented, the project will move into the actual experimental phase:

 - Design controlled prompt suite
 - Run FP32 baselines
 - Run INT8 experiments
 - Compare Logit Lens behavior
- Compare activation-patching behavior
 - Analyze and document findings

 ## Research Direction

The central question is:

Does compressing a transformer change the internal representations and causal mechanisms that produce its behavior?

Manifold will investigate this by comparing interpretability results between the original and quantized models.

The goal is not to assume that quantization preserves or destroys internal mechanisms, but to measure what actually changes.
