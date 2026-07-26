"""
Adapter for GPT-Neox based architectures(Pythia).

Provides a stable inference for accessing transformer components required by 
mechanistic interpretability algorithms.
"""
    
    class GPTNeoXAdapter:

        def __init__(self, model):
            self.model = model

        @property 
        def layers(self):
            return self.model.gpt_neox.layers
        
        @property
        def attention_modules(self):
            return [ layer.attention for layer in self.layers]
        
        @property
        def mlp_modules(self):
            return [layer.mlp for layer in self.layers]

        @property
        def attention_output(self):
            return [ layer.attention.dense for layer in self.layers]
        
        @property
        def mlp_output(self):
            return [ layer.mlp.dense_4h_to_h for layer in self.layers]
        
        @property
        def final_layer_norm(self):
            return self.model.gpt_neox.final_layer_norm
        
        @property
        def lm_head(self):
            return self.model.lm_head
        
        @property
        def embedding(self):
            return self.model.gpt_neox.embed_in
        