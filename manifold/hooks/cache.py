from torch import Tensor

class ActivationCache:
    """
    Store activations captured during model execution.

    The cache is independent of hook registration and interpretability methods.
    """

    def __init__(self):
        self.activations: dict[str, Tensor] = {}
    
    def store(self, name, activation: Tensor)-> None:
        self.activations[name] = activation.detach().clone()
    
    def get(self, name)-> Tensor:
        return self.activations[name]

    def get_all(self)-> dict[str, Tensor]:
        return self.activations.copy()
    
    def clear(self)-> None:
        self.activations.clear()
    
    def contains(self, name)-> bool:
        return name in self.activations
            