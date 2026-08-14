"""
Return a forward hook that replaces a module's output with a cached activation.
"""

from manifold.hooks.cache import ActivationCache

def replace_hook(cache: ActivationCache, activation_name: str):

    def hook(module, inputs, output):

        if not cache.contains(activation_name):
            raise KeyError(f"Activation '{activation_name}' not in cache")

        replace= cache.get(activation_name)

        if replace.shape != output.shape:
            raise ValueError(f"Shape mismatch for '{activation_name}")

        if isinstance(output, tuple):
            return (replace, *output[1:])
        
        return replace

    return hook
