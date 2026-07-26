"""
Manage the lifecycle of PyTorch forward hooks.

This class registers and removes forward hooks without knowing anything
about the hook's purpose.
"""



class HookManager:
    
    def __init__(self):
        self.handles = []

    def register(self, module, hook_fn):
        """
        Attach a forward hook to a module and track its handle so it can
        be removed safely later.
        """
        handle = module.register_forward_hook(hook_fn)
        self.handles.append(handle)
        return handle
    
    def remove(self):
        for handle in self.handles:
            handle.remove()
            
        self.handles.clear()
    
    def has_hooks(self):
        return len(self.handles)>0