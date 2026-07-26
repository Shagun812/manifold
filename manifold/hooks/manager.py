
class HookManager:
    
    def __init__(self):
        self.handles = []

    def register(self, module, hook_fn):
        handle = module.register_forward_hook(hook_fn)
        self.handles.append(handle)
        return handle
    
    def remove(self):
        for handle in self.handles:
            handle.remove()
            
        self.handles.clear()
    
    def has_hooks(self):
        return len(self.handles)>0