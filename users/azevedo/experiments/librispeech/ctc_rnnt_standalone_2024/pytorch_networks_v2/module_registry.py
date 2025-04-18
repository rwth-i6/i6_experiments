_CONFIG_REGISTRY = {}

def register_config(name: str):
    def decorator(cls):
        if name in _CONFIG_REGISTRY:
            raise KeyError(f"{name} already in registry.")
        _CONFIG_REGISTRY[name] = cls
        return cls
    return decorator

def get_config(name: str):
    if name not in _CONFIG_REGISTRY:
        raise KeyError(f"Config {name} not in registry")

    return _CONFIG_REGISTRY[name]

def set_config(name: str, cls):
    if name in _CONFIG_REGISTRY:
        raise KeyError("name already in registry.")
    
    _CONFIG_REGISTRY[name] = cls