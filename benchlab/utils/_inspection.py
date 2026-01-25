import inspect

__all__ = ["get_init_args"]


def get_init_args(obj):
    sig = inspect.signature(obj.__init__)
    return {
        name: getattr(obj, name)
        for name in sig.parameters
        if name != "self" and hasattr(obj, name)
    }
