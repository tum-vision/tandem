from functools import wraps
import numpy as np
import torch


def make_recursive_func(func):
    @wraps(func)
    def wrapper(x, **kwargs):
        if isinstance(x, list):
            return [wrapper(xx, **kwargs) for xx in x]
        elif isinstance(x, tuple):
            return tuple([wrapper(xx, **kwargs) for xx in x])
        elif isinstance(x, dict):
            return {k: wrapper(v, **kwargs) for k, v in x.items()}
        else:
            return func(x, **kwargs)

    return wrapper


@make_recursive_func
def tensor2numpy(x):
    if isinstance(x, (np.ndarray, str)):
        return x
    elif torch.is_tensor(x):
        return x.detach().cpu().numpy().copy()
    else:
        raise NotImplementedError(f"Invalid input type {type(x)} for tensor2numpy")

@make_recursive_func
def to_device(x, *, device):
    if torch.is_tensor(x):
        return x.to(device)
    elif isinstance(x, str):
        return x
    else:
        raise NotImplementedError(f"Invalid input type {type(x)} for to_device")
