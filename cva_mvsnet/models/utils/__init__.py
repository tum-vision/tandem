import torch
import numpy as np
from .tb_logger import TBLogger
from .warmup_multi_step_lr import WarmupMultiStepLR
from .helpers import make_recursive_func, tensor2numpy


class StreamingBinCount:
    def __init__(self, minlength):
        self.counts = np.zeros([minlength], dtype=np.int64)
        
    def bincount(self, x):
        out = np.bincount(x, minlength=self.counts.size)
        assert out.size == self.counts.size, f"The max of input {np.amax(x)} is not < {self.counts.size}"
        self.counts += out
        
    def quantile(self, q):        
        cs = np.cumsum(self.counts)
        
        # cs[bin_idx -1] < q*self.n <= cs[bin_idx]
        bin_idx = np.searchsorted(cs, q*cs[-1], side='left')
        
        return bin_idx
    
    def save(self, fname):
        return np.save(fname, self.counts)
        
    def load(self,fname):
        self.counts = np.load(fname)


class ValueCount:
    def __init__(self):
        self.value = 0.0
        self.count = 0

    def __repr__(self):
        return f"ValueCount(value={self.value}, count={self.count})"

    def __str__(self):
        return f"ValueCount(value={self.value}, count={self.count})"


def empty_like(x):
    if isinstance(x, dict):
        return type(x)()
    elif isinstance(x, list):
        return [None] * len(x)
    else:
        raise NotImplementedError(f"Type {type(x)} not implemented.")


def iterate(x):
    if isinstance(x, dict):
        for k, v in x.items():
            yield k, v
    else:
        for k, v in enumerate(x):
            yield k, v


def _zero_element(x, with_count=False):
    if torch.is_tensor(x):
        if not with_count:
            return 0.0
        else:
            return ValueCount()
    if isinstance(x, (list, tuple)):
        return [_zero_element(xx, with_count=with_count) for xx in x]
    if isinstance(x, dict):
        return {k: _zero_element(v, with_count=with_count) for k, v in x.items()}
    raise NotImplementedError(f"Type {type(x)} not implemented.")


def _plus(accum, x):
    if isinstance(accum, list):
        for i in range(len(accum)):
            if torch.is_tensor(accum[i]) or isinstance(accum[i], float):
                accum[i] += torch.mean(x[i])
            else:
                _plus(accum[i], x[i])
    elif isinstance(accum, dict):
        for k in accum:
            if torch.is_tensor(accum[k]) or isinstance(accum[k], float):
                accum[k] += torch.mean(x[k])
            else:
                _plus(accum[k], x[k])
    else:
        raise NotImplementedError(f"Only implemented for list, dict not for {type(accum)}.")


def _plus_named(name, accum, x, names):
    for k, v in iterate(accum):
        if isinstance(accum[k], ValueCount):
            assert torch.is_tensor(x[k]) and list(x[k].shape) == [len(names)], f"x[k]={x[k]}, names={names}"
            x_name = x[k][[n == name for n in names]]
            v.value += torch.sum(x_name).item()
            v.count += x_name.numel()
        else:
            _plus_named(name, accum[k], x[k], names)


def _normalize_named(accums):
    res = empty_like(accums)
    for k, v in iterate(accums):
        if isinstance(v, ValueCount):
            res[k] = v.value / v.count
        else:
            res[k] = _normalize_named(v)
    return res


def _increment_named(accum, inc):
    for k, v in iterate(accum):
        if isinstance(v, ValueCount):
            v.value += inc[k].value
            v.count += inc[k].count
        else:
            _increment_named(accum[k], inc[k])


def _scalar_multiplication(accum, s: float):
    if isinstance(accum, list):
        for i in range(len(accum)):
            if torch.is_tensor(accum[i]) or isinstance(accum[i], float):
                accum[i] *= s
            else:
                _scalar_multiplication(accum[i], s)
    elif isinstance(accum, dict):
        for k in accum:
            if torch.is_tensor(accum[k]) or isinstance(accum[k], float):
                accum[k] *= s
            else:
                _scalar_multiplication(accum[k], s)
    else:
        raise NotImplementedError(f"Only implemented for list, dict not for {type(accum)}.")


def epoch_end_mean(x: list):
    accum = _zero_element(x[0])

    for xx in x:
        _plus(accum, xx)

    _scalar_multiplication(accum, 1.0 / len(x))
    return accum


def epoch_end_mean_named(x: list, names: list):
    unique_names = set(item for sublist in names for item in sublist)
    accums = {name: _zero_element(x[0], with_count=True) for name in unique_names}
    accum_all = _zero_element(x[0], with_count=True)
    for name, accum in accums.items():
        for batch_names, xx in zip(names, x):
            _plus_named(name=name, accum=accum, x=xx, names=batch_names)
        _increment_named(accum=accum_all, inc=accum)

    return _normalize_named([accum_all])[0], _normalize_named(accums)
