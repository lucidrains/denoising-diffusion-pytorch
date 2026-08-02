from inspect import signature

import torch
from torch.nn import Module
from torch import cat, is_tensor
from torch.utils._pytree import tree_map, tree_flatten
from einops import repeat, reduce, rearrange

# explorative modeling (forward xm)
# Alexi Gladstone et al. https://arxiv.org/abs/2607.27372

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def is_batch_tensor(t, b):
    if not is_tensor(t) or t.ndim == 0:
        return False

    batch = t.shape[0]
    return batch == b

# xm wrapper

class XMWrapper(Module):
    def __init__(
        self,
        flow_model: Module,
        candidates = 1,
        max_batch_size = None,
        random_time_method = 'random_times',
        random_time_kwarg = 'times'
    ):
        super().__init__()
        self.flow_model = flow_model

        assert candidates >= 1, 'candidates must be at least 1'
        self.candidates = candidates
        self.max_batch_size = max_batch_size
        self.has_loss_reduction = 'loss_reduction' in signature(flow_model.forward).parameters

        self.random_time_method = random_time_method
        self.random_time_kwarg = random_time_kwarg

    @property
    def data_shape(self):
        return getattr(self.flow_model, 'data_shape', None)

    @data_shape.setter
    def data_shape(self, val):
        if hasattr(self.flow_model, 'data_shape'):
            self.flow_model.data_shape = val

    def sample(self, *args, **kwargs):
        return self.flow_model.sample(*args, **kwargs)

    def forward(
        self,
        *args,
        candidates = None,
        max_batch_size = None,
        **kwargs
    ):
        candidates = default(candidates, self.candidates)
        max_batch_size = default(max_batch_size, self.max_batch_size)

        if candidates == 1:
            return self.flow_model(*args, **kwargs)

        # find batch size from first tensor in inputs

        leaves, _ = tree_flatten((args, kwargs))
        first_tensor = next(t for t in leaves if is_tensor(t))
        batch = first_tensor.shape[0]

        if self.random_time_kwarg not in kwargs:
            assert hasattr(self.flow_model, self.random_time_method), f'flow_model must have a {self.random_time_method} method'
            fn = getattr(self.flow_model, self.random_time_method)
            kwargs[self.random_time_kwarg] = fn(batch)

        # repeat inputs K candidates times

        args_K, kwargs_K = tree_map(
            lambda t: repeat(t, 'b ... -> (b k) ...', k = candidates) if is_batch_tensor(t, batch) else t,
            (args, kwargs)
        )

        total = batch * candidates
        chunk_size = default(max_batch_size, total) if self.has_loss_reduction else 1
        extra_kwargs = {'loss_reduction': 'none'} if self.has_loss_reduction else dict()

        # forward passes

        losses = []

        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)

            chunk_args, chunk_kwargs = tree_map(
                lambda t: t[start:end] if is_batch_tensor(t, total) else t,
                (args_K, kwargs_K)
            )
            chunk_kwargs.update(extra_kwargs)

            loss = self.flow_model(*chunk_args, **chunk_kwargs)
            loss = rearrange(loss, '-> 1 1') if loss.ndim == 0 else rearrange(loss, 'b ... -> b (...)')
            losses.append(loss)

        # select candidate with minimum loss for each sample in batch

        raw_loss = cat(losses, dim = 0)
        candidate_losses = reduce(raw_loss, '(b k) ... -> b k', 'mean', b = batch, k = candidates)

        return candidate_losses.amin(dim = -1).mean()
