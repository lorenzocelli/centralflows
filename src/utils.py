import time
from dataclasses import asdict, is_dataclass, fields, replace
from pathlib import Path
from typing import Any, Dict, Union, Callable, TypeVar

import numpy as np
import torch
from torch.utils._pytree import tree_flatten, tree_unflatten
import math

Array = Any

def D(f: Callable, w: torch.Tensor, order: int =1, *vs: torch.Tensor) -> torch.Tensor:
    """Function for computing arbitrary higher-order derivatives.
    
    Contracts the `order`-order derivative tensor of f at w, with the vectors `vs`.
    See examples below.
        
    Args:
      f: a scalar-valued function to differentiate
      w: the point to differentiate it at
      order: the order of derivative to take
      *vs: optional directional vectors for computing directional derivatives.
           The number of vectors must be less than or equal to order.
           When provided, computes directional derivatives in the direction
           of these vectors instead of full derivatives.
           
    Returns:
      derivatives of f, as a tensor of rank: order - len(vs) 
                   
    Examples:
      - D(f, w, 0) returns f(w), a scalar
      - D(f, w, 1) returns ∇f(w), a vector
      - D(f, w, 2, u, v) returns u' H(w) v, a scalar
      - D(f, w, 2, u) returns H(w) u, a vector
      - D(f, w, 3, u, v) returns ∇_w [u' H(w) v], a vector
    """
    assert len(vs) <= order
    if order == 0:
        return f(w)
    elif len(vs) == order:
        v, *vs = vs
        df = lambda p: torch.func.jvp(f, (p,), (v,))[1].to(v.dtype)
    else:
        df = lambda p: torch.func.jacrev(f)(p)
    return D(df, w, order - 1, *vs)


def save_pytree(pytree, path):
    """Save a pytree to disk."""
    flattened, spec = tree_flatten(pytree)
    torch.save({'flat': flattened, 'spec': spec}, path)

def load_pytree(path, map_location=None):
    """Load a pytree from disk."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    return tree_unflatten(checkpoint['flat'], checkpoint['spec'])

def flatten_pytree(pytree):
    """Flatten a pytree of tensors into one tensor.
    
    Args:
      pytree: a pytree (e.g. a hierarchical dict) of tensors
    
    Returns:
      Array: a single flat tensor that contains that concatenation
        of all the pytree leaf tensors
      Function: an unflattening function that turns such flat tensors
        back into pytrees
    
    """
    listp, treedef = tree_flatten(pytree)
    flatp = torch.concatenate([p.ravel() for p in listp])
    sizes, shapes = zip(*[(x.numel(), x.shape) for x in listp])

    def unflatten(flatp):
        chunks = flatp.split(sizes)
        listp = [chunk.reshape(shape) for chunk, shape in zip(chunks, shapes)]
        return tree_unflatten(listp, treedef)

    return flatp, unflatten


def to_numpy(x):
    """Convert to a numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif x is None:
        return x
    else:
        return np.array(x)

def flatten_dict(
    data: Dict[str, Any], parent_key: str = "", sep: str = "."
) -> Dict[str, Any]:
    """Flattens a nested dict into a flat dict."""
    items = {}
    for key, v in data.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep))
        else:
            items[new_key] = v

    return items


def unflatten_dict(d: Dict[str, Any], sep: str = ".") -> Dict[str, Any]:
    """Unflattens a flat dict into a nested dict."""
    result = {}
    for key, value in d.items():
        *path, last = key.split(sep)
        target = result
        for part in path:
            target = target.setdefault(part, {})
        target[last] = value
    return result


def convert_dataclasses(data: Any) -> Union[Dict, Any]:
    """Converts dataclasses to standard python objects so it can be serialized as json."""
    if isinstance(data, Path):
        return str(data)
    elif isinstance(data, set):
        return list(sorted(data))
    elif is_dataclass(data):
        result = asdict(data)
        result["class"] = data.__class__.__name__
        return {key: convert_dataclasses(value) for key, value in result.items()}
    elif isinstance(data, dict):
        return {key: convert_dataclasses(value) for key, value in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(convert_dataclasses(item) for item in data)
    return data

DEFAULT_TYPE = TypeVar("D")
OVERRIDE_TYPE = TypeVar("O")

def apply_overrides(default: DEFAULT_TYPE, override: OVERRIDE_TYPE) -> DEFAULT_TYPE:
    """
    Return a new dataclass of the same type as `default`,
    where any field on `override` that is not None
    replaces the corresponding field on `default`.
    
    Both `default` and `override` must be dataclass instances.
    Fields present in default but missing on override are ignored.
    """
    if not is_dataclass(default) or not is_dataclass(override):
        raise TypeError("Both default and override must be dataclass instances.")
    overrides = {}
    for f in fields(default):
        # only consider fields that both share
        if hasattr(override, f.name):
            val = getattr(override, f.name)
            if val is not None:
                overrides[f.name] = val
    return replace(default, **overrides)

def apply_to_pairs(f, U, vectorized=False):
    """
    Apply a function f to each pair of set of vectors.
    
    Args:
        f: Symmetric bilinear function f(a,b) = f(b,a)
        U: Tensor where last dimension contains basis vectors U_i

    Returns:
        torch.Tensor: Flattened upper triangular part of the matrix M[i,j] = f(U_i, U_j), following triu_indices ordering
    """
    k = U.shape[-1]
    i, j = torch.triu_indices(k, k)
    if vectorized:
        return torch.func.vmap(f, -1)(U[..., i], U[..., j])
    else:
        return torch.stack([f(U[..., i], U[..., j]) for i, j in zip(i, j)])


def mat_to_upper(x):
    """
    Converts a square matrix to a vector containing its upper triangular entries.

    Args:
        x (torch.Tensor): A k x k matrix

    Returns:
        torch.Tensor: Vector of length k(k+1)/2 containing upper triangular entries
    """
    k = x.shape[0]
    i, j = torch.triu_indices(k, k)
    return x[i, j]


def upper_to_mat(x):
    """
    Converts a vector/matrix of upper triangular entries back to symmetric matrix/tensor.

    Args:
        x (torch.Tensor): Either:
            - Vector of length k(k+1)/2 containing upper triangular entries
            - Matrix of size k(k+1)/2  x  k(k+1)/2 containing flattened tensor entries

    Returns:
        torch.Tensor: Either:
            - k x k symmetric matrix if input is 1D
            - k x k x k x k tensor if input is 2D
    """
    k = int(math.sqrt(2 * len(x)))
    i, j = torch.triu_indices(k, k)
    s = range(len(i))
    T = torch.zeros((k, k, k * (k + 1) // 2), dtype=x.dtype, device=x.device)
    T[i, j, s] = T[j, i, s] = 1
    T = T.reshape(k**2, -1)
    if x.ndim == 1:
        return (T @ x).reshape(k, k)
    elif x.ndim == 2:
        return (T @ x @ T.T).reshape(k, k, k, k)

def get_upper_to_mat_metric(k: int) -> Array:
    """
    Returns a metric that is used to contract the upper triangular entries of a matrix.
    The metric puts a weight of 2 on the off-diagonal entries and 1 on the diagonal entries.
    """
    metric = (
        2 * torch.ones((k, k)) - torch.eye(k)
    ).cuda()  # metric[i,j] = 2 if i != j else 1
    return mat_to_upper(metric)  # convert to upper triangular form

def tree_flatten_with_names(pytree, prefix=""):
    """
    Flatten a pytree into an ordered dict {path_string: tensor}.
    
    Args:
        pytree: any pytree structure (dict, list, tuple, tensor)
        prefix: path prefix used internally during recursion

    Returns:
        flat: OrderedDict mapping unique names -> tensors
        treedef: same treedef returned by tree_flatten (optional to save)
    """

    from collections import OrderedDict

    flat = OrderedDict()

    if torch.is_tensor(pytree):
        # Leaf tensor
        flat[prefix.rstrip("/")] = pytree
        return flat

    elif isinstance(pytree, dict):
        for key, subtree in pytree.items():
            name = f"{prefix}{key}/"
            flat.update(tree_flatten_with_names(subtree, name))
        return flat

    elif isinstance(pytree, (list, tuple)):
        for idx, subtree in enumerate(pytree):
            name = f"{prefix}{idx}/"
            flat.update(tree_flatten_with_names(subtree, name))
        return flat

    else:
        raise TypeError(f"Unsupported pytree leaf type: {type(pytree)}")


class Timer:
    def __init__(self):
        self.times = {}

    def __call__(self, name):
        return TimingContext(name, self.times)


class TimingContext:
    def __init__(self, name, times_dict):
        self.name = name
        self.times_dict = times_dict

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.times_dict[self.name] = time.time() - self.start
