from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch
import re

from .utils import flatten_pytree


Array = Any


class UpdateRule:
    """Abstract class for an update rule that defines an optimization algorithm.
    
    Specifically, this is for optimization algorithms that perform preconditioned
    gradient descent:
           w_{t+1} = w_t - P^{-1}_t ∇L(w_t)
    where the preconditioner P_t depends on some evolving state.
    
    This formulation encompasses gradient descent with a fixed learning rate (a trivial special case 
    with P_t = 1/η I), gradient descent with a learning rate schedule, Scalar RMSProp, and RMSProp.
    
    The functional design here is inspired by Jax's Optax library.
    """
        
    def initialize_state(self, w: Array, unflatten_w: callable) -> Array:
        """Initialize the state.
        
        Args:
          w: the initial weights
          unflatten_w (callable): function to unflatten weights
          
        Returns:
          Array: the state, as a flat vector (see subclasses for examples)
        """
        raise NotImplementedError()

    def P(self, flat_state: Array) -> Preconditioner:
        """Return the current preconditioner.
        
        Args:
          flat_state (Array): the current state, as a flat vector
        
        Returns:
          Preconditioner: the current preconditioner
        """
        raise NotImplementedError()

    def update_state(self, flat_state: Array, gradient: Array) -> Array:
        """Update the state (discrete time).
        
        Args:
          flat_state (Array): the current state, as a flat vector
          gradient (Array): the current gradient
        
        Returns:
          Array: the next state
        """
        raise NotImplementedError()

    def dstate_dt(self, flat_state: Array, gradient: Array):
        """Update the state (continuous time).
        
        Args:
          flat_state (Array): the current state, as a flat vector
          gradient (Array): the current gradient
        
        Returns:
          Array: the time derivative of the state
        """
        return self.update_state(flat_state, gradient) - flat_state

    def summarize_state(self, flate_state: Array) -> Dict[str, Any]:
        """Summarize the state.
        
        Args:
          flat_state (Array): the current state, as a flat vector
          
        Returns:
          Dict[str, Any]: a dictionary with a summary of the current state.
        """
        raise NotImplementedError()
    
    def update(self, w: Array, flat_state: Array, gradient: Array) -> Tuple[Array, Array]:
        """Update both the weights and optimizer state."""
        flat_state = self.update_state(flat_state, gradient)
        w = w - self.P(flat_state)(gradient)
        return w, flat_state


@dataclass # we make it a dataclass so that it can be instantiated from the command line
class GradientDescent(UpdateRule):
    """Gradient descent with a fixed or scheduled learning rate: 
             w_{t+1} = w_t - η(t) ∇L(w_t)
    where η(t) is the learning rate at step t.
    
    This is a special case of `UpdateRule` with P_t set to P_t = 1/η(t) I.
    
    The state consists of the current step counter t (to support learning rate schedules).
    """
    lr: float  # the learning rate

    def __post_init__(self):
        self.lr_fn = to_schedule(self.lr)

    def initialize_state(self, w: Array, unflatten_w: callable) -> Array:
        state = {
            "t": torch.tensor(0.0, dtype=w.dtype, device=w.device),
        }
        flat_state, self.unflatten = flatten_pytree(state)
        self.n = w.shape[0]
        return flat_state

    def P(self, flat_state: Array) -> Array:
        state = self.unflatten(flat_state)
        return DiagonalPreconditioner(self.lr_fn(state["t"]), self.n)

    def update_state(self, flat_state: Array, gradient: Array) -> Array:
        state = self.unflatten(flat_state)
        state = {"t": state["t"] + 1.0}
        return flatten_pytree(state)[0]

    def summarize_state(self, flat_state: Array) -> Array:
        state = self.unflatten(flat_state)
        return {
            "t": state["t"],
            "lr": self.lr_fn(state["t"]),
        }

    def raw_eigs_from_eigs(self, flat_state: Array, eigs: Array):
        """Given the top eigenvalues of the effective Hessian, return the top 
        eigenvalues of the 'raw' Hessian.
        
        This function is used by the 'raw' Hessian eigenvalue logger.
        
        Args:
          flate_state (Array): the current state, as a flattened vector
          eigs (Array): the top eigenvalues of the effective Hessian
          
        Returns:
          (Array): the top eigenvalues of the 'raw' Hessian
        """
        if eigs is None:
            return None
        lr = self.P(flat_state)(1.0)
        return eigs / lr


@dataclass # we make it a dataclass so that it can be instantiated from the command line
class ScalarRMSProp(UpdateRule):
    """The Scalar RMSProp optimizer.
    
    This optimizer maintains an EMA ν of the squared gradient norm,
    and takes gradient steps using the effective step size η/sqrt(ν).
    Our implementation supports optional learning rate scheduling,
    bias correction, and ε:
    
           ν_{t} = (1 - β_2) ν_{t-1} + β_2 ||∇L(w_t)||^2
           ν̂_{t} = ν_t / (1 - β_2 ^ t)
           w_{t+1} = w_t - η(t) / sqrt (ν̂_t + ε) * ∇L(w_t)
           
    The optimizer's state consists of the tuple (t, ν).
    """
    
    lr: float
    beta2: float
    eps: float = 0.
    bias_correction: bool = False

    def __post_init__(self):
        self.lr_fn = to_schedule(self.lr)

    def initialize_state(self, w: Array, unflatten_w: callable) -> Array:
        state = {
            "t": torch.tensor(0.0, dtype=w.dtype, device=w.device),
            "nu": torch.tensor(0.0, dtype=w.dtype, device=w.device),
        }
        flat_state, self.unflatten = flatten_pytree(state)
        self.n = w.shape[0]
        return flat_state

    def P(self, flat_state: Array) -> Array:
        state = self.unflatten(flat_state)
        t, nu = state["t"], state["nu"]
        if self.bias_correction:
            nu_hat =  (nu / (1 - self.beta2**(t)))
        else:
            nu_hat = nu
        lrs = self.lr_fn(t) / (torch.sqrt(nu_hat) + self.eps)
        return DiagonalPreconditioner(lrs, self.n)

    def update_state(self, flat_state: Array, gradient: Array) -> Array:
        state = self.unflatten(flat_state)
        t, nu = state["t"], state["nu"]
        nu = nu + (1 - self.beta2) * (gradient.square().sum() - nu)
        state = {"t": t + 1.0, "nu": nu}
        return flatten_pytree(state)[0]

    def dstate_dt(self, flat_state: Array, gradient: Array) -> Array:
        update = self.update_state(flat_state, gradient) - flat_state
        return update / self.beta2  # see footnote TODO in paper for explanation of this

    def summarize_state(self, flat_state: Array) -> Array:
        state = self.unflatten(flat_state)
        ess = self.P(flat_state).pow(-1)(1.0)
        return {
            "t": state["t"],
            "nu": state["nu"],
            "lr": self.lr_fn(state["t"]),
            "effective_step_size": ess,
        }

    def raw_eigs_from_eigs(self, flat_state: Array, eigs: Array):
        """Given the top eigenvalues of the effective Hessian, return the top 
        eigenvalues of the 'raw' Hessian.
        
        This function is used by the 'raw' Hessian eigenvalue logger.
        
        Args:
          flate_state (Array): the current state, as a flattened vector
          eigs (Array): the top eigenvalues of the effective Hessian
          
        Returns:
          (Array): the top eigenvalues of the Hessian
        """
        if eigs is None:
            return None
        ess = self.P(flat_state).pow(-1)(1.0) # effective step size
        return eigs / ess


@dataclass # we make it a dataclass so that it can be instantiated from the command line
class RMSProp(UpdateRule):
    """The RMSProp optimizer.
    
    This optimizer maintains an EMA ν of the elementwise squared gradient, 
    and takes gradient steps using the effective step sizes η/sqrt(ν).
    Our implementation supports optional learning rate scheduling,
    bias correction, and ε:
    
           ν_{t} = (1 - β_2) ν_{t-1} + β_2 ∇L(w_t)^2
           ν̂_{t} = ν_t / (1 - β_2 ^ t)
           w_{t+1} = w_t - η(t) / sqrt (ν̂_t + ε) * ∇L(w_t)
           
    The optimizer's state consists of the tuple (t, ν).
    """
    
    lr: float
    beta2: float
    eps: float = 0
    bias_correction: bool = False

    def __post_init__(self):
        self.lr_fn = to_schedule(self.lr)

    def initialize_state(self, w: Array, unflatten_w: callable) -> Array:
        state = {
            "t": torch.tensor(0.0, dtype=w.dtype, device=w.device),
            "nu": torch.zeros_like(w),
        }
        flat_state, self.unflatten = flatten_pytree(state)
        return flat_state

    def P(self, flat_state: Array) -> Array:
        state = self.unflatten(flat_state)
        t, nu = state["t"], state["nu"]
        if self.bias_correction:
            nu_hat =  (nu / (1 - self.beta2**(t)))
        else:
            nu_hat = nu
        lrs = self.lr_fn(t) / (torch.sqrt(nu_hat) + self.eps)
        return DiagonalPreconditioner(lrs, nu.shape[0])

    def update_state(self, flat_state: Array, gradient: Array) -> Array:
        state = self.unflatten(flat_state)
        t, nu = state["t"], state["nu"]
        nu = nu + (1 - self.beta2) * (gradient.square() - nu)
        state = {"t": t + 1.0, "nu": nu}
        return flatten_pytree(state)[0]

    def dstate_dt(self, flat_state: Array, gradient: Array) -> Array:
        update = self.update_state(flat_state, gradient) - flat_state
        return update / self.beta2 # see footnote TODO in paper for explanation of this

    def summarize_state(self, flat_state: Array) -> Array:
        state = self.unflatten(flat_state)
        nu = state["nu"]
        ess = self.P(flat_state).pow(-1)(torch.ones_like(nu))
        selected_idx = np.linspace(0, len(nu) - 1, 25, dtype=int)
        return {
            "t": state["t"],
            "nu_l1": nu.sum(),                   # L1 norm of nu
            "nu_selected_idx": nu[selected_idx], # selected indices of nu
            "ess_mean": ess.mean(),              # mean of effective step sizes
                                                 # harmonic mean of effective step sizes
            "ess_harmonic_mean": ess.reciprocal().mean().reciprocal(),
            "lr": self.lr_fn(state["t"]), # current learning rate
        }


def preconditioner_ns(G: Array, steps: int) -> Tuple[Array, Array]:
    """
    Performs the Newton-Schulz iteration to compute the Muon update and the explicit
    preconditioner matrix.

    DESIGN CHOICE: "Parallel Identity" Algorithm
    --------------------------------------------
    Standard Muon only computes the update X_K = P^{-1} G_0. However, to analyze eigenvalues
    of P^{-1}H, we need the explicit operator P^{-1}.
    This function runs a parallel iteration on an accumulator matrix A (initialized to Identity)
    applying the same transformations B(X_k) to A that are applied to X.
    By linearity, if X_{k+1} = B(X_k) X_k, then A_{final} satisfies X_{final} = A_{final} G_0.
    Thus, A_final is the explicit matrix form of P^{-1}.

    Mathematical Notation (from Proposal):
    - Input G is the gradient block G_0 (or momentum).
    - X tracks the iterates X_k, converging to the orthogonalized update.
    - A tracks the preconditioner P_{Muon}^{-1}(G_0).
    - The polynomial update is B(X) = alpha*I + beta*XX^T + gamma*(XX^T)^2.

    Args:
        G (Tensor): The input matrix G_0 (usually momentum).
        steps (int): Number of Newton-Schulz steps (usually 5).

    Returns:
        X (Tensor): The final update step X_K approx P^{-1} G_0.
        A (Tensor): The explicit left-preconditioner matrix P_{Muon}^{-1}(G_0).
    """
    assert G.ndim == 2
    m, n = G.shape

    # 1. Normalization
    # We define the scaling factor c = ||G_0||_F + epsilon.
    # To ensure stability, inputs to Newton-Schulz must have spectral norm < 1.
    norm = G.norm(dim=(-2, -1), keepdim=True) + 1e-7

    # Initialize iterates.
    # We delay the final Muon scaling factor (sqrt(m/n)) until the end to avoid
    # exploding values during the iteration if m >> n.
    # X_0 = (1/c) * G_0
    X = G / norm

    # Initialize the accumulator A.
    # Since X_0 = (1/c) * I * G_0, the operator starts as A_0 = (1/c) * I.
    A = torch.eye(m, dtype=G.dtype, device=G.device) / norm

    # Quintic coefficients for the polynomial B(X).
    # These correspond to alpha, beta, gamma in the proposal's update rule.
    a, b, c = (3.4445, -4.7750, 2.0315)

    # 2. Iteration
    for _ in range(steps):
        # Compute X_k * X_k^T
        XXt = X @ X.mT

        # Compute the polynomial B(X_k) = alpha*I + beta*XX^T + gamma*(XX^T)^2
        # This operator is symmetric and depends only on the singular values of X_k.
        B = a * torch.eye(m, device=G.device, dtype=G.dtype) + b * XXt + c * (XXt @ XXt)

        # Update X: X_{k+1} = B(X_k) @ X_k
        X = B @ X

        # Update A: A_{k+1} = B(X_k) @ A_k
        # We apply the exact same left-operator B(X_k) to A.
        # This preserves the invariant X_k = A_k @ G_0.
        A = B @ A

    # 3. Muon Scaling
    # The proposal defines the final update as scaled by sqrt(max(1, m/n)).
    # We apply this scalar factor to both the update X and the operator A
    # so that X = A @ G_0 still holds.
    if m > n:
        scale = (m / n) ** 0.5
        X = X * scale
        A = A * scale
        # TODO: apply transpose in this case (see Muon original repo)
        raise NotImplementedError("Muon preconditioner not implemented for m < n.")

    return X, A


@dataclass
class Muon(UpdateRule):
    """TODO"""

    lr: float = 0.02
    ns_steps: int = 5

    def initialize_state(self, w: Array, unflatten_w: callable) -> Array:
        self.unflatten_w = unflatten_w
        matrix_w = unflatten_w(w)
        assert matrix_w.ndim >= 2

        state = {
            "t": torch.tensor(0.0, dtype=w.dtype, device=w.device),
            "gradient": torch.zeros_like(matrix_w),
        }
        flat_state, self.unflatten = flatten_pytree(state)
        return flat_state

    def P(self, flat_state: Array) -> Array:
        state = self.unflatten(flat_state)
        gradient = state["gradient"]
        _, p = preconditioner_ns(gradient, steps=self.ns_steps)
        p.mul_(self.lr)
        n = max(gradient.size(-2), gradient.size(-1))
        return SingleBlockDiagonalPreconditioner(GenericPreconditioner(p), n)

    def update_state(self, flat_state: Array, gradient: Array) -> Array:
        state = self.unflatten(flat_state)
        gradient = self.unflatten_w(gradient)

        if gradient.ndim == 4:  # for the case of conv filters
            raise NotImplementedError(
                "Muon optimizer does not support dim > 2 yet."
            )
            gradient = gradient.view(len(gradient), -1)

        state = {"t": state["t"] + 1.0, "gradient": gradient}
        return flatten_pytree(state)[0]

    def summarize_state(self, flat_state: Array) -> Array:
        # TODO
        return {}


@dataclass
class OptimizerSelector:
    """Generic class for selecting optimizers for different layers."""

    def get_optimizer(self, layer_name: str):
        """Return the appropriate optimizer for the given layer."""
        raise NotImplementedError()


@dataclass
class RegexOptimizerSelector(OptimizerSelector):
    """Select optimizers based on whether the layer name matches a regex pattern."""

    matching_factory: callable[UpdateRule]
    non_matching_factory: callable[UpdateRule]
    pattern: str

    def get_optimizer(self, layer_name: str) -> UpdateRule:
        if re.search(self.pattern, layer_name):
            return self.matching_factory()
        return self.non_matching_factory()


@dataclass
class CompositeUpdateRule(UpdateRule):
    """An update rule that applies different optimizers to different parameter groups."""

    lr: float = 0.01

    @dataclass
    class UpdateRuleGroup:
        """Utility class for grouping an optimizer with the parameter span it applies to."""

        layer_name: str
        span: Tuple[int, int]
        optimizer: UpdateRule

    def __post_init__(self):
        self.groups = []
        self.selector: OptimizerSelector = RegexOptimizerSelector(
            matching_factory=lambda: None, # TODO: bias layers are disabled for now
            non_matching_factory=lambda: Muon(lr=self.lr, ns_steps=5),
            pattern=".*bias",
        )

    def initialize_state(self, w: Array, unflatten_w: callable) -> Array:
        indices = torch.arange(w.shape[0])
        tree_w = unflatten_w(w)
        tree_ix = unflatten_w(torch.tensor(indices))

        for layer_name in tree_w.keys():
            optimizer = self.selector.get_optimizer(layer_name)
            indices = tree_ix[layer_name]
            span = (indices.min().item(), indices.max().item() + 1)
            self.groups.append(
                CompositeUpdateRule.UpdateRuleGroup(layer_name, span, optimizer)
            )

        self.groups.sort(key=lambda g: g.span[0])

        state = {"t": torch.tensor(0.0, dtype=w.dtype, device=w.device)}
        for group in self.groups:
            w_slice = w[group.span[0] : group.span[1]]
            _, unflatten_w_slice = flatten_pytree(tree_w[group.layer_name])
            state[group.layer_name] = group.optimizer.initialize_state(w_slice, unflatten_w_slice)

        flat_state, self.unflatten = flatten_pytree(state)
        return flat_state

    def P(self, flat_state: Array) -> Array:
        state = self.unflatten(flat_state)
        return BlockDiagonalPreconditioner(
            [group.optimizer.P(state[group.layer_name]) for group in self.groups]
        )

    def update_state(self, flat_state: Array, gradient: Array) -> Array:
        state = self.unflatten(flat_state)
        state["t"] += 1.0
        for group in self.groups:
            grad_slice = gradient[group.span[0] : group.span[1]]
            state[group.layer_name] = group.optimizer.update_state(
                state[group.layer_name], grad_slice
            )
        return flatten_pytree(state)[0]

    def summarize_state(self, flat_state: Array) -> Array:
        state = self.unflatten(flat_state)
        return {"t": state["t"]}


class Preconditioner:
    """Abstract class for a preconditioner."""
    
    def __call__(self, v: Array) -> Array:
        """Precondition a vector.
        
        Args:
          v: the vector to precondition
          
        Returns:
          the preconditioned vector Pv
        """
        raise NotImplementedError()
    
    def sqrt(self) -> Preconditioner:
        """Return a new preconditioner which is the square root of this preconditioner.
        
        Returns:
          (Preconditioner): a new preconditioner
        """
        raise NotImplementedError()

    def size(self) -> int:
        """Return the size of the preconditioner."""
        raise NotImplementedError()


class GenericPreconditioner(Preconditioner):
    """A generic preconditioner represented as a matrix."""

    def __init__(self, P: Array):
        """Constructor for the generic preconditioner.

        Args:
          P (Array): the preconditioner, as a matrix
        """
        if P.ndim != 2 or P.shape[0] != P.shape[1]:
            raise ValueError("Preconditioner must be a square matrix.")

        self.P = P

    def __call__(self, v: Array) -> Array:
        return self.P @ v

    def sqrt(self) -> GenericPreconditioner:
        # Compute sqrt with eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(self.P)
        eigenvalues = torch.clamp(eigenvalues, 0)
        root_eigenvalues = torch.sqrt(eigenvalues)
        sqrt = eigenvectors @ (root_eigenvalues[:, None] * eigenvectors.T)
        return GenericPreconditioner(sqrt)

    def size(self) -> int:
        return self.P.shape[0]


class DiagonalPreconditioner(Preconditioner):
    """A diagonal (i.e. elementwise) preconditioner."""
    
    def __init__(self, P, n: int):
        """Constructor for the diagonal preconditioner.
        Note: the size is a required argument since P may be a 
        scalar, in which case its size cannot be inferred.
        
        Args:
          P (Array): the diagonal preconditioner, as a vector
          n (int): the size of the preconditioner
        """
        if isinstance(P, torch.Tensor) and P.isinf().any():
            raise ValueError("Preconditioner contains infinite values.")
        self.n = n
        self.P = P

    def __call__(self, v: Array) -> Array:
        return v * self.P

    def sqrt(self) -> DiagonalPreconditioner:
        return DiagonalPreconditioner(self.P**0.5, self.n)

    def size(self) -> int:
        return self.n


class BlockDiagonalPreconditioner(Preconditioner):
    """A block-diagonal preconditioner."""

    def __init__(self, blocks: list[Preconditioner]):
        """Constructor for the block diagonal preconditioner.

        Args:
          blocks (list[Preconditioner]): the diagonal blocks of the preconditioner
        """
        self.blocks = blocks

    def __call__(self, v: Array) -> Array:
        offset = 0
        result = torch.zeros_like(v)
        for block in self.blocks:
            block_size = block.size()
            result[offset : offset + block_size] = block(
                v[offset : offset + block_size]
            )
            offset += block_size
        return result

    def sqrt(self) -> BlockDiagonalPreconditioner:
        new_blocks = [block.sqrt() for block in self.blocks]
        return BlockDiagonalPreconditioner(new_blocks)

    def size(self) -> int:
        return sum(block.size() for block in self.blocks)


class SingleBlockDiagonalPreconditioner(Preconditioner):
    def __init__(self, block: Preconditioner, n: int):
        """
        Initialize a block diagonal preconditioner with a 
        single block repeated along the diagonal.
        
        Args:
          block (Preconditioner): the block to repeat
          n (int): the number of times to repeat the block
        """
        self.block = block
        self.n = n
    
    def __call__(self, v: Array) -> Array:
        result = torch.zeros_like(v)
        for i in range(self.n):
            start = i * self.block.size()
            end = start + self.block.size()
            result[start:end] = self.block(v[start:end])
        return result

    def sqrt(self) -> SingleBlockDiagonalPreconditioner:
        return SingleBlockDiagonalPreconditioner(self.block.sqrt(), self.n)
    
    def size(self) -> int:
        return self.n * self.block.size()


def to_schedule(schedule_or_constant):
    """Optionally create an LR schedule from a constant LR."""
    if callable(schedule_or_constant):          # if it's a schedule ...
        return schedule_or_constant             #  ... do nothing.
    else:                                       # but if it's a constant...
        return lambda t: schedule_or_constant   # ... turn it into a schedule. 
