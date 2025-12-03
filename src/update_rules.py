from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, List

import numpy as np
import torch

from .utils import flatten_pytree


Array = Any


def _unflatten_helper(flat_tensor: torch.Tensor, shapes: List[torch.Size]) -> List[torch.Tensor]:
    """Unflatten a flat tensor into a list of tensors with given shapes.

    Args:
        flat_tensor: Flat 1D tensor
        shapes: List of shapes for each tensor

    Returns:
        List of tensors with the specified shapes
    """
    tensors = []
    cursor = 0
    for shape in shapes:
        size = np.prod(shape)
        tensors.append(flat_tensor[cursor:cursor+size].reshape(shape))
        cursor += size
    return tensors


def newton_schulz(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Compute the inverse square root of G^T G using Newton-Schulz iteration.

    This computes M = (G^T G)^{-1/2} using the Newton-Schulz algorithm.

    Args:
        G: Input matrix (can be non-square)
        steps: Number of Newton-Schulz iterations
        eps: Small constant for numerical stability

    Returns:
        M: The matrix (G^T G)^{-1/2}
    """
    # Compute G^T G
    a = G.T @ G

    # Normalize by the trace to improve conditioning
    # This helps the iteration converge to the correct solution
    dim = a.shape[0]
    normalization = a.trace() / dim + eps
    a = a / normalization

    # Initialize: I is the starting point
    # We're computing the inverse sqrt, so we start with identity
    Y = torch.eye(dim, dtype=a.dtype, device=a.device)

    # Newton-Schulz iteration: Y_{k+1} = Y_k * (3I - a @ Y_k^2) / 2
    # This converges to a^{-1/2}
    I = torch.eye(dim, dtype=a.dtype, device=a.device)
    for _ in range(steps):
        Y = Y @ (3 * I - a @ Y @ Y) / 2

    # Undo the normalization
    return Y / torch.sqrt(normalization)


def matrix_power(M: torch.Tensor, p: float) -> torch.Tensor:
    """Compute M^p for a matrix M using SVD.

    Args:
        M: Square matrix
        p: Power (can be fractional)

    Returns:
        M^p
    """
    # Use SVD: M = U S V^T
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)

    # Compute S^p (element-wise power on singular values)
    S_p = S ** p

    # Reconstruct: M^p = U S^p V^T
    return U @ torch.diag(S_p) @ Vh


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
        
    def initialize_state(self, w: torch.Tensor, unflatten: Any = None) -> Array:
        """Initialize the state.
        
        Args:
          w: the initial weights
          
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
        w = w - self.P(flat_state).pow(-1)(gradient)
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

    def initialize_state(self, w: Array, unflatten: Any = None) -> Array:
        state = {"t": torch.tensor(0.0, dtype=w.dtype, device=w.device)}
        flat_state, self.unflatten = flatten_pytree(state)
        return flat_state

    def P(self, flat_state: Array) -> Array:
        state = self.unflatten(flat_state)
        return DiagonalPreconditioner(1 / self.lr_fn(state["t"]))

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
        lr = self.P(flat_state).pow(-1)(1.0)
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

    def initialize_state(self, w: Array, unflatten: Any = None) -> Array:
        state = {
            "t": torch.tensor(0.0, dtype=w.dtype, device=w.device),
            "nu": torch.tensor(0.0, dtype=w.dtype, device=w.device),
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
        return DiagonalPreconditioner(1 / lrs)

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

    def initialize_state(self, w: Array, unflatten: Any = None) -> Array:
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
        return DiagonalPreconditioner(1 / lrs)

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

@dataclass
class Muon(UpdateRule):
    """
    The Muon optimizer.

    Muon uses a hybrid approach:
    - 2D+ parameters: Momentum + Muon preconditioner (Newton-Schulz orthogonalization)
    - 1D parameters: Plain SGD with momentum (Identity preconditioner)

    This matches the original Muon paper's approach.
    """
    lr: float
    momentum: float = 0.95
    ns_steps: int = 5  # Newton-Schulz iteration steps
    shapes: List[torch.Size] = None  # Will be set during initialization

    def __post_init__(self):
        self.lr_fn = to_schedule(self.lr)

    def initialize_state(self, w: Array, unflatten: Any = None) -> Array:
        """Initialize Muon optimizer state.

        Args:
            w: Initial weights (flat)
            unflatten: Function to unflatten weights (required to extract shapes)

        Returns:
            Flat state vector
        """
        if unflatten is None:
            raise ValueError("unflatten function must be provided to initialize Muon optimizer state.")

        # Extract shapes from the unflattened parameters
        params = unflatten(w)
        self.shapes = [p.shape for p in params.values()]

        # Initialize state: momentum for all parameters, step counter
        state = {
            "momentum": torch.zeros_like(w),
            "step": torch.tensor(0.0, dtype=w.dtype, device=w.device)
        }
        flat_state, self.unflatten = flatten_pytree(state)
        return flat_state

    def P(self, flat_state: Array) -> Preconditioner:
        """Build the hybrid preconditioner.

        Args:
            flat_state: Current optimizer state (flat)

        Returns:
            HybridPreconditioner with Muon blocks for 2D params, Identity for 1D
        """
        # 1. Unpack state
        state = self.unflatten(flat_state)
        momentum = state["momentum"]

        # 2. Unflatten momentum to access per-layer gradients
        mom_tensors = _unflatten_helper(momentum, self.shapes)

        # 3. Build the Mixed Preconditioner list
        preconditioner_blocks = []

        for m in mom_tensors:
            if m.ndim >= 2:
                # --- CASE A: MUON LAYER ---
                # Calculate M = (G^T G)^{-1/2}
                # We use Newton-Schulz on the momentum matrix
                M = newton_schulz(m, steps=self.ns_steps)
                preconditioner_blocks.append(M)
            else:
                # --- CASE B: AUX LAYER (SGD) ---
                # For SGD, the preconditioner is Identity (1.0).
                preconditioner_blocks.append(1.0)

        return HybridPreconditioner(preconditioner_blocks, self.shapes)

    def update_state(self, flat_state: Array, gradient: Array) -> Array:
        """Update momentum state.

        Standard momentum update for ALL parameters (both 1D and 2D).

        Args:
            flat_state: Current state
            gradient: Current gradient

        Returns:
            Updated state
        """
        state = self.unflatten(flat_state)
        new_mom = state["momentum"] * self.momentum + gradient

        new_state = {"momentum": new_mom, "step": state["step"] + 1}
        return flatten_pytree(new_state)[0]

    def summarize_state(self, flat_state: Array) -> Dict[str, Any]:
        """Summarize current optimizer state.

        Args:
            flat_state: Current state

        Returns:
            Dictionary with state summary
        """
        state = self.unflatten(flat_state)
        return {
            "step": state["step"],
            "lr": self.lr_fn(state["step"]),
            "momentum_norm": state["momentum"].norm().item(),
        }

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
    
    def pow(self, p: float) -> Preconditioner:
        """Return a new preconditioner which is this preconditioner raised to a power.
        
        Args:
          p: the power
        
        Returns:
          (Preconditioner): a new preconditioner
        """
        raise NotImplementedError()


class DiagonalPreconditioner(Preconditioner):
    """A diagonal (i.e. elementwise) preconditioner."""

    def __init__(self, P):
        """Constructor for the diagonal preconditioner.

        Args:
          P (Array): the diagonal preconditioner, as a vector
        """
        self.P = P

    def __call__(self, v: Array) -> Array:
        return v * self.P

    def pow(self, power: float) -> DiagonalPreconditioner:
        return DiagonalPreconditioner(self.P**power)


@dataclass
class HybridPreconditioner(Preconditioner):
    """
    Applies Muon logic to 2D blocks and Identity/SGD logic to 1D blocks.

    This preconditioner handles mixed parameter types:
    - 2D+ parameters: Uses Muon preconditioner (matrix M from Newton-Schulz)
    - 1D parameters: Uses scalar preconditioner (typically 1.0 for SGD)
    """
    blocks: List[Any]  # List of M matrices or Scalars
    shapes: List[torch.Size]
    internal_power: float = 1.0

    def __call__(self, v: Array) -> Array:
        """Apply P to vector v.

        Args:
            v: Flat vector to precondition

        Returns:
            P @ v (using internal_power)
        """
        return self._apply(v, power=self.internal_power)

    def pow(self, p: float) -> HybridPreconditioner:
        """Create new preconditioner with power p.

        Args:
            p: Power to raise the preconditioner to

        Returns:
            New HybridPreconditioner with internal_power = p
        """
        return HybridPreconditioner(self.blocks, self.shapes, internal_power=p)

    def _apply(self, v_flat: Array, power: float) -> Array:
        """Apply P^power to a flat vector.

        Args:
            v_flat: Flat vector
            power: Power to apply

        Returns:
            Result of applying P^power to v_flat
        """
        # 1. Slice v_flat into per-layer tensors
        v_tensors = _unflatten_helper(v_flat, self.shapes)

        out_tensors = []

        for v, block in zip(v_tensors, self.blocks):
            if isinstance(block, torch.Tensor):
                # --- MUON BLOCK ---
                # block is M = (G^T G)^{-1/2}
                # We need P^power. P is M^{-1}. So we need M^{-power}.

                # If power = -1 (Standard Step): M^1. Result = v @ M
                # If power = -0.5 (Eig Solver): M^0.5. Result = v @ M^0.5

                # Compute fractional power of M (using SVD for stability)
                # Note: For integer powers like -1, we can just multiply/inverse.
                op = matrix_power(block, -power)
                out_tensors.append(v @ op)

            else:
                # --- SGD BLOCK ---
                # block is scalar (e.g., 1.0 for SGD)
                # Result = v * (block ^ power)

                # If SGD: block=1.0. 1.0^(-1) = 1.0. No change.
                if isinstance(block, float) and block == 1.0:
                    out_tensors.append(v)  # Identity
                else:
                    out_tensors.append(v * (block ** power))

        return torch.cat([t.reshape(-1) for t in out_tensors])


def to_schedule(schedule_or_constant):
    """Optionally create an LR schedule from a constant LR."""
    if callable(schedule_or_constant):          # if it's a schedule ...
        return schedule_or_constant             #  ... do nothing.
    else:                                       # but if it's a constant...
        return lambda t: schedule_or_constant   # ... turn it into a schedule. 
