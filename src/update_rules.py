from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch

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
        
    def initialize_state(self, w: torch.Tensor) -> Array:
        """Initialize the state.
        
        Args:
          w: the initial weights
          
        Returns:
          Array: the state, as a flat vector (see subclasses for examples)
        """
        raise NotImplementedError()

    def step(self, w: Array, flat_state: Array, gradient: Array) -> Array:
        """Perform a single optimization step.
        
        Args:
            w (Array): the current weights
            flat_state (Array): the current state, as a flat vector
            gradient (Array): the current gradient
        
        Returns:
            Array: the updated weights
        """
        P = self.P(flat_state)
        return w - P.pow(-1)(gradient)

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

    def initialize_state(self, w: Array) -> Array:
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

    def initialize_state(self, w: Array) -> Array:
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

    def initialize_state(self, w: Array) -> Array:
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
    lr: float
    momentum: float = 0.95
    ns_steps: int = 5
    # We need to pass shapes to know how to reconstruct matrices
    shapes: list[torch.Size] = None

    def __post_init__(self):
        self.lr_fn = to_schedule(self.lr)

    def initialize_state(self, w: Array) -> Array:
        # State is just momentum, same shape as w
        state = {
            "t": torch.tensor(0.0, dtype=w.dtype, device=w.device),
            "momentum": torch.zeros_like(w)
        }
        flat_state, self.unflatten = flatten_pytree(state)
        return flat_state

    def update_state(self, flat_state: Array, gradient: Array) -> Array:
        state = self.unflatten(flat_state)
        t = state["t"]
        mom = state["momentum"]

        # Update momentum in-place (standard SGD momentum)
        mom.lerp_(gradient, 1 - self.momentum)

        state = {"t": t + 1.0, "momentum": mom}
        return flatten_pytree(state)[0]

    def P(self, flat_state: Array) -> Preconditioner:
        """
        Constructs the Preconditioner for Analysis.
        Uses SVD to allow calculating P^{-1/2}.
        """
        state = self.unflatten(flat_state)
        momentum_flat = state["momentum"]

        # We need to unflatten the momentum to process block-by-block
        blocks = self._unflatten_vector(momentum_flat)

        svd_factors = []
        for block in blocks:
            # Reshape Conv2d (Out, In, H, W) -> (Out, In*H*W)
            if block.ndim > 2:
                block = block.view(block.size(0), -1)

            # SVD for (MM^T)^{1/2}
            # If Tall (Rows > Cols), we transpose to run SVD on smaller dim
            transposed = False
            if block.size(0) > block.size(1):
                block = block.mT
                transposed = True

            try:
                # Use float32 for stability
                U, S, _ = torch.linalg.svd(block.float(), full_matrices=False)
                U, S = U.to(block.dtype), S.to(block.dtype)
            except:
                # Fallback
                U = torch.eye(block.size(0), device=block.device, dtype=block.dtype)
                S = torch.ones(block.size(0), device=block.device, dtype=block.dtype)

            svd_factors.append((U, S, transposed, block.shape))

        return MuonPreconditioner(svd_factors, self.lr_fn(state["t"]), self.shapes)

    def step(self, w: Array, flat_state: Array, gradient: Array) -> Array:
        """
        Performs the Muon Step using Newton-Schulz.
        """
        state = self.unflatten(flat_state)
        momentum_flat = state["momentum"]

        # 1. Unflatten momentum
        mom_blocks = self._unflatten_vector(momentum_flat)
        updates = []

        # 2. Apply Newton-Schulz to each block
        for mom in mom_blocks:
            updates.append(self._newton_schulz_update(mom))

        # 3. Flatten back
        update_flat = torch.cat([u.flatten() for u in updates])

        # 4. Apply update
        lr = self.lr_fn(state["t"])
        return w - lr * update_flat

    def _unflatten_vector(self, flat_vec):
        """Helper to break flat vector into blocks based on self.shapes"""
        blocks = []
        curr = 0
        for shape in self.shapes:
            numel = shape.numel()
            blocks.append(flat_vec[curr : curr + numel].view(shape))
            curr += numel
        return blocks

    def _newton_schulz_update(self, G):
        """Original Muon Newton-Schulz implementation"""
        # Handle shapes (Conv2d -> 2D)
        original_shape = G.shape
        if G.ndim > 2:
             G = G.view(G.size(0), -1)

        X = G.bfloat16()
        if X.size(0) > X.size(1): X = X.mT

        # NS Loop
        X = X / (X.norm() + 1e-7)
        for _ in range(self.ns_steps):
            A = X @ X.mT
            B = -4.7750 * A + 2.0315 * A @ A # Simplification of coeffs
            X = 3.4445 * X + B @ X

        if G.size(0) > G.size(1): X = X.mT

        # Scale factor
        scale = max(1, G.size(0)/G.size(1))**0.5
        update = X.to(G.dtype) * scale

        return update.view(original_shape)

    def summarize_state(self, flat_state):
        return {"lr": self.lr_fn(self.unflatten(flat_state)["t"])}

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


class MuonPreconditioner(Preconditioner):
    """Block-diagonal preconditioner for Muon optimizer using SVD factors."""

    def __init__(self, factors, lr, shapes):
        """Constructor for the Muon preconditioner.

        Args:
            factors: List of (U, S, transposed, shape) tuples from SVD
            lr: Learning rate
            shapes: List of original parameter shapes
        """
        self.factors = factors
        self.lr = lr
        self.shapes = shapes

    def __call__(self, v):
        # Applies P * v
        # P = (1/lr) * (MM^T)^{1/2}
        curr = 0

        # Unflatten v
        v_blocks = []
        for shape in self.shapes:
            numel = shape.numel()
            v_blocks.append(v[curr : curr+numel].view(shape))
            curr += numel

        # Apply blocks
        out_blocks = []
        for v_blk, (U, S, transposed, _) in zip(v_blocks, self.factors):
            target = v_blk
            if target.ndim > 2: target = target.view(target.size(0), -1)

            if transposed: target = target.mT

            # Apply (MM^T)^{1/2} via SVD factors: U * S * U.T
            res = U @ (S.unsqueeze(-1) * (U.mT @ target))

            if transposed: res = res.mT
            out_blocks.append(res.view(v_blk.shape).flatten())

        return torch.cat(out_blocks) * (1.0 / self.lr)

    def pow(self, p):
        # P^p = (1/lr)^p * (MM^T)^{p/2}
        new_factors = []
        for (U, S, tr, sh) in self.factors:
            new_factors.append((U, S.pow(p), tr, sh))
        return MuonPreconditioner(new_factors, self.lr**p, self.shapes)


def to_schedule(schedule_or_constant):
    """Optionally create an LR schedule from a constant LR."""
    if callable(schedule_or_constant):          # if it's a schedule ...
        return schedule_or_constant             #  ... do nothing.
    else:                                       # but if it's a constant...
        return lambda t: schedule_or_constant   # ... turn it into a schedule. 
