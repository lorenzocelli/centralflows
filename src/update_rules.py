from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch

from .utils import flatten_pytree
from torch.utils._pytree import tree_map


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
    """Muon optimizer with momentum, weight normalization and gradient whitening.
    
    Muon combines:
    1. Momentum (optional Nesterov)
    2. Weight normalization: keeps weights on a sphere
    3. Gradient whitening via Newton-Schulz iteration
    
    The algorithm is:
        m_t = β m_{t-1} + ∇L(w_t)
        g_t = m_t + β m_t  (if Nesterov, otherwise g_t = m_t)
        w_t = w_t · √d / ||w_t||  (normalization)
        Δ = ZeroPower(g_t)  (whitening)
        w_{t+1} = w_t - η Δ
    
    N.B.: The whitening reshapes the gradient as a matrix (first_dim, -1) and applies
    Newton-Schulz to approximate UV^T of the SVD.
    """

    lr: float
    momentum: float = 0.9
    nesterov: bool = True
    ns_steps: int = 3 # Newton-Schulz steps
    eps: float = 1e-7 # for numerical stability

    def __post_init__(self):
        self.lr_fn = to_schedule(self.lr)

    def bind_model_structure(self, model: torch.nn.Module):
        """Bind the model structure to the optimizer (debug included)."""

        self._model = model

        assert hasattr(self, "_model"), (
            "You must call opt.bind_model_structure(model) before initialize_state()!"
        )

        print("\n[DEBUG] >>> binding model structure <<<")

        params = list(self._model.named_parameters())
        print(f"[DEBUG] Found {len(params)} parameters in model")

        for name, p in params:
            print(f"[DEBUG] param: {name:30s} shape={tuple(p.shape)} dim={p.dim()} size={p.numel()}")

        self._names = [name for name, _ in params]
        self._shapes = [p.shape for _, p in params]
        self._sizes = [p.numel() for _, p in params]

        print("[DEBUG] sizes:", self._sizes)

        self._offsets = torch.cumsum(
            torch.tensor([0] + self._sizes[:-1]), dim=0
        )

        print("[DEBUG] offsets:", self._offsets.tolist())
        total_sizes = sum(self._sizes)

        print(f"[DEBUG] Total num params (sum shapes) = {total_sizes}")
        print("[DEBUG] Done binding model structure\n")

    def initialize_state(self, w: Array) -> Array:
        """Initialize the state with momentum buffer and step counter."""
        print("\n[DEBUG] >>> initialize_state called <<<")

        assert hasattr(self, "_shapes"), (
            "You should call Muon.bind_model_structure(model) before initialize_state"
        )

        momenta_chunks = []
        for name, shape, size, offset in zip(self._names, self._shapes, self._sizes, self._offsets):
            # * Get the portion of w corresponding to this parameters
            param_flat = w[offset: offset + size]
            param = param_flat.view(shape)

            # * Distinguish between Muon and non-Muon parameters (although they all get 0 momentum init)
            if param.dim() == 2:
                print(f"[DEBUG] Initializing Muon momentum for MATRIX {name}: shape={shape}")
                m = torch.zeros_like(param)   # momentum matrix Muon (all 0 init)
            else:
                print(f"[DEBUG] Initializing NON-Muon momentum for {name}: shape={shape}")
                m = torch.zeros_like(param)   # normal momentum vector (all 0 init)

            momenta_chunks.append(m.reshape(-1)) # * flatten and store

        m_flat = torch.cat(momenta_chunks)
        t = torch.tensor(0.0, dtype=w.dtype, device=w.device)

        flat_state = torch.cat([m_flat, t.view(1)])
        print("[DEBUG] flat_state shape:", flat_state.shape)
        print("[DEBUG] >>> initialize_state completed <<<\n")
        return flat_state
    
    def _zeropower_via_newtonschulz5(self, G: torch.Tensor) -> torch.Tensor:
        """
        Newton-Schulz iteration to calculate the zeroth power / orthogonalization of G.
        
        Approximates UV^T where USV^T = G is the SVD, but much faster.
        The coefficients are optimized to maximize the slope at zero.

        This implementation is taken from CIFAR10-muon repo.
        """
        assert len(G.shape) == 2
        a, b, c = (3.4445, -4.7750, 2.0315)
        
        # Work in bfloat16 for speed (if supported)
        original_dtype = G.dtype
        X = G.to(torch.bfloat16) if torch.cuda.is_bf16_supported() else G
        
        # Normalize to ensure top singular value <= 1
        X = X / (X.norm() + self.eps)
        
        # Transpose if necessary (work on smaller dimension)
        transposed = G.size(0) > G.size(1)
        if transposed:
            X = X.T
        
        # Newton-Schulz iteration (quintic)
        for _ in range(self.ns_steps):
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X
        
        if transposed:
            X = X.T
        
        return X.to(original_dtype)

    def P(self, flat_state):
        # Just return identity-scaling preconditioner
        lr = self.lr_fn(flat_state[-1])
        return DiagonalPreconditioner(torch.full_like(flat_state[:-1], 1.0 / lr))
    
    def update_state(self, flat_state: Array, gradient: Array) -> Array:
        """Updates momentum buffer and step counter."""
        state = self.unflatten(flat_state)
        t, m = state["t"], state["m"]
        
        # Update momentum buffer: m_t = β m_{t-1} + g_t
        m = self.momentum * m + gradient
        
        state = {"t": t + 1.0, "m": m}
        return flatten_pytree(state)[0]

    def update(self, w, state, grad):
        # 1. split state
        m_flat = state[:-1]
        t = state[-1]

        # 2. ricostruisci momentum e gradient chunk per chunk
        new_m_chunks = []
        new_w_chunks = []

        idx = 0
        for name, shape, size, offset in zip(self._names, self._shapes, self._sizes, self._offsets):
            w_chunk = w[offset: offset+size].view(shape)
            g_chunk = grad[offset: offset+size].view(shape)
            m_chunk = m_flat[offset: offset+size].view(shape)

            if w_chunk.dim() == 2:
                # --- Muon update ---
                U = self._zeropower_via_newtonschulz5(g_chunk)

                m_new = beta * m_chunk + (1 - beta) * U
                w_new = w_chunk - lr * m_new

            else:
                # --- normal momentum ---
                m_new = beta * m_chunk + (1 - beta) * g_chunk
                w_new = w_chunk - lr * m_new

            new_m_chunks.append(m_new.reshape(-1))
            new_w_chunks.append(w_new.reshape(-1))

        # 3. ricombina tutto
        new_m_flat = torch.cat(new_m_chunks)
        new_w_flat = torch.cat(new_w_chunks)

        # 4. nuovo step counter
        new_t = t + 1

        new_state = torch.cat([new_m_flat, new_t.view(1)])
        return new_w_flat, new_state

    def dstate_dt(self, flat_state: Array, gradient: Array) -> Array:
        """
        Temporal derivative of the state for continuous flows.
        
        For momentum: dm/dt = (g - m) / β_scaled
        where β_scaled accounts for discretization.
        """
        state = self.unflatten(flat_state)
        m = state["m"]
        
        # Derivative of the momentum buffer (see paper for details)
        dmdt = (gradient - m) / (1 - self.momentum) if self.momentum < 1 else gradient
        
        new_state = {"t": torch.ones_like(state["t"]), "m": dmdt}
        return flatten_pytree(new_state)[0]

    def summarize_state(self, flat_state: Array) -> Dict[str, Any]:
        """Summary of the state for logging."""
        state = self.unflatten(flat_state)
        m = state["m"]
        
        # Statistics on the momentum buffer
        m_norm = m.norm()
        m_mean = m.mean()
        m_std = m.std()
        
        return {
            "t": state["t"],
            "lr": self.lr_fn(state["t"]),
            "momentum_norm": m_norm,
            "momentum_mean": m_mean,
            "momentum_std": m_std,
        }

    def raw_eigs_from_eigs(self, flat_state: Array, eigs: Array):
        """
        Transform effective Hessian eigenvalues into raw eigenvalues.
        
        For Muon this is complex because the preconditioner is not diagonal.
        For now we use a simple approximation.
        """
        if eigs is None:
            return None
        lr = self.P(flat_state).pow(-1)(1.0)
        return eigs / lr


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


def to_schedule(schedule_or_constant):
    """Optionally create an LR schedule from a constant LR."""
    if callable(schedule_or_constant):          # if it's a schedule ...
        return schedule_or_constant             #  ... do nothing.
    else:                                       # but if it's a constant...
        return lambda t: schedule_or_constant   # ... turn it into a schedule. 
