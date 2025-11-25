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
    """
    Muon optimizer with momentum, weight normalization and gradient whitening.
    
    State Structure (Flat):
        The state is stored as a flat tensor: [m_flat | t]
        - m_flat: momentum buffer, same size as weights (size: len(w))
        - t: step counter (size: 1)
        
    Note: Unlike other optimizers (GradientDescent, RMSProp), Muon uses a
    flat state representation instead of pytree for performance reasons,
    as Muon's update involves expensive Newton-Schulz iterations.
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
        self._is_matrix = [len(shape) == 2 for shape in self._shapes]

        print("[DEBUG] Parameters that will use Newton-Schulz whitening:")
        for name, is_mat in zip(self._names, self._is_matrix):
            marker = "✓" if is_mat else "✗"
            print(f"[DEBUG]   {marker} {name}")

        print("[DEBUG] sizes:", self._sizes)

        self._offsets = torch.cumsum(
            torch.tensor([0] + self._sizes[:-1]), dim=0
        )

        print("[DEBUG] offsets:", self._offsets.tolist())
        total_sizes = sum(self._sizes)

        print(f"[DEBUG] Total num params (sum shapes) = {total_sizes}")
        print("[DEBUG] Done binding model structure\n")

    def initialize_state(self, w: Array) -> Array:
        """
        Returns flat state with structure: [m_flat | t]
        - m_flat: momentum buffer (size: len(w))
        - t: step counter (size: 1)
        """

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

        # * ASSERT check
        expected_size = len(w) + 1  # momentum + step counter
        actual_size = len(flat_state)
        assert actual_size == expected_size, (
            f"State size mismatch: expected {expected_size}, got {actual_size}"
        )

        assert len(m_flat) == len(w), (
            f"Momentum buffer size mismatch: expected {len(w)}, got {len(m_flat)}"
        )
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
        assert len(G.shape) == 2, f"Expected 2D tensor, got shape {G.shape}"
        assert not torch.isnan(G).any(), "Input contains NaN"
        assert not torch.isinf(G).any(), "Input contains Inf"
        a, b, c = (3.4445, -4.7750, 2.0315)
        
        # Work in bfloat16 for speed (if supported)
        original_dtype = G.dtype
        X = G.to(torch.bfloat16) if torch.cuda.is_bf16_supported() else G
        
        # Normalize to ensure top singular value <= 1
        X = X / (X.norm() + self.eps)

        print(f"    [Newton-Schulz] Input shape={G.shape}, norm={G_norm:.4f}")
        
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
        
        assert not torch.isnan(X).any(), "Newton-Schulz produced NaN"
        assert not torch.isinf(X).any(), "Newton-Schulz produced Inf"
        return X.to(original_dtype)

    def _extract_state(self, flat_state: Array) -> Tuple[Array, torch.Tensor]:
        """Extract momentum and step counter from flat state."""
        assert len(flat_state) > 1, (
            f"State too small: expected at least 2 elements, got {len(flat_state)}"
        )
        m = flat_state[:-1]  # * All but last element (momentum buffer)
        t = flat_state[-1]    # * Last element (step counter)

        assert t.numel() == 1, f"Step counter should be scalar, got shape {t.shape}"
        if hasattr(self, "_total_size"):  # _total_size viene da bind_model_structure
            assert len(m) == self._total_size, (
                f"Momentum size mismatch: expected {self._total_size}, got {len(m)}"
            )
        return m, t

    def P(self, flat_state):
        # Just return identity-scaling preconditioner
        _, t = self._extract_state(flat_state)
        lr = self.lr_fn(t)
        return DiagonalPreconditioner(1.0 / lr)
    
    def update_state(self, flat_state: Array, gradient: Array) -> Array:
        """
        Update ONLY the state (momentum + step counter).
        This follows the standard template.
        """
        m, t = self._extract_state(flat_state)
        assert len(m) == len(gradient), (
            f"Momentum and gradient size mismatch: m={len(m)}, grad={len(gradient)}"
        )
        
        # * Standard momentum update: m = β*m + g
        m_old_norm = m.norm().item()
        m = self.momentum * m + gradient
        m_new_norm = m.norm().item()

        print(f"[update_state] t={t.item():.0f}, ||m_old||={m_old_norm:.4f}, "
              f"||m_new||={m_new_norm:.4f}, ||grad||={gradient.norm().item():.4f}")
        
        # * Increment step counter
        t = t + 1.0
        
        # * Reconstruct flat state
        new_state = torch.cat([m, t.view(1)])
        return new_state

    # ^ Override update method to implement Muon-specific update (other optimizers do not do this)
    def update(self, w: Array, flat_state: Array, gradient: Array) -> Tuple[Array, Array]:
        """
        Complete Muon update with weight normalization and gradient whitening.
        
        This OVERRIDES the base class update() because Muon doesn't follow
        the standard formula w = w - P^{-1}(gradient).
        """
        assert hasattr(self, "_is_matrix"), (
            "Missing _is_matrix attribute. Did you call bind_model_structure()?"
        )
        assert len(w) == len(gradient), (
            f"Weight and gradient size mismatch: w={len(w)}, grad={len(gradient)}"
        )
        m, t = self._extract_state(flat_state)
        lr = self.lr_fn(t)

        print(f"\n[update] Step {t.item():.0f}, lr={lr:.6f}")
        
        # * Update momentum buffer: m = β*m + g
        m = self.momentum * m + gradient
        
        # * Compute effective gradient (with Nesterov if requested)
        if self.nesterov:
            effective_grad = gradient + self.momentum * m
        else:
            effective_grad = m
        
        # * Process each parameter based on dimensionality
        w_chunks = []
        grad_chunks = []
        
        for offset, size, shape, is_matrix in zip(
            self._offsets, self._sizes, self._shapes, self._is_matrix
        ):
            # Extract parameter and gradient chunks
            w_chunk = w[offset:offset+size].view(shape)
            g_chunk = effective_grad[offset:offset+size].view(shape)

            if i < 3 or is_matrix:  # Mostra i primi 3 o tutti i matrix
                print(f"[update]   Param {i}: name={self._names[i]}, "
                    f"shape={shape}, is_matrix={is_matrix}, "
                    f"||w||={w_chunk.norm().item():.4f}, "
                    f"||g||={g_chunk.norm().item():.4f}")
            
            if is_matrix:
                # ? === MUON UPDATE FOR 2D MATRICES ===

                assert len(shape) == 2, f"is_matrix=True but shape={shape}"
                
                # Weight normalization: w ← w · √d / ||w||
                d = torch.tensor(size, dtype=w.dtype, device=w.device)
                w_norm = w_chunk.norm()
                w_chunk = w_chunk * (torch.sqrt(d) / (w_norm + self.eps))

                assert w_norm > self.eps, (
                    f"Weight norm too small for param {self._names[i]}: {w_norm}"
                )

                new_norm = w_chunk.norm().item()
                expected_norm = torch.sqrt(d).item()
                if abs(new_norm - expected_norm) > 0.1 * expected_norm:
                    print(f"[WARNING] Normalization check failed for {self._names[i]}: "
                        f"expected ||w||≈{expected_norm:.2f}, got {new_norm:.2f}")
                
                # Gradient whitening via Newton-Schulz
                g_whitened = self._zeropower_via_newtonschulz5(g_chunk)

                # * Sanity check for NaNs
                if torch.isnan(g_whitened).any():
                    raise ValueError(f"NaN detected in whitened gradient for {self._names[i]}")
                
            else:
                # ? === STANDARD UPDATE FOR 1D VECTORS (biases) ===
                # No normalization, no whitening
                g_whitened = g_chunk
            
            # Store processed chunks
            w_chunks.append(w_chunk.reshape(-1))
            grad_chunks.append(g_whitened.reshape(-1))
        
        # * Concatenate all parameters
        w = torch.cat(w_chunks)
        whitened_grad = torch.cat(grad_chunks)
        
        # * Apply update: w = w - lr * whitened_grad
        w = w - lr * whitened_grad
        
        # * Update state (momentum + step counter)
        new_state = torch.cat([m, (t + 1.0).view(1)])
        
        return w, new_state

    def dstate_dt(self, flat_state: Array, gradient: Array) -> Array:
        """
        Temporal derivative of state for continuous flows.
        """
        m, t = self._extract_state(flat_state)
        assert len(m) == len(gradient), (
            f"Momentum and gradient size mismatch in dstate_dt"
        )
        
        # Continuous-time momentum derivative
        if self.momentum < 1:
            dmdt = (gradient - m) / (1 - self.momentum)
        else:
            dmdt = gradient

        print(f"[dstate_dt] t={t.item():.0f}, "
              f"||dmdt||={dmdt.norm().item():.4f}, "
              f"||grad||={gradient.norm().item():.4f}")
        
        # dt/dt = 1
        dtdt = torch.tensor(1.0, dtype=t.dtype, device=t.device)
        
        # Reconstruct derivative
        dstate = torch.cat([dmdt, dtdt.view(1)])

        assert len(dstate) == len(flat_state), (
            f"dstate_dt output size mismatch: expected {len(flat_state)}, got {len(dstate)}"
        )
        return dstate

    def summarize_state(self, flat_state: Array) -> Dict[str, Any]:
        """Summary of state for logging."""
        m, t = self._extract_state(flat_state)

        assert hasattr(self, "_offsets"), (
            "summarize_state requires bind_model_structure() to be called first"
        )
        
        # Compute statistics per parameter type
        m_matrix_norm_sq = 0.0
        m_bias_norm_sq = 0.0
        n_matrix = 0
        n_bias = 0
        
        for offset, size, is_matrix in zip(self._offsets, self._sizes, self._is_matrix):
            m_chunk = m[offset:offset+size]
            if is_matrix:
                m_matrix_norm_sq += m_chunk.norm().item()**2
                n_matrix += 1
            else:
                m_bias_norm_sq += m_chunk.norm().item()**2
                n_bias += 1

        expected_params = len(self._offsets)
        actual_params = n_matrix + n_bias
        assert actual_params == expected_params, (
            f"Processed {actual_params} params but expected {expected_params}"
        )
        
        return {
            "t": t.item(),
            "lr": self.lr_fn(t),
            "momentum_norm": m.norm().item(),
            "momentum_matrix_norm": torch.sqrt(torch.tensor(m_matrix_norm_sq)).item() if n_matrix > 0 else 0.0,
            "momentum_bias_norm": torch.sqrt(torch.tensor(m_bias_norm_sq)).item() if n_bias > 0 else 0.0,
            "n_matrix_params": n_matrix, # * for debugging
            "n_bias_params": n_bias,     # * for debugging
        }

    def raw_eigs_from_eigs(self, flat_state: Array, eigs: Array):
        """Transform effective Hessian eigenvalues to raw eigenvalues."""
        if eigs is None:
            return None
        _, t = self._extract_state(flat_state)
        lr = self.lr_fn(t)
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
