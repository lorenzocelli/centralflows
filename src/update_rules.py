from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from .utils import flatten_pytree


Array = Any


def newtonschulz_step(G: torch.Tensor, steps: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
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
        B = a * torch.eye(m, device=G.device, dtype=G.dtype) + \
            b * XXt + \
            c * (XXt @ XXt)
            
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
        
    return X, A


def matrix_power(M: torch.Tensor, p: float, eps: float = 1e-3) -> torch.Tensor:
    """
    Computes M^p for a symmetric matrix M using SVD/Eigendecomposition.
    Used to compute P^{-1/2} from P^{-1} for eigenvalue analysis.
    
    Args:
        M: Square matrix.
        p: Power (e.g., 0.5 for sqrt, -1 for inverse).
        eps: Epsilon for stability.
    """
    # Use SVD: M = U S V^T. Since M is symmetric (P^-1 approx), U=V.
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)

    # Clamp singular values for numerical stability
    S = torch.clamp(S, min=eps)

    # Compute S^p element-wise
    S_p = S ** p

    # Reconstruct
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


@dataclass  # we make it a dataclass so that it can be instantiated from the command line
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
        t = state["t"]

        lrs = self.lr_fn(t)

        # ^ [DEBUG] Print P stats at first few steps
        if t.item() < 3:
            print(f"[Gradient Descent P()] Step {int(t.item())}: lr={self.lr_fn(t):.6f}", flush=True)
            print(f"[Gradient Descent P()] P_mean={(1/lrs).mean():.6f} | P_min={(1/lrs).min():.6f} | P_max={(1/lrs).max():.6f}", flush=True)
        return DiagonalPreconditioner(1 / lrs)

    def update_state(self, flat_state: Array, gradient: Array) -> Array:
        state = self.unflatten(flat_state)
        t = state["t"]
        # ^ [DEBUG] Print gradient stats at first few steps
        if t.item() < 3:
            print(f"[Gradient Descent update_state] Step {int(t.item())}: grad_norm={gradient.norm():.6f} | grad_min={gradient.min():.6f} | grad_max={gradient.max():.6f}", flush=True)
        state = {"t": state["t"] + 1.0}
        return flatten_pytree(state)[0]

    def summarize_state(self, flat_state: Array) -> Array:
        state = self.unflatten(flat_state)
        return {
            "t": state["t"],
            "lr": self.lr_fn(state["t"]),
        }

    def raw_eigs_from_eigs(self, flat_state: Array, eigs: Array):
        if eigs is None:
            return None
        lr = self.P(flat_state).pow(-1)(1.0)
        return eigs / lr


@dataclass
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
        return update / self.beta2 

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
        if eigs is None:
            return None
        ess = self.P(flat_state).pow(-1)(1.0) 
        return eigs / ess


@dataclass
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
        return update / self.beta2

    def summarize_state(self, flat_state: Array) -> Array:
        state = self.unflatten(flat_state)
        nu = state["nu"]
        ess = self.P(flat_state).pow(-1)(torch.ones_like(nu))
        selected_idx = np.linspace(0, len(nu) - 1, 25, dtype=int)
        return {
            "t": state["t"],
            "nu_l1": nu.sum(),                  
            "nu_selected_idx": nu[selected_idx],
            "ess_mean": ess.mean(),             
            "ess_harmonic_mean": ess.reciprocal().mean().reciprocal(),
            "lr": self.lr_fn(state["t"]),
        }

def warmup_cosine_schedule(base_lr: float, warmup_steps: int, total_steps: int = 2000):
    """Linear warmup + cosine decay schedule"""
    import math
    
    def schedule(t):
        if t < warmup_steps:
            # Linear warmup
            return base_lr * ((t + 1) / warmup_steps)
        else:
            # Cosine decay dopo warmup
            progress = (t - warmup_steps) / (total_steps - warmup_steps)
            return base_lr * 0.5 * (1 + math.cos(math.pi * progress))
    return schedule

@dataclass
class AdamW(UpdateRule):
    """The AdamW optimizer.

    This optimizer maintains an EMA m of the gradient and an EMA ν of the elementwise squared gradient,
    and takes gradient steps using the effective step sizes η * m / sqrt(ν).
    Our implementation supports optional learning rate scheduling,
    bias correction, and ε:

           m_{t} = (1 - β_1) m_{t-1} + β_1 ∇L(w_t)
           ν_{t} = (1 - β_2) ν_{t-1} + β_2 ∇L(w_t)^2
           m̂_{t} = m_t / (1 - β_1 ^ t)
           ν̂_{t} = ν_t / (1 - β_2 ^ t)
           w_{t+1} = w_t - η(t) * m̂_t / (sqrt (ν̂_t) + ε)
    The optimizer's state consists of the tuple (t, m, ν).
    """

    lr: float
    beta1: float = 0.9
    beta2: float = 0.999
    bias_correction: bool = True # Just like RMSProp
    eps: float = 1e-6 # Just like RMSProp
    warmup_steps: int = 100
    max_grad_norm: float = 1.0

    def clip_gradient(self, gradient: Array) -> Array:
        """Clip gradient norm for stability"""
        grad_norm = gradient.norm()
        if grad_norm > self.max_grad_norm:
            gradient = gradient * (self.max_grad_norm / grad_norm)
            print(f"[AdamW] Clipping gradient: {grad_norm:.2e} → {self.max_grad_norm}", flush=True)
        return gradient

    def __post_init__(self):
        # self.lr_fn = to_schedule(self.lr)
        self.lr_fn = warmup_cosine_schedule(self.lr, self.warmup_steps, total_steps=2000)
    
    def initialize_state(self, w: Array, unflatten: Any = None) -> Array:
        state = {
            "t": torch.tensor(0.0, dtype=w.dtype, device=w.device),
            "m": torch.zeros_like(w),
            "v": torch.zeros_like(w),
        }
        flat_state, self.unflatten = flatten_pytree(state)
        return flat_state

    def P(self, flat_state: Array) -> Array:
        # TODO could also include weight decay here
        state = self.unflatten(flat_state)
        t, m, v = state["t"], state["m"], state["v"]
        if self.bias_correction:
            # ! m_hat is not needed for the preconditioner, but will be used in update()
            # m_hat =  (m / (1 - self.beta1**(t)))
            v_hat = (v / (1 - self.beta2**(t)))
        else:
            # m_hat = m
            v_hat = v
        lrs = self.lr_fn(t) / (torch.sqrt(v_hat) + self.eps)

        # ^ [DEBUG] Print P stats at first few steps
        if t.item() < 3:
            print(f"[AdamW P()] Step {int(t.item())}: lr={self.lr_fn(t):.6f} | v_mean={v.mean():.6e} | v_hat_mean={v_hat.mean():.6e}", flush=True)
            print(f"[AdamW P()] P_mean={(1/lrs).mean():.6f} | P_min={(1/lrs).min():.6f} | P_max={(1/lrs).max():.6f}", flush=True)
        return DiagonalPreconditioner(1 / lrs)

    def update_state(self, flat_state: Array, gradient: Array) -> Array:
        state = self.unflatten(flat_state)
        t, m, v = state["t"], state["m"], state["v"]

        # ^ [DEBUG] Print gradient stats at first few steps
        if t.item() < 3:
            print(f"[AdamW update_state] Step {int(t.item())}: grad_norm={gradient.norm():.6f} | grad_min={gradient.min():.6f} | grad_max={gradient.max():.6f}", flush=True)

        # ^ Update m and v
        m_new = m * self.beta1 + gradient * (1 - self.beta1)
        v_new = v * self.beta2 + gradient**2 * (1 - self.beta2)

        # ^ [DEBUG] Print state changes at first few steps
        if t.item() < 3:
            print(f"[AdamW update_state] m: {m.norm():.6e} → {m_new.norm():.6e} | v: {v.mean():.6e} → {v_new.mean():.6e}", flush=True)

        state = {"t": t + 1.0, "m": m_new, "v": v_new}
        return flatten_pytree(state)[0]

    def update(self, w: Array, flat_state: Array, gradient: Array) -> Tuple[Array, Array]:
        """
            Custom update for AdamW: need to apply the preconditioner to MOMENTUM, not gradient
        """
        # Get state
        # ! Here we do not call update_state, it's already been called by the DiscreteProcess
        state = self.unflatten(flat_state)
                
        t, m = state["t"], state["m"]

        if self.bias_correction:
            m_hat =  (m / (1 - self.beta1**(t)))

            # ^ [DEBUG] Print bias correction effect at first few steps
            if t.item() <= 3:
                print(f"[AdamW update] Bias correction: factor={(1 - self.beta1**(t)):.6f} | m_norm={m.norm():.6e} | m_hat_norm={m_hat.norm():.6e}", flush=True)
        else:
            m_hat = m

        preconditioned_update = self.P(flat_state).pow(-1)(m_hat)

        # ^ [DEBUG] Print update stats
        if t.item() <= 3 or t.item() % 10 == 0:
            w_norm_before = w.norm().item()
            w_norm_after = (w - preconditioned_update).norm().item()
            print(f"[AdamW update] Step {int(t.item())}: prec_update_norm={preconditioned_update.norm():.6f} | w_norm: {w_norm_before:.4f} → {w_norm_after:.4f}", flush=True)

        # Apply preconditioner now to MOMENTUM
        w = w - preconditioned_update
        return w, flat_state

    def dstate_dt(self, flat_state: Array, gradient: Array) -> Array:
        update = self.update_state(flat_state, gradient) - flat_state
        return update / self.beta2

    def summarize_state(self, flat_state: Array) -> Array:
        state = self.unflatten(flat_state)
        v = state["v"]
        ess = self.P(flat_state).pow(-1)(torch.ones_like(v))
        selected_idx = np.linspace(0, len(v) - 1, 25, dtype=int)
        return {
            "t": state["t"],
            "v_l1": v.sum(),                  
            "v_selected_idx": v[selected_idx],
            "ess_mean": ess.mean(),             
            "ess_harmonic_mean": ess.reciprocal().mean().reciprocal(),
            "lr": self.lr_fn(state["t"]),
        }


@dataclass
class Muon(UpdateRule):
    #! WARNING: For testing purposes, right now we should implement Muon using always AdamW (for all layers)
    """
    The Muon optimizer.

    DESIGN CHOICE: Hybrid Optimization
    ----------------------------------
    This class treats parameters differently based on their dimensions:
    1. 2D+ parameters (Weights): Optimized via the Muon Newton-Schulz preconditioner.
    2. 1D parameters (Biases, LayerNorm): Optimized via standard AdamW.

    Hence, the optimizer state tracks both:
    - `exp_avg`: Momentum (used by both Muon and AdamW blocks).
    - `exp_avg_sq`: Variance (used only by AdamW blocks).
    """

    # Generic parameters
    lr: float

    # Muon parameters
    momentum: float = 0.95
    nesterov: bool = True

    # AdamW parameters
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    weight_decay: float = 0.01

    def __post_init__(self):
        self.lr_fn = to_schedule(self.lr)
        self.unflatten_params = None

    def initialize_state(self, w: Array, unflatten: Any = None) -> Array:
        """
        Initializes the optimizer state.
        Determines which parameters are 'Muon blocks' (2D+) and which are 'Adam blocks' (1D).
        """
        if unflatten is None:
            raise ValueError("unflatten function must be provided to initialize Muon optimizer state.")
        self.unflatten_params = unflatten

        params = self.unflatten_params(w)

        self.param_blocks = []
        cursor = 0

        for name, p in params.items():
            numel = p.numel()
            start = cursor
            end = cursor + numel

            # DESIGN CHOICE:
            # Matrices (ndim >= 2) get the Muon preconditioner.
            # Vectors (ndim < 2) get the AdamW preconditioner.
            block_type = "muon" if p.ndim >= 2 else "adam"

            #! For testing purposes, we force all layers to be AdamW
            block_type = "adam"

            self.param_blocks.append({
                "name": name,
                "type": block_type,
                "idx": (start, end),
                "shape": tuple(p.shape),
                "ndim": p.ndim,
            })

            cursor = end

        # [DEBUG] Print block summary
        adam_blocks = sum(1 for b in self.param_blocks if b["type"] == "adam")
        muon_blocks = sum(1 for b in self.param_blocks if b["type"] == "muon")
        total_2d = sum(1 for b in self.param_blocks if b["ndim"] >= 2)
        print(f"[Muon Init] Total blocks: {len(self.param_blocks)} | Adam: {adam_blocks} | Muon: {muon_blocks} | 2D+: {total_2d}", flush=True)

        state = {
            "t": torch.tensor(0.0, dtype=w.dtype, device=w.device),
            "exp_avg": torch.zeros_like(w),          # Momentum buffer
            "exp_avg_sq": torch.zeros_like(w),       # Variance buffer (Adam only)
        }

        flat_state, self.unflatten_state = flatten_pytree(state)
        return flat_state

    def P(self, flat_state: Array) -> Array:
        """
        Constructs the hybrid preconditioner object.

        DESIGN CHOICE: Storage of Explicit P^{-1}
        -----------------------------------------
        Normally, an UpdateRule's P() returns the forward preconditioner P.
        However, for Muon, the Newton-Schulz iteration naturally yields the *inverse*
        preconditioner A = P_{Muon}^{-1}(G_0).
        
        Instead of inverting A (which would be unstable and unnecessary), we store 
        A directly in the block and tag it as "muon_inverse". The logic in 
        BlockDiagonalPreconditioner handles this inversion flag.
        """
        state = self.unflatten_state(flat_state)
        t = state["t"]
        exp_avg = state["exp_avg"]        
        exp_avg_sq = state["exp_avg_sq"]  

        lr = self.lr_fn(t)

        P_blocks = []
        
        # [DEBUG] Print preconditioner info only at first step
        if t == 1.0:
            print(f"[Muon P()] Step {int(t.item())}: Building preconditioner with lr={lr:.6f}", flush=True)

        for block in self.param_blocks:
            block_type = block["type"]
            start_idx, end_idx = block["idx"]
            block_size = end_idx - start_idx

            if block_type == "muon":
                # For Muon blocks, the "gradient" G_0 is the momentum.
                m_flat = exp_avg[start_idx:end_idx]
                m_tensor = m_flat.view(block["shape"])

                # Compute Explicit P^-1 Matrix (A) via Parallel Newton-Schulz.
                # We disregard X here; we only need the operator A.
                _, A = newtonschulz_step(m_tensor, steps=5)

                P_blocks.append({
                    "type": "muon_inverse", # Tag: This block data IS ALREADY P^-1.
                    "size": block_size,
                    "data": A * lr,         # Store scaled P^{-1}
                    "shape": block["shape"],
                    "name": block["name"]
                })
            elif block_type == "adam":
                #! For testing purposes, we implement all layers as AdamW
                # For Adam blocks, we compute standard diagonal preconditioning.
                v_block = exp_avg_sq[start_idx:end_idx]
                P_block = (torch.sqrt(v_block) + self.adam_eps) / lr

                P_blocks.append({
                    "type": "diagonal",
                    "size": block_size,
                    "data": P_block,        # Store P
                    "name": block["name"]
                })

        return BlockDiagonalPreconditioner(P_blocks)

    def update_state(self, flat_state: Array, gradient: Array) -> Array:
        """
        Updates the internal state (momentum/variance) based on the raw gradient.
        This happens BEFORE preconditioning.
        """
        state = self.unflatten_state(flat_state)
        t = state["t"]
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]

        t_new = t + 1.0
        exp_avg_new = exp_avg.clone()
        exp_avg_sq_new = exp_avg_sq.clone()
        
        # [DEBUG] Print gradient stats at first few steps
        if t.item() < 3:
            print(f"[update_state] Step {int(t.item())}: grad_norm={gradient.norm().item():.6f} | grad_min={gradient.min().item():.6f} | grad_max={gradient.max().item():.6f}", flush=True)

        for block in self.param_blocks:
            start, end = block["idx"]
            g_block = gradient[start:end] 

            if block["type"] == "muon":
                # Muon: Update momentum only.
                # m_{t+1} = mu * m_t + g_t
                m_block = exp_avg[start:end]
                m_new = self.momentum * m_block + g_block
                exp_avg_new[start:end] = m_new

            elif block["type"] == "adam":
                # Adam: Update momentum and variance.
                m_block = exp_avg[start:end]
                m_new = self.adam_beta1 * m_block + (1 - self.adam_beta1) * g_block
                exp_avg_new[start:end] = m_new

                v_block = exp_avg_sq[start:end]
                v_new = self.adam_beta2 * v_block + (1 - self.adam_beta2) * (g_block ** 2)
                exp_avg_sq_new[start:end] = v_new
                
                # [DEBUG] Print variance stats at first few steps
                if t.item() < 3:
                    print(f"[update_state] Block {block['name']}: v_mean_before={v_block.mean().item():.6e} | v_mean_after={v_new.mean().item():.6e} | g_sq_mean={(g_block**2).mean().item():.6e}", flush=True)

        new_state = {
            "t": t_new,
            "exp_avg": exp_avg_new,
            "exp_avg_sq": exp_avg_sq_new,
        }

        flat_new_state = flatten_pytree(new_state)[0]
        return flat_new_state

    def update(self, w: Array, flat_state: Array, gradient: Array) -> Tuple[Array, Array]:
        """
        Perform the full optimization step: w_{t+1} = w_t - P^{-1}(state) * update_vector.
        
        DESIGN CHOICE: Update Vector Source
        -----------------------------------
        - For AdamW blocks, the update source is the Raw Gradient.
        - For Muon blocks, the update source is the Momentum buffer.
        """
        # 1. Update state (Momentum / Variance)
        flat_state = self.update_state(flat_state, gradient)

        state = self.unflatten_state(flat_state)

        #! Not calling update_state, because it's already been called by the DiscreteProcess.prepare()
        
        # [DEBUG] Print state (every step for now to track behavior)
        t = int(state['t'].item())
        if t <= 20 or t % 10 == 0:  # Verbose at start, then every 10 steps
            print(f"[Muon update] Step {t}: lr={self.lr_fn(state['t']):.6f} | exp_avg_norm={state['exp_avg'].norm().item():.4f} | exp_avg_sq_mean={state['exp_avg_sq'].mean().item():.6e}", flush=True)
        
        exp_avg = state["exp_avg"]

        # 2. Build the vector to be preconditioned
        update_blocks = []

        for block in self.param_blocks:
            start_idx, end_idx = block["idx"]

            if block["type"] == "muon":
                # Muon applies P^-1 to the Momentum.
                update_blocks.append(exp_avg[start_idx:end_idx])
            elif block["type"] == "adam":
                # Adam applies P^-1 to the Gradient.
                update_blocks.append(gradient[start_idx:end_idx])

        update_vector = torch.cat(update_blocks)

        # 3. Apply Preconditioner P^{-1}
        # .pow(-1) correctly handles the "muon_inverse" blocks by simply returning them.
        preconditioned_update = self.P(flat_state).pow(-1)(update_vector)
        
        # [DEBUG] Print update stats at first few steps
        if t <= 3 or t % 10 == 0:
            prec_norm = preconditioned_update.norm().item()
            prec_mean = preconditioned_update.mean().item()
            w_norm_before = w.norm().item()
            w_norm_after = (w - preconditioned_update).norm().item()
            print(f"[update] Step {t}: prec_update_norm={prec_norm:.6f} | prec_update_mean={prec_mean:.6e} | w_norm: {w_norm_before:.4f} → {w_norm_after:.4f}", flush=True)

        # 3. Update Weights
        w = w - preconditioned_update

        return w, flat_state

    def summarize_state(self, flat_state: Array) -> Dict[str, Any]:
        state = self.unflatten_state(flat_state)
        return {
            "t": state["t"],
            "lr": self.lr_fn(state["t"]),
            "exp_avg_norm": state["exp_avg"].norm().item(),
            "exp_avg_sq_mean": state["exp_avg_sq"].mean().item(),
        }

class Preconditioner:
    """Abstract class for a preconditioner."""
    
    def __call__(self, v: Array) -> Array:
        raise NotImplementedError()
    
    def pow(self, p: float) -> Preconditioner:
        raise NotImplementedError()


class DiagonalPreconditioner(Preconditioner):
    """A diagonal (i.e. elementwise) preconditioner."""
    
    def __init__(self, P):
        self.P = P

    def __call__(self, v: Array) -> Array:
        return v * self.P

    def pow(self, power: float) -> DiagonalPreconditioner:
        return DiagonalPreconditioner(self.P**power)


class BlockDiagonalPreconditioner(Preconditioner):
    """
    A block-diagonal preconditioner supporting hybrid Muon/AdamW blocks.

    Block Types:
    - "muon_inverse":
        Holds the Explicit Matrix A = P^{-1}.
        This block CANNOT be applied directly (forward P is not supported).
        It is a storage state waiting for .pow(-1).
    - "muon_matrix":
        Holds a generic matrix M (e.g. A, or A^{0.5}).
        This block CAN be applied via left-multiplication.
    - "diagonal":
        Holds a vector D (Standard AdamW).
        Applied via element-wise multiplication.
    """

    def __init__(self, blocks: list[dict]):
        self.blocks = blocks
        self.total_size = sum(b["size"] for b in blocks)
    
    def __call__(self, v: Array) -> Array:
        """Apply the preconditioner blocks to vector v."""
        if v.numel() != self.total_size:
            raise ValueError(f"Input vector size {v.numel()} doesn't match preconditioner size {self.total_size}")
        
        result_blocks = []
        cursor = 0
        
        for block in self.blocks:
            size = block["size"]
            v_block = v[cursor:cursor+size]
            
            if block["type"] == "muon_matrix":
                raise NotImplementedError("This block is muon_matrix, which is not yet implemented, now trying only AdamW")
                # Apply explicit matrix multiplication: M @ v_block
                # The matrix M is m x m. The vector v is flattened (m*n).
                # We interpret v as (m, n) and apply M on the left: M @ V.
                M = block["data"]
                m, n = block["shape"]
                
                v_reshaped = v_block.view(m, n)
                res = M @ v_reshaped
                result_blocks.append(res.flatten())
                
            elif block["type"] == "muon_inverse":
                # We hold P^-1, but the user asked to apply P.
                # Inverting A is unstable and unnecessary for this codebase.
                # raise NotImplementedError("Applying forward P (inverting A) is not supported for Muon blocks. Use .pow(-1) first.")
                raise NotImplementedError("This block is muon_inverse, which is not yet implemented, now trying only AdamW")
                
            else:  # diagonal
                #! For testing purposes, the code always uses AdamW blocks
                result_blocks.append(block["data"] * v_block)
            
            cursor += size
        
        return torch.cat(result_blocks, dim=0)
    
    def pow(self, power: float) -> BlockDiagonalPreconditioner:
        """
        Return a new preconditioner raised to power `p`.
        
        Crucial logic for Muon blocks:
        Since we store A = P^{-1}, computing P^p means computing A^{-p}.
        """
        new_blocks = []
        
        for block in self.blocks:
            new_block = block.copy()
            
            if block["type"] == "muon_inverse":
                raise NotImplementedError("This block is muon_inverse, which is not yet implemented, now trying only AdamW")
                # Stored data is A = P^{-1}.
                A = block["data"]
                
                if power == -1:
                    # Request: P^{-1}. 
                    # Logic: We already have A. Just change type to "muon_matrix".
                    new_block["data"] = A
                    new_block["type"] = "muon_matrix" 
                    
                elif power == -0.5:
                    # Request: P^{-1/2}.
                    # Logic: We need sqrt(P^{-1}) = sqrt(A).
                    # A is an m x m matrix, so we use SVD/Eig to compute A^0.5.
                    new_block["data"] = matrix_power(A, 0.5)
                    new_block["type"] = "muon_matrix"
                    
                else:
                    # General case: P^p = (P^{-1})^{-p} = A^{-p}
                    new_block["data"] = matrix_power(A, -power)
                    new_block["type"] = "muon_matrix"

            elif block["type"] == "muon_matrix":
                raise NotImplementedError("This block is muon_matrix, which is not yet implemented, now trying only AdamW")
                # Already a generic matrix M. Compute M^p.
                new_block["data"] = matrix_power(block["data"], power)

            else:  # diagonal
                #! For testing purposes, the code always uses AdamW blocks
                # (Diagonal)^p = element-wise power
                new_block["data"] = block["data"] ** power
            
            new_blocks.append(new_block)
        
        return BlockDiagonalPreconditioner(new_blocks)


def to_schedule(schedule_or_constant):
    """Optionally create an LR schedule from a constant LR."""
    if callable(schedule_or_constant):          
        return schedule_or_constant             
    else:                                       
        return lambda t: schedule_or_constant