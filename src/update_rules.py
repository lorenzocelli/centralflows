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

    This class is the entire Muon optimizer, hence it should deal with ALL PARAMETERS together,
    and internally should handle the difference between 2D (layers) and 1D (biases) parameters.

    In particular, 1D parameters should be assigned an AdamW-like update rule, while 2D parameters should
    compute the Muon preconditioner.

    Hence, the state should contain both the AdamW state for the 1D parameters, as well as the Muon state for the 2D parameters.
    """

    # ^ Generic parameters
    lr: float

    # ^ Muon parameters
    momentum: float = 0.95
    nesterov: bool = True

    # ^ AdamW parameters
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    weight_decay: float = 0.01

    def __post_init__(self):
        self.lr_fn = to_schedule(self.lr)
        self.unflatten_fn = None

    def initialize_state(self, w: Array, unflatten: Any = None) -> Array:
        """
        This function should initialize the state (t [scalar], momentum [for all parameters, shape of w], 
                        variance [for 1D parameters only, shape of w, with 0s for 2D parameters])

        The unflatten function is required to obtain the dictionary of the parameters
        
        Args:
            w: the initial weights
            unflatten: function to unflatten the weights into a pytree (dict of tensors) [optional]
        """
        if unflatten is None:
            raise ValueError("unflatten function must be provided to initialize Muon optimizer state.")
        self.unflatten_fn = unflatten

        params = self.unflatten_fn(w)

        self.param_blocks = []
        cursor = 0

        for name, p in params.items():
            numel = p.numel()
            start = cursor
            end = cursor + numel

            block_type = "muon" if p.ndim >= 2 else "adam"

            self.param_blocks.append({
                "name": name,
                "type": block_type,
                "idx": (start, end),
                "shape": tuple(p.shape),
                "ndim": p.ndim,
            })

            cursor = end

        state = {
            "t": torch.tensor(0.0, dtype=w.dtype, device=w.device),
            "exp_avg": torch.zeros_like(w),          #!  used by BOTH Muon and AdamW blocks
            "exp_avg_sq": torch.zeros_like(w),       #!  used only for AdamW blocks
        }

        # ------------------------- DEBUG BLOCK -------------------------
        print("\n[Muon Initialize] Parameter Block Summary:")
        count_muon = 0
        count_adam = 0

        for b in self.param_blocks:
            name = b["name"]
            btype = b["type"]
            s, e = b["idx"]
            shape = b["shape"]

            if btype == "muon":
                count_muon += 1
            else:
                count_adam += 1

            print(f"  - {name:25s} | {btype.upper():5s} | idx=({s},{e}) | shape={shape}")

        print(f"\n[Muon Initialize] TOTAL blocks: {len(self.param_blocks)}")
        print(f"   - Muon (2D+) blocks : {count_muon}")
        print(f"   - AdamW (1D) blocks : {count_adam}")
        print(f"   - Total parameters  : {w.numel()} entries")
        print("[Muon Initialize] Flat state created with fields: t, momentum, exp_avg, exp_avg_sq\n")
        # ---------------------------------------------------------------

        # * Remember to store the unflatten function (different from the unflatten_fn) for later use
        # * self.unflatten -> function to unflatten the optimizer state
        # * self.unflatten_fn -> function to unflatten the model parameters
        flat_state, self.unflatten = flatten_pytree(state)

        # ------------------------- TEST BLOCK -------------------------
        print("\n[Muon Test] Testing P construction...")
        test_P = self.P(flat_state)
        
        # Test 1: Size consistency
        assert test_P.total_size == w.numel(), f"P size mismatch: {test_P.total_size} vs {w.numel()}"
        print(f"[Muon Test] ✓ P.total_size matches w.numel() = {w.numel()}")
        
        # Test 2: Forward pass (P @ ones)
        test_vec = torch.ones_like(w)
        result = test_P(test_vec)
        assert result.shape == w.shape, f"P(v) shape mismatch: {result.shape} vs {w.shape}"
        print(f"[Muon Test] ✓ P(ones) has correct shape")
        
        # Test 3: Inverse (P^-1 @ ones)
        P_inv = test_P.pow(-1)
        result_inv = P_inv(test_vec)
        print(f"[Muon Test] ✓ P^-1(ones) computed successfully")
        print(f"[Muon Test]   P^-1(ones) stats: mean={result_inv.mean():.6f}, std={result_inv.std():.6f}")
        
        print("[Muon Test] All tests passed!\n")
        # --------------------------------------------------------------
        return flat_state

    def P(self, flat_state: Array) -> Array:
        """This function should do a distinction between 1D and 2D parameters,
        and build the block-diagonal preconditioner accordingly. For the 1D parameters, 
        should use AdamW preconditioner, while for the 2D parameters should compute the Muon preconditioner.
        """
        state = self.unflatten(flat_state)
        t = state["t"]
        exp_avg_sq = state["exp_avg_sq"] # should be the variance buffer
        print(f"[DEBUG P] t={t.item()}, exp_avg_sq shape={exp_avg_sq.shape}", flush=True)

        lr = self.lr_fn(t)

        P_diag_blocks = []
        print(f"[DEBUG P] Processing {len(self.param_blocks)} blocks", flush=True)
        for i, block in enumerate(self.param_blocks):
            print(f"[DEBUG P] Processing block {i}/{len(self.param_blocks)}: {block['name']} ({block['type']})", flush=True)
            block_type = block["type"]
            start_idx, end_idx = block["idx"]
            block_size = end_idx - start_idx

            if block_type == "muon":
                # TODO implement muon case
                # This is not the final implementation, just a placeholder
                P_diag_blocks.append({
                    "type": "identity",
                    "size": block_size,
                    "data": None,
                    "name": block["name"],
                    "scale": 1.0 / lr
                })
            elif block_type == "adam":
                # * Extract the v for this specific block
                v_block = exp_avg_sq[start_idx:end_idx]
                P_block = (torch.sqrt(v_block) + self.adam_eps) / lr

                P_diag_blocks.append({
                    "type": "diagonal",
                    "size": block_size,
                    "data": P_block,
                    "name": block["name"]
                })
        print("[DEBUG P] Finished processing blocks, constructing BlockDiagonalPreconditioner", flush=True)
        return BlockDiagonalPreconditioner(P_diag_blocks)

    def update_state(self, flat_state: Array, gradient: Array) -> Array:
        """This function should update the state accordingly:
        - for t, increment by 1
        - for momentum, update for ALL parameters
        - for variance, update only for 1D parameters
        """
        # * Get the current state
        state = self.unflatten(flat_state)
        t = state["t"]
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]

        t_new = t + 1.0
        exp_avg_new = exp_avg.clone()
        exp_avg_sq_new = exp_avg_sq.clone()

        debug_info = {
            "muon_blocks": {"count": 0, "grad_norm": 0.0, "exp_avg_norm": 0.0},
            "adam_blocks": {"count": 0, "grad_norm": 0.0, "exp_avg_norm": 0.0, "exp_avg_sq_mean": 0.0}
        }

        for block in self.param_blocks:
            start, end = block["idx"]
            g_block = gradient[start:end] #! This is the gradient for the current block of parameters

            if block["type"] == "muon":
                # ^ Muon Update (TODO IMPLEMENT)
                m_block = exp_avg[start:end]
                m_new = self.momentum * m_block + g_block
                exp_avg_new[start:end] = m_new

                debug_info["muon_blocks"]["count"] += 1
                debug_info["muon_blocks"]["grad_norm"] += g_block.norm().item() ** 2
                debug_info["muon_blocks"]["exp_avg_norm"] += m_new.norm().item() ** 2

            elif block["type"] == "adam":
                # ^ AdamW Update
                m_block = exp_avg[start:end]
                m_new = self.adam_beta1 * m_block + (1 - self.adam_beta1) * g_block
                exp_avg_new[start:end] = m_new

                v_block = exp_avg_sq[start:end]
                v_new = self.adam_beta2 * v_block + (1 - self.adam_beta2) * (g_block ** 2)
                exp_avg_sq_new[start:end] = v_new

                debug_info["adam_blocks"]["count"] += 1
                debug_info["adam_blocks"]["grad_norm"] += g_block.norm().item() ** 2
                debug_info["adam_blocks"]["exp_avg_norm"] += m_new.norm().item() ** 2
                debug_info["adam_blocks"]["exp_avg_sq_mean"] += v_new.mean().item()
        
        # === DEBUG: Print summary (only every N steps to avoid spam) ===
        if int(t_new.item()) % 100 == 0 or t_new.item() <= 2:  # Print at start and every 100 steps
            print(f"\n[Muon update_state] Step t={int(t_new.item())}", flush=True)
            
            # Muon blocks summary
            if debug_info["muon_blocks"]["count"] > 0:
                muon_grad_rms = (debug_info["muon_blocks"]["grad_norm"] / debug_info["muon_blocks"]["count"]) ** 0.5
                muon_exp_avg_rms = (debug_info["muon_blocks"]["exp_avg_norm"] / debug_info["muon_blocks"]["count"]) ** 0.5
                print(f"  [Muon blocks] count={debug_info['muon_blocks']['count']}", flush=True)
                print(f"    └─ grad RMS      = {muon_grad_rms:.6e}", flush=True)
                print(f"    └─ exp_avg RMS   = {muon_exp_avg_rms:.6e}", flush=True)
            
            # AdamW blocks summary
            if debug_info["adam_blocks"]["count"] > 0:
                adam_grad_rms = (debug_info["adam_blocks"]["grad_norm"] / debug_info["adam_blocks"]["count"]) ** 0.5
                adam_exp_avg_rms = (debug_info["adam_blocks"]["exp_avg_norm"] / debug_info["adam_blocks"]["count"]) ** 0.5
                adam_exp_avg_sq_mean = debug_info["adam_blocks"]["exp_avg_sq_mean"] / debug_info["adam_blocks"]["count"]
                print(f"  [AdamW blocks] count={debug_info['adam_blocks']['count']}", flush=True)
                print(f"    ├─ grad RMS      = {adam_grad_rms:.6e}", flush=True)
                print(f"    ├─ exp_avg RMS   = {adam_exp_avg_rms:.6e}", flush=True)
                print(f"    └─ exp_avg_sq    = {adam_exp_avg_sq_mean:.6e}", flush=True)
            
            # Overall gradient norm
            total_grad_norm = gradient.norm().item()
            print(f"  [Overall] Total gradient norm = {total_grad_norm:.6e}\n", flush=True)
        # ================================================================

        # * Construct the new state
        new_state = {
            "t": t_new,
            "exp_avg": exp_avg_new,
            "exp_avg_sq": exp_avg_sq_new,
        }

        flat_new_state = flatten_pytree(new_state)[0]
        return flat_new_state

    def update(self, w: Array, flat_state: Array, gradient: Array) -> Tuple[Array, Array]:
        """
        Perform the full optimization step.
        
        OVERRIDE REQUIRED: We cannot rely on the base class implementation (w - P_inv * g) because
        Muon and AdamW apply updates to the Momentum, not the raw Gradient, and Muon uses a non-linear
        orthogonalization step.
        """
        raise NotImplementedError()

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


class BlockDiagonalPreconditioner(Preconditioner):
    """A block-diagonal preconditioner.

    The preconditioner is represented as a list of blocks, where each block
    can be either an Identity matrix (for Muon) or a Diagonal matrix (for AdamW).
    """
    # TODO CHANGE DESCRIPTION, FOR NOW IT'S OK BUT MUON IS NOT IDENTITY IN FINAL VERSION

    def __init__(self, blocks: list[dict]):
        """Constructor for the block-diagonal preconditioner.
        
        Args:
            blocks: List of dicts, each containing:
                - "type": "identity" or "diagonal"
                - "size": number of parameters in this block
                - "data": None (for identity) or torch.Tensor (for diagonal)
                - "name": parameter name (for debugging)
        """
        self.blocks = blocks
        
        # Precompute total size for validation
        self.total_size = sum(b["size"] for b in blocks)
        
        # === DEBUG INFO ===
        """print("\n[BlockDiagonalPreconditioner] Created with structure:")
        for i, b in enumerate(blocks):
            btype = b["type"]
            size = b["size"]
            name = b["name"]
            if btype == "identity":
                print(f"  Block {i:2d}: {name:25s} | IDENTITY   | size={size:6d}")
            else:
                print(f"  Block {i:2d}: {name:25s} | DIAGONAL   | size={size:6d} | mean={b['data'].mean().item():.6f}")
        print(f"[BlockDiagonalPreconditioner] Total size: {self.total_size}\n")"""
    
    def __call__(self, v: Array) -> Array:
        """Apply the preconditioner: compute P @ v.
        
        Args:
            v: vector of size (total_size,)
        
        Returns:
            P @ v
        """
        if v.numel() != self.total_size:
            raise ValueError(f"Input vector size {v.numel()} doesn't match preconditioner size {self.total_size}")
        
        result_blocks = []
        cursor = 0
        
        for block in self.blocks:
            size = block["size"]
            v_block = v[cursor:cursor+size]
            
            if block["type"] == "identity":
                # Identity: P @ v = v
                result_blocks.append(v_block)
            else:  # diagonal
                # Diagonal: P @ v = diag * v (element-wise)
                result_blocks.append(block["data"] * v_block)
            
            cursor += size
        
        return torch.cat(result_blocks, dim=0)
    
    def pow(self, power: float) -> BlockDiagonalPreconditioner:
        """Return P^power (each block raised to the power).
        
        Args:
            power: the exponent
        
        Returns:
            A new BlockDiagonalPreconditioner with each block raised to the power.
        """
        print(f"[BlockDiagonalPreconditioner] Raising preconditioner to power {power}", flush=True)
        new_blocks = []
        
        for block in self.blocks:
            new_block = block.copy()
            
            if block["type"] == "identity":
                # Identity^p = Identity
                new_block["data"] = None
            else:  # diagonal
                # (Diagonal)^p = element-wise power
                new_block["data"] = block["data"] ** power
            
            new_blocks.append(new_block)
        
        print(f"[BlockDiagonalPreconditioner] Completed power operation", flush=True)
        return BlockDiagonalPreconditioner(new_blocks)



def to_schedule(schedule_or_constant):
    """Optionally create an LR schedule from a constant LR."""
    if callable(schedule_or_constant):          # if it's a schedule ...
        return schedule_or_constant             #  ... do nothing.
    else:                                       # but if it's a constant...
        return lambda t: schedule_or_constant   # ... turn it into a schedule. 
