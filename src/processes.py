from __future__ import annotations

from collections import defaultdict
from dataclasses import InitVar, dataclass, field, replace
from math import ceil
from typing import Any, Optional, Dict, Literal

import torch
from torch.func import vmap

from .sdcp import SDCPSolver
from .eig_solvers import WarmStartEigSolver
from .flows import central_flow_substep, stable_flow_substep
from .loss_function import LossFunction
from .update_rules import UpdateRule, Preconditioner
from .utils import Timer, apply_to_pairs, apply_overrides

Array = Any


@dataclass
class Process:
    """Parent class of discrete optimizer, central flow, stable flow."""

    opt: UpdateRule          # the optimizer update rule
    loss_fn: LossFunction    # the loss function
    w: Array                 # the current iterate
    state: Array             # the current optimizer state
    
    # for computing eigenvalues of the effective hessian
    eig_manager: EigManager = field(init=False)
    
    # [optional] for computing eigenvalues of the "raw" hessian
    raw_eig_manager: Optional[EigManager] = field(init=False, default=None)

    # these fields are exposed to users of the class (especially the loggers)
    loss: Array = field(init=False, default=None)
    gradient: Array = field(init=False, default=None)
    eff_eigs: Array = field(init=False, default=None)

    # settings for the eigenvalue computation
    eig_config: InitVar[EigConfig]

    def __post_init__(self, eig_config: EigConfig):
        self.w = self.w.clone()
        self.state = self.state.clone()
        
        # eigenvalue manager for the effective Hessian
        self.eig_manager = EigManager(self.loss_fn, self.w, config=eig_config)
        
        # if the optimizer admits no trivial way of computing the "raw" Hessian eigenvalues
        # given the effective Hessian eigenvalues, create an EigManger for the "raw" Hessian
        if not hasattr(self.opt, "raw_eigs_from_eigs") and eig_config.raw_eigenvalues:
            self.raw_eig_manager = EigManager(self.loss_fn, self.w, config=replace(eig_config, track_threshold=None))

    def to_dict(self) -> Dict:
        """Serialize to a dict."""
        return dict(w=self.w, state=self.state)

    def from_dict(self, d: Dict):
        """Unserialize from a dict."""
        self.w = torch.tensor(d["w"], dtype=self.w.dtype, device=self.w.device)
        self.state = torch.tensor(
            d["state"], dtype=self.state.dtype, device=self.state.device
        )

    def prepare(self) -> Dict:
        """Prepare to take the step.
        
        Return:
          logs (dict)
        """
        pass
        
    def step(self):
        """Actually take the step."""
        pass
    
    
@dataclass
class EigConfig:
    # Frequency, in steps, at which to recompute Hessian
    # eigenvalues/eigenvectors using a warm-started
    # iterative algorithm (-1 means never)
    frequency: int = -1
    
    # All eigenvalues of the effective Hessian that are
    # above this threshold will be tracked
    track_threshold: float = 1.5

    # Number of eigenvalues to track initially.
    initial_neigs: int = 1
    
    # Solver to use for eigenvalue computations (lobpcg or scipy)
    solver: Literal["lobpcg", "scipy"] = "lobpcg"
    
    # Numerical tolerance for eigenvalue solving
    tol: float = 1e-10
    
    # Maximum number of HVP's to do on the GPU at once (-1 means no bound)
    chunk_size: int = -1
    
    # Whether to compute the eigenvalues of the "raw" Hessian in addition
    # to the eigenvalues of the effective Hessian
    raw_eigenvalues: bool = True
    

# process-specific EigConfig overrides.
# has same entries as EigConfig, but all defaulting to None
@dataclass
class EigConfigOverride:
    frequency: Optional[int] = None
    track_threshold: Optional[float] = None
    initial_neigs: Optional[int] = None
    solver: Optional[Literal["lobpcg", "scipy"]] = None
    tol: Optional[float] = None
    chunk_size: Optional[int] = None
    raw_eigenvalues: Optional[bool] = None

    
@dataclass
class DiscreteProcessConfig:
    # optional process-specific configs for eigenvalue computation.
    # settings here will override those in the global eig config
    eig: EigConfigOverride = field(default_factory=EigConfigOverride)


@dataclass
class DiscreteProcess(Process):
    """Run the discrete optimization algorithm."""
    
    config: DiscreteProcessConfig

    def __post_init__(self, global_eig_config: EigConfig):
        # override global eig config with non-None entries of the process-specific eig config
        eig_config = apply_overrides(global_eig_config, self.config.eig)
        super().__post_init__(eig_config)

    def prepare(self):
        timer = Timer()
        with timer("grad_and_value"):
            self.gradient, self.loss = self.loss_fn.grad_and_value(self.w)
            self.state = self.opt.update_state(self.state, self.gradient)
            P = self.opt.P(self.state)
        with timer("eig"):
            self.eff_eigs, _, eig_logs = self.eig_manager.get(self.w, P=P)
        return dict(times=timer.times, eig_logs=eig_logs)

    def step(self):
        P = self.opt.P(self.state)
        self.w -= P(self.gradient)


@dataclass
class MidpointProcessConfig:
    # optional process-specific configs for eigenvalue computation.
    # settings here will override those in the global eig config
    eig: EigConfigOverride = field(default_factory=EigConfigOverride)


@dataclass
class MidpointProcess(Process):
    """Run the discrete optimization algorithm, but view the second-order
    'midpoints' (i.e. the midpoints of midpoints) as the real iterates.
    
    The second-order midpoint at step t is:
         w^star_t := (1/4) [2 w_t + w_{t+1} + w_{t-1} ]
    """
    
    config: MidpointProcessConfig
    
    w_prev: Array = field(init=False, default=None)

    def __post_init__(self, global_eig_config: EigConfig):
        # override global eig config with non-None entries of the process-specific eig config
        eig_config = apply_overrides(global_eig_config, self.config.eig)
        super().__post_init__(eig_config)

    def prepare(self):
        # self.w = w_t, self.w_prev = w_{t-1}
        timer = Timer()
        with timer("grad_and_value"):
            self.gradient, self.loss = self.loss_fn.grad_and_value(self.w)
            self.state = self.opt.update_state(self.state, self.gradient)
        P = self.opt.P(self.state)
        self.w_next = self.w - P(self.gradient)  # self.w_next = w_{t+1}
        w_curr = self.w  # w_curr = w_t
        if self.w_prev is not None:
            # set self.w to the midpoint of midpoints w^\star_t
            self.w = (self.w_prev + 2 * self.w + self.w_next) / 4
            self.gradient = self.loss_fn.D(self.w)
        # store w_t in self.w_prev for next prepare
        self.w_prev = w_curr
        with timer("eig"):
            self.eff_eigs, _, eig_logs = self.eig_manager.get(self.w, P=P)
        return dict(times=timer.times, eig_logs=eig_logs)

    @property
    def delta(self):
        # this only gives the correct answer if called between prepare() and step()
        # self.w_prev is storing w_t, self.w is storing w^\star_t
        # therefore this returns delta := w_t - w^\star_t
        return self.w_prev - self.w

    def step(self):
        # set self.w to w_{t+1}
        self.w = self.w_next

    # need to override parent class implementation so that we also save w_prev
    def to_dict(self) -> Dict:
        out = dict(w=self.w, state=self.state)
        if self.w_prev is not None:
            out["w_prev"] = self.w_prev
        return out

    # need to override parent class implementation so that we also load w_prev
    def from_dict(self, d: Dict):
        self.w = torch.tensor(d["w"], dtype=self.w.dtype, device=self.w.device)
        self.state = torch.tensor(
            d["state"], dtype=self.state.dtype, device=self.state.device
        )
        if "w_prev" in d:
            self.w_prev = torch.tensor(
                d["w_prev"], dtype=self.w.dtype, device=self.w.device
            )
        else:
            self.w_prev = None

@dataclass
class StableFlowConfig:  
    # To discretize the stable flow, we dynamically adapt the 
    # number of substeps based on the current effective
    # sharpness, but we never allow it to be below this number.
    min_nsubsteps: int = 4

    # We stop trying to run the stable flow when the
    # effective sharpness exceeds this value.
    max_Seff: Optional[float] = 100.0

    # optional process-specific configs for eigenvalue computation.
    # settings here will override those in the global eig config
    eig: EigConfigOverride = field(default_factory=EigConfigOverride)


@dataclass
class StableFlow(Process):
    """Discretize the stable flow (i.e. for gradient descent, the gradient flow).
    
    To discretize the flow for one unit of time, we pick an integer 'nsubsteps' and do:
       for _ in range(nsubsteps):
          w <- w + (1/nsubsteps) * dw/dt
        
    The number of substeps must be set large enough that the flow remains stable, but
    making it larger increases the computation time.
    
    This implementation sets the nstepsteps to:
       max(config.min_nsubsteps, 2 * current effective sharpness).
       
    Note that setting nsubsteps to 0.5 * effective sharpness suffices to 
    ensure stability.  We use 2 rather than 0.5 to ensure that discretization effects
    are small, and to guard against the possibility that the effective sharpness has
    increased since we last measured it.
    
    If the effective sharpness reaches a certain user-specified limit (config.max_Seff),
    we stop discretizing the stable flow, as it would be too computationally
    expensive.
    """
    
    config: StableFlowConfig
    
    # whether the flow has been stopped
    _stopped: bool = field(init=False, default=False)
    
    # the current number of substeps used to
    # discretize the flow
    _current_nsubsteps: int = field(init=False, default=0)

    def __post_init__(self, global_eig_config: EigConfig):
        # the stable flow shouldn't dynamically track all eigenvalues of the effective Hessian
        # greater than some threshold, as the eigenvalues of the effective Hessian will
        # grow very large along the stable flow trajectory.
        global_eig_config = replace(global_eig_config, track_threshold=None)
        
        # override global eig config with non-None entries of the process-specific eig config
        eig_config = apply_overrides(global_eig_config, self.config.eig)
        
        super().__post_init__(eig_config)
        
        self._current_nsubsteps = self.config.min_nsubsteps
        
        # TODO comment why we do this
        self.state = self.opt.update_state(self.state, self.loss_fn.D(self.w))

    def prepare(self):
        timer = Timer()
        
        # if we've stopped running the flow, do nothing.
        if self._stopped:
            self.next = self.w, self.state
            return {}
        
        # compute preconditioner
        P = self.opt.P(self.state)
        
        # compute top eigenvalues of effective Hessian
        with timer("eig"):
            self.eff_eigs, _, eig_logs = self.eig_manager.get(self.w, P=P)
        Seff = self.eff_eigs[0]
        
        if self.config.max_Seff is not None:
            # if the effective sharpness exceeds config.max_Seff,
            # stop running the flow.
            if Seff > self.config.max_Seff:
                self._stopped = True
                self.next = self.w, self.state
                return {}
            
            # adjust nsubsteps based on the effective sharpness
            self._current_nsubsteps = max(self.config.min_nsubsteps, ceil(2 * Seff))

        # loop over substeps
        w, state = self.w, self.state
        with timer("substeps"):
            for i in range(self._current_nsubsteps):
                gradient, loss = self.loss_fn.grad_and_value(w)
                w, state = stable_flow_substep(
                    opt=self.opt,
                    w=w,
                    state=state,
                    gradient=gradient,
                    dt=1 / self._current_nsubsteps,
                )
                # keep only the loss and gradient at the first substep
                if i == 0:
                    self.gradient = gradient
                    self.loss = loss
        self.next = w, state
        return dict(times=timer.times, substeps=self._current_nsubsteps, eig_logs=eig_logs)

    def step(self):
        self.w, self.state = self.next


@dataclass
class CentralFlowConfig:
    # To discretize the central flow, we take this many 
    # "substeps" per unit of time.  In other words, we use
    # Euler's method with `dt = 1/nsubsteps`.
    nsubsteps: int = 4
    
    # All tracked eigenvalues of the effective Hessian that are
    # above this threshold will be considered potentially critical 
    # and will be thrown into the SDCP.
    sdcp_threshold: float = 1.95
    
    # If set to False (default), the eigenvalue computation frequency
    # will be interpreted in units of steps.  If set to True, it will
    # be interpreted in units of substeps.  Set this to True if you
    # need a higher frequency of eigenvalue computation.
    frequency_in_substeps: bool = False
    
    # optional process-specific configs for eigenvalue computation.
    # settings here will override those in the global eig config
    eig: EigConfigOverride = field(default_factory=EigConfigOverride)



@dataclass
class CentralFlow(Process):
    config: CentralFlowConfig
    
    # solves semidefinite complementarity problems (SDCPs)
    sdcp_colver: SDCPSolver = field(init=False)
    
    # Σ = UXUᵀ, where UᵀPU=I and X is diagonal
    U: Array = field(init=False, default=None)
    X: Array = field(init=False, default=None)
    
    def __post_init__(self, global_eig_config: EigConfig):
        # override global eig config with non-None entries of the process-specific eig config
        eig_config = apply_overrides(global_eig_config, self.config.eig)
        
        # if the eigenvalue computation frequency is given in units of steps,
        # change it to be in units of substeps
        if not self.config.frequency_in_substeps:
            eig_config = replace(
                eig_config,
                frequency=eig_config.frequency * self.config.nsubsteps,
            )
            
        super().__post_init__(eig_config)
        
        # initialize SDCP solver
        self.sdcp_colver = SDCPSolver()
        
        # TODO comment why we do this
        self.state = self.opt.update_state(self.state, self.loss_fn.D(self.w))

    def prepare(self):
        w, state = self.w, self.state
        times = defaultdict(lambda: 0.0)
        
        # loop over substeps
        for i in range(self.config.nsubsteps):
            timer = Timer()
            
            # compute gradient and loss
            with timer("grad_and_value"):
                gradient, loss = self.loss_fn.grad_and_value(w)
                
            # compute top eigenvalues and eigenvectors of effective Hessian
            with timer("eig"):
                # eff_U are chosen to be are orthonormal wrt P, i.e. uᵢᵀPuⱼ = δ[i=j]
                eff_eigs, eff_U, eig_logs = self.eig_manager.get(w, P=self.opt.P(state))
                
            # the number of eigenvalues that are nearly critical
            crit_k = (eff_eigs > self.config.sdcp_threshold).sum()
            
            # extract nearly-critical eigenvalues and eigenvectors
            crit_eigs, crit_U = eff_eigs[:crit_k], eff_U[:, :crit_k]
            
            # this represents Σ in the basis of crit_U, i.e. Σ = crit_U X crit_Uᵀ.
            X = torch.zeros(
                crit_k,
                crit_k,
                dtype=crit_U.dtype,
                device=crit_U.device,
            )
            
            if crit_k > 0:
                # compute d/dw u'H(w)v for all pairs (u, v) of critical eigenvectors
                with timer("dH"):
                    dH_dw = apply_to_pairs(
                        lambda u, v: self.loss_fn.D(w, 3, u, v), crit_U
                    )
                
                # take substep
                with timer("flow"):
                    w, state, flow_aux = central_flow_substep(
                        opt=self.opt,
                        w=w,
                        state=state,
                        gradient=gradient,
                        eigs=crit_eigs,
                        U=crit_U,
                        dH_dw=dH_dw,
                        dt=1 / self.config.nsubsteps,
                        sdcp_solver=self.sdcp_colver,
                    )
                    X = flow_aux["X"]
            # if no eigenvalues are critical, the central flow reduces to the stable flow
            else:
                timer.times["dH"] = 0.0
                with timer("flow"):
                    w, state = stable_flow_substep(
                        opt=self.opt,
                        w=w,
                        state=state,
                        gradient=gradient,
                        dt=1 / self.config.nsubsteps,
                    )
                    flow_aux = {}

            for k, v in timer.times.items():
                times[k] += v
            if i == 0:
                self.loss = loss
                self.gradient = gradient
                self.eff_eigs = eff_eigs

                # diagonalize X
                X_eigs, X_evecs = torch.linalg.eigh(X)
                X_eigs, X_evecs = X_eigs.flip(-1), X_evecs.flip(-1)
                
                # store generalized eigenvectors and eigenvalues of Σ wrt P
                self.X = X_eigs
                self.U = crit_U @ X_evecs
                
                flow_aux0 = flow_aux
                flow_aux0['eig_logs'] = eig_logs
                
        self.next = w, state
        return dict(times=times, flow_aux=flow_aux0)

    def step(self):
        self.w, self.state = self.next
        
    
class EigManager:
    """Manages eigenvalue computation for the processes.
    
    We don't always re-run the eigenvalue solver upon every call to .get(),
    Instead, we run it every `config.frequency` calls, and the rest of the time
    we use a poor man's approximation to the top eigenvalues: the quantities
    uᵢᵀH uᵢ where {uᵢ} are the most recently computed top eigenvectors.
    """
    
    def __init__(self, loss_fn: LossFunction, w_example: Array, config: EigConfig):
        self.loss_fn = loss_fn
        self.config = config
        
        # this counter increments upon each call to .get()
        self.counter = 0
        
        # cache for storing the eigenvalues and eigenvectors.
        # here, `symU` means the eigenvectors of P⁻½ H P⁻½, and `counter` records
        # the value of self.counter when we last ran the eigenvalue solver.
        self.cache = dict(symU=None, eigs=None, counter=None)
        
        self.solver = WarmStartEigSolver(
            loss_fn=loss_fn, w_example=w_example,
            initial_neigs=config.initial_neigs, return_sym_evecs=True,
            solver=config.solver, tol=config.tol,
            chunk_size=config.chunk_size, track_threshold=config.track_threshold
        )
        
    def get(self, w: Array, P: Optional[Preconditioner] = None):
        if self.config.frequency > 0:
            # Only compute the square root when we need it
            P_inv_sqrt = (lambda x: x) if P is None else P.sqrt()

        # if frequency == -1, never compute the eigenpairs
        if self.config.frequency <= 0:
            eigs, U, log = None, None, {}
        # if this is the first call to .get(), or if it's been config.frequency
        # calls to .get() since we last ran the eigenvalue solver, then
        # re-run the eigenvalue solver.
        elif self.cache["counter"] is None or (
            self.counter >= self.cache["counter"] + self.config.frequency
        ):        
            eigs, symU, log = self.solver.update(w, P=P)
            self.cache.update(symU=symU, eigs=eigs, counter=self.counter)
            U = vmap(P_inv_sqrt, 1, 1)(symU) # convert to right evecs of P^{-1} H
        # otherwise, compute and return the poor man's approximation to the top
        # eigenvalues of the effective Hessian. 
        else:
            symU = self.cache["symU"]
            hessian_diag = vmap(lambda u: self.loss_fn.D(w, 2, P_inv_sqrt(u), P_inv_sqrt(u)), 1)(symU)
            # TODO do we really need this?
            order = torch.argsort(hessian_diag, descending=True)
            eigs, symU = hessian_diag[order], symU[:, order]
            self.cache.update(eigs=eigs, symU=symU)
            U = vmap(P_inv_sqrt, 1, 1)(symU) # convert to right evecs of P^{-1} H
            log = {}

        self.counter += 1
        return eigs, U, log
    
