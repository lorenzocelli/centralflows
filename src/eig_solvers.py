from typing import Any, Optional, Tuple, Dict

import numpy as np
import torch
from scipy.sparse import linalg as spla
from torch.func import vmap

from src.update_rules import Preconditioner
from src.loss_function import LossFunction

from .lobpcg import torch_lobpcg

Array = Any

"""Eigenvalue solvers."""


def compute_eigs(loss_fn: LossFunction, w: Array, neigs: int = None, warm_start_eigenvectors: Array = None,
                P: Optional[Preconditioner] = None, return_sym_evecs: bool = False, solver: str = "lobpcg",
                tol: float = 1e-10, chunk_size=-1) -> Tuple[Array, Array, Dict]:
    """Computes top eigenvectors/eigenvalues of the Hessian (or preconditioned Hessian).
    
    If P is None, returns the top eigenpairs of the Hessian H(w).  If P is non-None,
    returns the top eigenpairs of the preconditioned Hessian P^{-1} H(w).
    
    In the latter case, if return_sym_evecs=True, it returns the eigenvectors of the 
    symmetric matrix P^{-1/2} H(w) P^{-1/2}.  If return_sym_evecs=False, it returns the 
    right eigenvectors of P^{-1} H(w).
        
    To specify the number of eigenpairs to compute, the user must pass in either `n_eigs` or `warm_start_eigenvectors`.
    In the former case, we randomly initialize the warmstart eigenvectors.  In the latter case, we use the provided ones.
    
    Args:
      loss_fn: the loss function
      w: the weights at which to compute the Hessian eigenvectors
      neigs: if set to non-None, compute this many eigs
      warm_start_eigenvectors: if set to non-None, warm start the solver from this array of eigenvectors
      P: the optional preconditioner
      return_sym_evecs: if true, return eigenvectors of P^{-1/2} H P^{-1/2}; otherwise, return right eigenvectors of P^(-1) H
      solver: either 'lobpcg' (runs custom lobpcb implementation) or 'scipy' (runs scipy built-in lanzcos)
      tol: the numerical tolerance for the eigenvalue solver
      chunk_size: the max number of HVP's that we attempt to do on-device concurrently (-1 means no bound)
      return_sym_evecs: if true, return eigenvectors of P^{-1/2} H P^{-1/2}; otherwise, return right eigenvectors of P^(-1) H
                
    Returns:
      eigenvalues (Array)
      eigenvectors (Array)
      matvec_count (int): the number of matrix-vector products that were used
    """
    assert neigs is not None or warm_start_eigenvectors is not None
    
    if warm_start_eigenvectors is None:
        warm_start_eigenvectors = _initialize_eigenvectors(len(w), neigs, w)

    P_0, P_1 = (lambda x: x, lambda x: x) if P is None else P.decompose()

    matvec_count = 0

    def matvec(v):
        nonlocal matvec_count
        matvec_count += np.prod(v.shape[1:])        
        _matvec = lambda v: P_0(loss_fn.D(w, 2, P_1(v)))
        _matvec = torch.func.vmap(_matvec, 1, 1, chunk_size=chunk_size)
        return _matvec(v)
    
    # switch from -1 convention (our code) to None convention (PyTorch)
    if chunk_size == -1:
        chunk_size = None
        
    log = {}

    if solver == "lobpcg":
        eigenvalues, eigenvectors, n_iter = torch_lobpcg(matvec, warm_start_eigenvectors, tol=tol)
        log['n_iter'] = n_iter
    elif solver == "scipy":
        eigenvalues, eigenvectors = scipy_eig(matvec, warm_start_eigenvectors)
    else:
        raise Exception(f"unknown solver {solver}, must be lobpcg or scipy")
    
    log['matvec_count'] = matvec_count
    
    if not return_sym_evecs:
        eigenvectors = vmap(P_0, 1, 1)(eigenvectors)
    
    return eigenvalues, eigenvectors, log


def _initialize_eigenvectors(nparams: int, n_eigs: int, array: Array):
    """Initialize eigenvectors for lobpcg iteration.
    
    Args:
      n_eigs: number of eigenvectors
      nparams: number of parameters in the net. i.e. dimension of each eigenvector
      array: an example array, which we use for its dtype and device
    """
    return torch.randn((nparams, n_eigs), dtype=array.dtype, device=array.device)


    
class WarmStartEigSolver:
    """A Hessian eigenvalue solver that warm-starts from the last computed eigenvectors.
    
    Initially, the solver tracks the top `n_eigs_initial` eigenpairs.  If track_threshold is not None,
    then whenever the lowest tracked eigenvalue goes above track_threshold, then we start tracking
    another eigenvalue, so as to ensure that we are always tracking all eigenvalues >= track_threshold.
    
    If you want to just track a fixed number of eigenvectors, then simply leave track_threshold as None.
    """
    
    def __init__(self, loss_fn: LossFunction, w_example: Array,
                 initial_neigs: int = 1, return_sym_evecs=False, solver: str = "lobpcg",
                 tol: float = 1e-10, chunk_size: int = -1, track_threshold: float = None):
        """Initialize the eigenvalue solver.
        
        Args:
          loss_fn: the loss function
          w_example: an example of a weight tensor (we will use this only for its shape information)
          initial_neigs: initial number of eigenvalues to track
          solver: either 'lobpcg' (runs custom lobpcb implementation) or 'scipy' (runs scipy built-in lanzcos)
          tol: the numerical tolerance for the eigenvalue solver
          return_sym_evecs: if true, return eigenvectors of the symmetric matrix P^{-1/2} H P^{-1/2};
                            otherwise, return the right eigenvectors of P^(-1) H
          chunk_size: max num of HVP's to try to do on the GPU at once
          track_threshold: if set to non-None, track all eigenvalues greater than this value
        """
        self.loss_fn = loss_fn
        self.solver = solver
        self.tol = tol
        self.return_sym_evecs = return_sym_evecs
        self.n_eigs = initial_neigs
        self.chunk_size = chunk_size
        self.track_threshold = track_threshold
        
        if initial_neigs == 0:
            raise ValueError("initial_neigs must be > 0")
        
        # Eigenvectors that we keep around for warm-starting the eigenvalue solver.
        # These are eigenvectors of the 'symmetric' preconditioned Hessian P^{-1/2} H P^{-1/2}.
        self.symU = torch.randn((len(w_example), 0), dtype=w_example.dtype, device=w_example.device)

    def update(self, w: Array, P: Optional[Preconditioner] = None) -> Tuple[Array, Array, Dict]:
        """Re-run the Hessian eigenvalue solver.
        
        Args:
          w: the new weights
          
        Returns:
          eigs: the new eigenvalues
          U: the new eigenvectors
          logs: logs for eigenvalue solving
        """
        # keep growing the number of estimated eigenvalues until the smallest
        # estimated eigenvalue is below self.track_threshold
        while True:
            
            # add columns to symU if self.n_eigs exceeds the current number of columns
            if self.n_eigs > self.symU.shape[1]:
                new_k = self.n_eigs - self.symU.shape[1]
                new_vecs = _initialize_eigenvectors(self.symU.shape[0], new_k, self.symU)
                self.symU = torch.cat([self.symU, new_vecs], dim=1)

            # compute the new eigenvalues and eigenvectors
            eigs, symU, log = compute_eigs(
                self.loss_fn, w, P=P, return_sym_evecs=self.return_sym_evecs, warm_start_eigenvectors=self.symU,
                solver=self.solver, tol=self.tol, chunk_size=self.chunk_size)

            # break if we aren't enforcing a track threshold
            # or our smallest estimated eigenvalue is below this threshold
            if (self.track_threshold is None) or (eigs.min() < self.track_threshold):
                break
        
            # increment number of eigenvalues to track
            self.n_eigs += 1
            print(f"tracking {self.n_eigs} eigenvalues now") 
            
        # align the new eigenvectors with the last eigenvectors (this only really works when there is 1 unstable eigenvector)
        self.symU = align_evecs(symU, self.symU)
            
        # convert to right eigenvectors if appropriate
        if self.return_sym_evecs:
            eigenvectors = self.symU
        else:
            P_0 = (lambda x: x) if P is None else P.decompose()[0]
            eigenvectors = vmap(P_0, 1, 1)(self.symU)
            
        return eigs, eigenvectors, log


def scipy_eig(matvec, U):
    """Wraps scipy's Lanczos eigenvalue solver."""
    dim, n_eig = U.shape
    dtype = U.dtype
    device = U.device

    def np_matvec(v):
        v = torch.tensor(v, dtype=U.dtype, device=U.device)
        return matvec(v).cpu().numpy()

    op = spla.LinearOperator((dim, dim), matvec=np_matvec, dtype=np.float32)
    eigs, U = spla.eigsh(op, k=n_eig, which="LA")
    eigs, U = eigs[::-1], U[:, ::-1]
    eigs = torch.tensor(eigs.copy(), dtype=dtype, device=device)
    U = torch.tensor(U.copy(), dtype=dtype, device=device)
    return eigs, U

"""Helper functions"""

def align_evecs(U, refU):
    signs = torch.sign(torch.einsum("ij,ij->j", U, refU))
    return torch.einsum("ij,j->ij", U, signs)
