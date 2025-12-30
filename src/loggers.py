from dataclasses import dataclass
from typing import Any, Literal, Dict

import torch
from torch.func import vmap

from .datasets import Dataset
from .functional import FunctionalModel
from .loss_function import dataloop
from .processes import CentralFlow, Process
Array = Any

"""Loggers which record experimental data.

This code is tightly integrated with the processes.py code (to the point
where it probably breaks/stretches the principle of software modularity).
Basically the only reason why it is is a different file is that otherwise
the processes.py file would be too long.
"""

class GroupLogger:
    """These loggers log some collective property of a set of processes
    (e.g. the mutual distances between the iterates of all processes). """
    
    def log(self, processes: Dict[str, Process]) -> Dict[str, Any]:
        raise NotImplementedError


class ProcessLogger:
    """"These loggers log some property of an individual process."""
    
    def log(self, process: Process) -> Dict[str, Any]:
        raise NotImplementedError


class DistanceLogger(GroupLogger):
    """Logs the pairwise distances between the iterates of all processes."""
    
    def log(self, processes: Dict[str, Process]) -> Dict[str, Any]:
        out = {}
        names = list(sorted(processes.keys()))
        n_opt = len(names)
        for i in range(n_opt): # iterate through pairs of processes
            for j in range(i):
                a, b = processes[names[i]], processes[names[j]]
                out[names[j] + "-" + names[i]] = (a.w - b.w).norm()
        out = dict(distances=out)
        return out


class EmpiricalDeltaLogger(GroupLogger):
    """Logs the delta between the central flow and the other processes, along the
    central flow's top Hessian eigenvectors."""
    
    def log(self, processes: Dict[str, Process]) -> Dict[str, Any]:
        out = dict()

        # TODO comment this if we keep it
        if "central" in processes and "midpoint" in processes:
            central, midpoint = processes["central"], processes["midpoint"]
            delta = midpoint.delta
            P = central.opt.P(central.state)
            delta = central.U.T @ P(delta)
            out.update(midpoint_deltas=delta)  # TODO: better name for this

        if "central" in processes and "discrete" in processes:
            central, discrete = processes["central"], processes["discrete"]
            delta = discrete.w - central.w

            # we model wₜ = w(t) + δₜ where 𝔼[δₜ]=0 and 𝔼[δₜδₜᵀ]=Σ
            # we store Σ = UXUᵀ, where UᵀPU=I, HU=2PU, and X is diagonal
            # therefore, we have:
            # 𝔼[(UᵀPδₜ) (UᵀPδₜ)ᵀ] = Uᵀ P Σ P U = UᵀPU X UᵀPU = X
            P = central.opt.P(central.state)
            delta = central.U.T @ P(delta)
            # TODO: rename this empirical variance
            out.update(deltas=delta)

        return out


@dataclass
class LossAndAccuracy(ProcessLogger):
    """Logs the loss and accuracy on the train or test split."""
    
    model: FunctionalModel          # the model
    data: Dataset                   # the dataset
    split: Literal["train", "test"] # the dataset split to use

    def batch_loss_acc(self, w, batch):
        x, y = batch
        out = self.model.apply(w, x)
        loss = self.data.criterion_fn(out, y)
        acc = torch.tensor(-1.0)
        if self.data.accuracy_fn is not None:
            acc = self.data.accuracy_fn(out, y)
        return loss, acc

    def log(self, process: Process) -> Dict[str, Any]:
        w = process.w
        if self.split == "train":
            dataset = self.data.trainset
        elif self.split == "test":
            dataset = self.data.testset
        else:
            raise Exception("Invalid split")
        loss, acc = dataloop(lambda b: self.batch_loss_acc(w, b), dataset)
        return {
            self.split + "_loss": loss,
            self.split + "_acc": acc,
        }
        

class OutputLogger(ProcessLogger):
    """Logs the network's output on a sample of train and test examples."""
    
    def __init__(self, model, data, n_samples=10):
        self.model = model
        self.data = data
        self.n_samples = n_samples

    def log(self, process: Process) -> Dict[str, Any]:
        w = process.w
        train_examples = self.data.trainset[0][0][: self.n_samples]
        train_outputs = self.model.apply(w, train_examples)
        test_examples = self.data.testset[0][0][: self.n_samples]
        test_outputs = self.model.apply(w, test_examples)
        return dict(
            train_outputs=train_outputs,
            test_outputs=test_outputs,
        )


class CentralFlowPredictionLogger(ProcessLogger):
    """Logs the central flow predictions for time-averaged loss
    and gradient norm along the discrete trajectory."""
    
    def log(self, process: Process) -> Dict[str, Any]:
        if not isinstance(process, CentralFlow):
            return {}

        loss, gradient = process.loss, process.gradient
        
        # we model wₜ = w(t) + δₜ where 𝔼[δₜ]=0 and 𝔼[δₜδₜᵀ]=Σ
        # we store Σ = UXUᵀ, where UᵀPU=I, HU=2PU, and X is diagonal
        
        P = process.opt.P(process.state)
        HU = 2 * vmap(P, 1, 1)(process.U)
        
        # TODO consider changing the above ΛPU rather than 2PU
        # although it shouldn't matter due to complementarity

        # 𝔼[L(w+δ)] = L(w) + 0.5〈H(w), Σ〉= L(w) + 0.5 *〈2I, X〉= L(w) + tr(X)
        predicted_loss = loss + process.X.sum()

        # 𝔼‖∇L(w+δ)‖² = ‖∇L(w)‖² + 𝔼‖Hδ‖² = ‖∇L(w)‖² + tr(HHΣ) =  ‖∇L(w)‖² + tr(HU X (HU)ᵀ)
        grad_norm_sq = gradient.square().sum()
        predicted_grad_norm_sq = grad_norm_sq + process.X @ HU.square().sum(0)
        
        # 𝔼[∇L(w+δ)⊙²] = ∇L(w)⊙² + 𝔼[H(w)δ]⊙² = ∇L(w)⊙² + diag[HΣH]
        #              = ∇L(w)⊙² + diag[HU X (HU)ᵀ]
        predicted_grad_sq = gradient.square() + (HU.square()@torch.diag(process.X)).sum(1).detach()
        
        # ideally, these indices would be forced to be the same ones as
        # are returned by the rmsprop summarizer
        selected_idx = torch.linspace(0, len(process.gradient) - 1, 25, dtype=int)
        predicted_grad_sq_selected_idx = predicted_grad_sq[selected_idx]

        # tr(Σ) = tr(∑ Xᵢᵢuᵢuᵢᵀ) = ∑ Xᵢᵢ ‖uᵢ‖²
        sigma_trace = process.X @ process.U.square().sum(0)

        return {
            "predicted_variances": process.X,
            "sigma_trace": sigma_trace,
            "predicted_grad_norm_sq": predicted_grad_norm_sq,
            "predicted_grad_sq_selected_idx": predicted_grad_sq_selected_idx,
            "predicted_loss": predicted_loss,
        }


class GradientLogger(ProcessLogger):
    """Logs the gradient norm."""
    
    def log(self, process: Process) -> Dict[str, Any]:
        # it could be None if it's the stable flow and it's already
        # stopped and we loaded from a checkpoint
        if process.gradient is None:
            return dict()
        
        # ideally, these indices would be forced to be the same ones as
        # are returned by the rmsprop summarizer
        selected_idx = torch.linspace(0, len(process.gradient) - 1, 25, dtype=int)
        
        grad_norm_sq = process.gradient.square().sum()
        grad_sq_selected_idx = process.gradient[selected_idx].square()
        
        return dict(grad_norm_sq=grad_norm_sq, grad_sq_selected_idx=grad_sq_selected_idx)


class EigLogger(ProcessLogger):
    """Logs the top eigenvalues of the effective Hessian."""
    
    def __init__(self):
        self.eig_manager = None

    def log(self, process: Process) -> Dict[str, Any]:
        eigs = process.eff_eigs
        return dict(effective_hessian_eigs=eigs)


class RawEigLogger(ProcessLogger):
    """Logs the top eigenvalues of the 'raw' Hessian. """
    def __init__(self):
        self.eig_manager = None

    def log(self, process: Process) -> Dict[str, Any]:
        eigs = process.eff_eigs
        # if there is an efficient way to compute the top eigenvalues of the
        # raw Hessian from those of the effective Hessian, then use it.
        if hasattr(process.opt, "raw_eigs_from_eigs"):
            raw_eigs = process.opt.raw_eigs_from_eigs(process.state, eigs)
            return dict(hessian_eigs=raw_eigs)
        else: # otherwise, use the processes's raw_eig_manager to solve for the raw eigs
            if process.raw_eig_manager is not None:
                process.raw_eig_manager.solver.n_eigs = process.eig_manager.solver.n_eigs
                raw_eigs = process.raw_eig_manager.get(process.w)[0]
                return dict(hessian_eigs=raw_eigs)
            else:
                return dict()
            

class StateLogger(ProcessLogger):
    """Logs the optimizer state."""
    
    def log(self, process: Process) -> Dict[str, Any]:
        log = process.opt.summarize_state(process.state)
        return {"opt."+k : v for k,v in log.items()}
