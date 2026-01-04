from typing import Any, Tuple

import torch
from torch.func import grad, vmap

from src.sdcp import SDCPSolver
from src.update_rules import UpdateRule
from src.utils import D, apply_to_pairs, mat_to_upper, upper_to_mat, get_upper_to_mat_metric

Array = Any

"""Numerical code for discretizing stable and central flows."""


def stable_flow_substep(
    opt: UpdateRule,
    w: Array,
    state: Array,
    gradient: Array,
    dt: float,
) -> Tuple[Array, Array]:
    """Perform one substep of stable flow"""
    # compute preconditioner
    P = opt.P(state)

    # compute time derivatives of w and state
    dstate_dt = opt.dstate_dt(state, gradient)
    dw_dt = -P(gradient)

    # take euler step
    new_state = state + dt * dstate_dt
    new_w = w + dt * dw_dt

    return new_w, new_state


def central_flow_substep(
    opt: UpdateRule,
    w: Array,
    state: Array,
    gradient: Array,
    eigs: Array,
    U: Array,
    dH_dw: Array,
    dt: float,
    sdcp_solver: SDCPSolver,
) -> Tuple[Array, Array, Array]:
    """Perform one substep of central flow"""
    P = opt.P(state)  # get current preconditioner P

    # decompose dw_dt = dw_dt_stable + dw_dt_X @ X
    # dw_dt_stable does not depend on X, dw_dt_X is the part that multiplies X
    dw_dt_stable = -P(gradient)  #       NOTE: [dim(w)]
    dw_dt_X = -0.5 * vmap(P)(dH_dw).T  # NOTE: [dim(w), binom(k+1,2)]

    # decompose dstate_dt = dstate_dt_stable + dstate_dt_X @ X
    # see helper function compute_dstate_dt for details
    HU = (
        vmap(P, 1, 1)(U) * eigs[None, :]
    )  # U are generalized evecs, i.e. H U = P U diag(eigs)
    # TODO: using this version of HU can lead to \beta not being symmetric, either need to replace with HU≈2PU or allow for non-symmetric \beta
    dstate_dt_stable, dstate_dt_X = compute_dstate_dt(opt, state, gradient, HU)

    # let A(w,state) := 2P(state)-H(w), so stability requires A(w,state) ⪰ 0
    # we approximate A at time t+dt by its linearization: next_A = A + dt * dA_dt ⪰ 0
    # we further approximate by restricting to span[U]: Uᵀ (A + dt * dA_dt) U ⪰ 0
    # to compute Uᵀ dA_dt U, we compute Uᵀ dP_dt U and Uᵀ dH_dt U separately

    # decompose dH_dt = dH_dt_stable + dH_dt_X @ X
    # dH_dt_stable does not depend on X, dH_dt_X is the part that multiplies X
    # note that dH_dt really represents d(Uᵀ H U)/dt ∈ Sym(ℝᵏ)
    dH_dt_stable = dH_dw @ dw_dt_stable  # NOTE: [binom(k+1,2)]
    dH_dt_X = dH_dw @ dw_dt_X  #           NOTE: [binom(k+1,2), binom(k+1,2)]

    # decompose dP_dt = dP_dt_stable + dP_dt_X @ X
    # dP_dt_stable does not depend on X, dP_dt_X is the part that multiplies X
    # note that dP_dt really represents d(Uᵀ P U)/dt ∈ Sym(ℝᵏ)
    # we begin by computing dP_dstate = ∇_{state} Uᵀ P(state) U
    dP_dstate_bilinear_form = lambda u, v: grad(lambda s: u @ opt.P(s)(v))(
        state
    )  # u,v → ∇_{state} uᵀ P(state) v
    dP_dstate = apply_to_pairs(
        dP_dstate_bilinear_form, U
    )  # NOTE: [binom(k+1,2), dim(state)]
    dP_dt_stable = dP_dstate @ dstate_dt_stable  # NOTE: [binom(k+1,2)]
    dP_dt_X = dP_dstate @ dstate_dt_X  #           NOTE: [binom(k+1,2), binom(k+1,2)]

    # before computing the next A, we need to compute the current A
    # because Uᵀ P U = Iₖ and HU = PUdiag(eigs), we have UᵀHU = diag(eigs)
    # therefore Uᵀ A U = Uᵀ(2P-H)U = diag(2-eigs)
    A = mat_to_upper(torch.diag(2 - eigs))

    # decompose dA_dt = dA_dt_stable + dA_dt_X @ X using A = 2P - H
    dA_dt_stable = 2 * dP_dt_stable - dH_dt_stable  # NOTE: [binom(k+1,2)]
    dA_dt_X = 2 * dP_dt_X - dH_dt_X  #                NOTE: [binom(k+1,2), binom(k+1,2)]

    # decompose next_A = (A + dt * dA_dt) into next_A = next_A_stable + next_A_X @ X
    next_A_stable = A + dt * dA_dt_stable # NOTE: [binom(k+1,2)]
    next_A_X = dt * dA_dt_X               # NOTE: [binom(k+1,2), binom(k+1,2)]

    # solve for X so that 0 ≼ X ⟂ next_A ⪰ 0 using an SDCP solver
    alpha = upper_to_mat(next_A_stable).cpu().numpy() # NOTE: [k,k]
    beta = upper_to_mat(next_A_X).cpu().numpy()       # NOTE: [k,k,k,k]
    X = sdcp_solver(A=alpha, B=beta) # solve SDCP (CVXPY)
    X = torch.from_numpy(X).float().cuda()  # convert to torch tensor
    X_vech = mat_to_upper(X)  # convert to upper triangular form

    # we are currently representing dw_dt_X, dstate_dt_X, X as vectors of upper triangular entries
    # to contract them correctly, we need to count the off-diagonal entries twice
    # this is equivalent to using a metric which puts a weight of 2 on the off-diagonal entries
    metric = get_upper_to_mat_metric(U.shape[1])  #                   NOTE: [binom(k+1,2)]
    dw_dt = dw_dt_stable + dw_dt_X @ (metric * X_vech)  #             NOTE: [dim(w)]
    dstate_dt = dstate_dt_stable + dstate_dt_X @ (metric * X_vech)  # NOTE: [dim(state)]

    # take a substep of size dt
    new_w = w + dt * dw_dt
    new_state = state + dt * dstate_dt

    aux = dict(
        dH_dt_stable=dH_dt_stable,
        dH_dt_X=dH_dt_X,
        dP_dt_stable=dP_dt_stable,
        dP_dt_X=dP_dt_X,
        X=X,
    )
    return new_w, new_state, aux


def compute_dstate_dt(
    opt: UpdateRule, state: Array, gradient: Array, HU: Array
) -> Tuple[Array, Array]:
    """
    Computes dstate_dt ≈ 𝔼[f(state,g)] where f = opt.dstate_dt and g is the gradient at w + δ

    Derivation:
        1. If δ has covariance Σ = UXUᵀ, the gradient has covariance (HU)X(HU)ᵀ
        2. Taylor expand f around 𝔼[g]:
            𝔼[f(state,g)] ≈ f(state,g) + 0.5 * ∇²_g f(state,g) @ [(HU)X(HU)ᵀ]
        3. Suffices to compute the bilinear form u,v → uᵀ [0.5 * ∇²_g f(state,g)] v applied to HU

    Returns:
        - dstate_dt_stable: the part of dstate_dt that does not depend on X NOTE: [dim(state)]
        - dstate_dt_X: the part of dstate_dt that multiplies X              NOTE: [dim(state), binom(k+1,2)]
    """
    dstate_dt_stable = opt.dstate_dt(state, gradient)  # NOTE: [dim(state)]
    dstate_dt_X_bilinear_form = lambda u, v: 0.5 * D(
        lambda g: opt.dstate_dt(state, g), gradient, 2, u, v
    )
    dstate_dt_X = apply_to_pairs(
        dstate_dt_X_bilinear_form, HU
    ).T  # NOTE: [dim(state), binom(k+1,2)]
    return dstate_dt_stable, dstate_dt_X


