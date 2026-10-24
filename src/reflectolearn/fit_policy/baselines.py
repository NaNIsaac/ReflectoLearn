"""Baselines B (random actions) and C (everything at once with BIC model selection), and the two
references C_oracle and C_refine (DESIGN.md sections 5.2–5.3)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from reflectolearn.fit_policy import stack as st
from reflectolearn.fit_policy.env import BUDGET, FitEnv
from reflectolearn.fit_policy.stack import L_MAX, Stack
from reflectolearn.refine import RefineConfig, make_forward, refine

NO_STOP_FIRST = 5


def random_policy(env: FitEnv, rng: np.random.Generator) -> FitEnv:
    """B: uniform over valid actions; stop is masked for the first NO_STOP_FIRST actions."""
    while not env.done:
        actions = env.valid_actions()
        if len(env.trace) < NO_STOP_FIRST and len(actions) > 1:
            actions = [a for a in actions if a != ("stop",)]
        env.step(actions[rng.integers(len(actions))])
    return env


def _score(stack: Stack, q: np.ndarray, y: np.ndarray, scale: float, q_ho=None, y_ho=None) -> dict:
    dev = st.deviance(st.counts(stack, q, scale), y)
    out = {
        "layers": stack.layers.tolist(),
        "instr": stack.instr.tolist(),
        "n_layers": stack.n_layers,
        "bic": st.bic(dev, stack.n_layers, len(q)),
        "aic": st.aic(dev, stack.n_layers),
        "deviance": dev,
    }
    if q_ho is not None:
        out["deviance_ho"] = st.deviance(st.counts(stack, q_ho, scale), y_ho)
    return out


def full_fit(
    start: Stack,
    extra: np.ndarray,
    q: np.ndarray,
    y: np.ndarray,
    scale: float,
    *,
    q_ho=None,
    y_ho=None,
    budget: int = BUDGET,
) -> dict:
    """C: the start's layer after A1 steps 2–3 (estimate_rho, estimate_d) over extra[:L-1] for
    L = 1..L_MAX, every parameter free, Poisson deviance, budget / L_MAX evaluations each; the lowest
    BIC is returned. "capped" is whether the returned fit stopped at its evaluation cap (6.5)."""
    top = data_start(start, q, y, scale)
    fits = []
    for L in range(1, L_MAX + 1):
        stack = Stack(np.vstack([top.layers, extra[: L - 1]]), top.instr.copy())
        fits.append(
            st.fit(stack, np.ones(stack.n_params, bool), q, y, scale, "poisson", budget // L_MAX)
        )
    scores = [_score(f[0], q, y, scale, q_ho, y_ho) for f in fits]
    best = int(np.argmin([s["bic"] for s in scores]))
    return scores[best] | {
        "evaluations": sum(f[1] for f in fits),
        "jacobians": sum(f[2] for f in fits),
        "capped": fits[best][3] == 0,
        "capped_by_L": [f[3] == 0 for f in fits],
        "bic_by_L": [s["bic"] for s in scores],
    }


def data_start(start: Stack, q: np.ndarray, y: np.ndarray, scale: float) -> Stack:
    """A one-layer start after A1 steps 2–3, as C and C_oracle use it (amendment 3)."""
    return st.estimate_d(st.estimate_rho(start, q, y), q, y, scale)


CV_FINAL_FIT = 100


def inner_full_fit(
    start: Stack,
    extra: np.ndarray,
    q: np.ndarray,
    y: np.ndarray,
    scale: float,
    fit_mask: np.ndarray,
    *,
    q_ho=None,
    y_ho=None,
    budget: int = BUDGET,
) -> dict:
    """Cv (X2D.md section 4, amendment 1b and 1h): C on the training points outside the inner
    block, start estimates included, with (budget - 100) / L_MAX evaluations per L; the L with the
    lowest inner-block deviance is refit on all training points with one Poisson fit(100), A1v's
    step-9 refit, and returned."""
    qf, yf = q[fit_mask], y[fit_mask]
    top = data_start(start, qf, yf, scale)
    per_l = (budget - CV_FINAL_FIT) // L_MAX
    fits = []
    for L in range(1, L_MAX + 1):
        stack = Stack(np.vstack([top.layers, extra[: L - 1]]), top.instr.copy())
        fits.append(st.fit(stack, np.ones(stack.n_params, bool), qf, yf, scale, "poisson", per_l))
    inner = [st.deviance(st.counts(f[0], q[~fit_mask], scale), y[~fit_mask]) for f in fits]
    chosen = int(np.argmin(inner))
    final = st.fit(
        fits[chosen][0],
        np.ones(fits[chosen][0].n_params, bool),
        q,
        y,
        scale,
        "poisson",
        CV_FINAL_FIT,
    )
    return _score(final[0], q, y, scale, q_ho, y_ho) | {
        "evaluations": sum(f[1] for f in fits) + final[1],
        "jacobians": sum(f[2] for f in fits) + final[2],
        "capped": fits[chosen][3] == 0,
        "capped_by_L": [f[3] == 0 for f in fits],
        "inner_by_L": inner,
    }


def truth_fit(
    truth: Stack,
    q: np.ndarray,
    y: np.ndarray,
    scale: float,
    *,
    q_ho=None,
    y_ho=None,
    budget: int = BUDGET,
) -> dict:
    """TF (X2D.md section 4, diagnostic): the truth as the start, every parameter free, Poisson,
    one fit with max_nfev = budget, the fit returned."""
    fitted, nfev, njev, status = st.fit(
        truth, np.ones(truth.n_params, bool), q, y, scale, "poisson", budget
    )
    return _score(fitted, q, y, scale, q_ho, y_ho) | {
        "evaluations": nfev,
        "jacobians": njev,
        "capped": status == 0,
    }


def oracle_fit(
    start: Stack,
    extra: np.ndarray,
    n_true: int,
    q: np.ndarray,
    y: np.ndarray,
    scale: float,
    *,
    q_ho=None,
    y_ho=None,
    budget: int = BUDGET,
) -> dict:
    """C_oracle: C's start, the true layer count, every parameter free, Poisson, the whole budget.
    Uses the truth; reported, never in a verdict."""
    top = data_start(start, q, y, scale)
    stack = Stack(np.vstack([top.layers, extra[: n_true - 1]]), top.instr.copy())
    fitted, nfev, njev, status = st.fit(
        stack, np.ones(stack.n_params, bool), q, y, scale, "poisson", budget
    )
    return _score(fitted, q, y, scale, q_ho, y_ho) | {
        "evaluations": nfev,
        "jacobians": njev,
        "capped": status == 0,
    }


REFINE_RANGES = SimpleNamespace(thickness=(10.0, 1200.0), roughness=(0.0, 15.0), sld=(5.0, 100.0))


def refine_fit(start: Stack, q: np.ndarray, y: np.ndarray, scale: float, *, q_ho=None, y_ho=None):
    """C_refine, single-layer class only: `refine` with RefineConfig() from the start, on the
    training points with counts > 0. Its model (film + top oxide, Kα2, footprint) is not the stack
    model, so only deviances and the film parameters are reported."""
    ok = y > 0
    substrate = torch.tensor([[st.SUB_SLD, st.SUB_ROUGH]])
    config = RefineConfig()
    p, info = refine(
        torch.tensor(start.layers[:1], dtype=torch.float32),
        substrate,
        torch.as_tensor(q, dtype=torch.float32),
        torch.as_tensor(np.log10(np.maximum(y, 1) / scale), dtype=torch.float32)[None],
        torch.as_tensor(ok, dtype=torch.float32)[None],
        REFINE_RANGES,
        wavelength=st.WAVELENGTH,
        config=config,
        device=torch.device("cpu"),
    )

    def lam(qq: np.ndarray) -> np.ndarray:
        simulate = make_forward(
            torch.as_tensor(qq, dtype=torch.float32),
            substrate,
            wavelength=st.WAVELENGTH,
            config=config,
            device=torch.device("cpu"),
        )
        with torch.no_grad():
            return scale * simulate(info["state"], info["active"])[0].double().numpy()

    out = {
        "layers": p.double().numpy().tolist(),
        "n_layers": 1,
        "deviance": st.deviance(lam(q), y),
    }
    if q_ho is not None:
        out["deviance_ho"] = st.deviance(lam(q_ho), y_ho)
    return out
