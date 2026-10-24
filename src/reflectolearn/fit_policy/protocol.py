"""Baseline A: the scripted human protocol A1 and its variants A2–A4 (DESIGN.md section 5.1).

The script acts only through FitEnv actions. Candidates in step 7 run on forks that share the
episode's budget and best-state record (amendment 2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reflectolearn.fit_policy import stack as st
from reflectolearn.fit_policy.env import NEW_LAYER_SIGMA, RESIDUAL_Z, FitEnv
from reflectolearn.fit_policy.stack import LAYER_KEYS, Stack

EXCESS_SIGMAS = 3.0
MERGE_CONTRAST = 2.0


@dataclass(frozen=True)
class Protocol:
    name: str
    staged: bool = True  # False (A2): open everything before every fit, Poisson only
    finish_loss: str = "poisson"  # "log" (A3): the log loss throughout
    candidates: str = "all"  # "bottom" (A4): only add above the substrate
    accept: float = -10.0  # ΔBIC a candidate must reach to be kept
    # X2d (X2D.md section 4); all off is A1
    matched: bool = False  # compare candidates and merges with a refit of the current state
    kept_path: bool = False  # revert to and return the best state of the kept path
    inner: bool = False  # also require a lower deviance on the env's inner (unfitted) points
    estimate: bool = True  # False (A1o): skip steps 2-3, the start is the truth
    # X2e (X2E.md section 4); both off is A1
    s1: bool = False  # several screened starts for every added layer
    s2: bool = False  # screened stack starts from the data's fringe peaks before step 4

    def first_loss(self) -> str:
        return "log" if self.staged else "poisson"


A1 = Protocol("A1")
A2 = Protocol("A2", staged=False)
A3 = Protocol("A3", finish_loss="log")
A4 = Protocol("A4", candidates="bottom", accept=-2.0)
VARIANTS = {p.name: p for p in (A1, A2, A3, A4)}

A1m = Protocol("A1m", matched=True)
A1k = Protocol("A1k", kept_path=True)
A1mk = Protocol("A1mk", matched=True, kept_path=True)
A1v = Protocol("A1v", matched=True, kept_path=True, inner=True)  # env needs a fit_mask
A1o = Protocol("A1o", estimate=False)  # started at the truth
X2D_ARMS = {p.name: p for p in (A1, A1m, A1k, A1mk, A1v, A1o)}

S1 = Protocol("S1", s1=True)
S2 = Protocol("S2", s2=True)
S12 = Protocol("S12", s1=True, s2=True)  # primary
D3 = Protocol("D3", estimate=False)  # started at the true-L data start
X2E_ARMS = {p.name: p for p in (A1, S1, S2, S12, A1o, D3)}

S1_FACTORS = (1.0, 0.75, 1.25)  # rho of a new layer relative to A1's neighbour mean
S1_PEAKS = 3
SCREEN_STEPS = 20
S2_Z = np.arange(20.0, 801.0, 1.0)


def _do(env: FitEnv, *actions) -> None:
    """Apply the actions that are valid now; skip the rest (e.g. opening what is open)."""
    for a in actions:
        if env.done:
            return
        if env.valid(a):
            env.step(a)


def _open_all(env: FitEnv) -> None:
    _do(
        env,
        *(("open_all", g) for g in LAYER_KEYS),
        ("open", "bkg"),
        ("open", "dq"),
        ("open", "offset"),
    )


def _refit(env: FitEnv, p: Protocol) -> None:
    """Steps 7–8 refit of a changed stack: the new parameters with the first loss, then all with
    the finishing loss."""
    if p.staged:
        _do(env, ("loss", p.first_loss()), ("fit", 100))
    else:
        _open_all(env)
        _do(env, ("loss", "poisson"), ("fit", 100))
    _open_all(env)
    _do(env, ("loss", p.finish_loss), ("fit", 100))


def _candidates(env: FitEnv, p: Protocol) -> list[tuple]:
    L = env.state.stack.n_layers
    if p.candidates == "bottom":
        return [("add", L)]
    thickest = int(np.argmax(env.state.stack.layers[:, 0]))
    return [("add", 0), ("add", L), ("split", thickest)]


def _try(env: FitEnv, action: tuple, p: Protocol) -> FitEnv | None:
    if not env.valid(action):
        return None
    if p.s1 and action[0] == "add":
        fork = _s1_add(env, action)
    else:
        fork = env.fork()
        fork.step(action)
    _refit(fork, p)
    return fork


def _screen(env: FitEnv) -> None:
    _do(env, ("loss", "log"), ("fit", SCREEN_STEPS))


def _s1_add(env: FitEnv, action: tuple) -> FitEnv:
    """X2E.md 4.1: the new layer at each of the residual's three strongest peaks and three SLDs,
    each screened with fit(20); the best one is returned (refit by the caller as A1 refits)."""
    resid = np.log10(np.maximum(env.y, st.LOG_FLOOR_COUNTS)) - np.log10(env.lam)
    peaks = st.fringe_peaks(env.q, resid, 1.1 * env._qc(), RESIDUAL_Z, k=S1_PEAKS)
    starts = [(z, f) for z in peaks for f in S1_FACTORS] or [(None, 1.0)]
    pos = action[1]
    tried = []
    for z, factor in starts:
        fork = env.fork()
        fork.step(action)  # A1's new layer: strongest peak (or 50 Å), neighbour-mean SLD, σ 3 Å
        layers = fork.state.stack.layers.copy()
        if z is not None:
            layers[pos, 0] = z
        layers[pos, 1] = NEW_LAYER_SIGMA
        layers[pos, 2] *= factor
        fork.set_stack(Stack(layers, fork.state.stack.instr.copy()), fork.state.open_layers)
        _screen(fork)
        tried.append(fork)
    best = int(np.argmin([f.bic for f in tried]))
    env.episode.s1_changed.append(best != 0)
    return tried[best]


def _s2_stacks(env: FitEnv) -> list[Stack]:
    """X2E.md 4.2: stacks from the data's fringe peaks, read as interface depths in both orders."""
    base = env.state.stack
    values = np.log10(np.maximum(env.y, st.LOG_FLOOR_COUNTS))
    peaks = st.fringe_peaks(env.q, values, 1.1 * env._qc(), S2_Z)
    stacks = []
    for m in range(1, len(peaks) + 1):
        thickness = np.diff([0.0, *sorted(peaks[:m])])
        for order in (thickness, thickness[::-1]) if m > 1 else (thickness,):
            if order.min() < st.LAYER_BOUNDS["d"][0]:
                continue
            rho = [base.layers[0, 2]]
            for _ in range(m - 1):
                rho.append(0.5 * (rho[-1] + st.SUB_SLD))
            sigma = [base.layers[0, 1], *[NEW_LAYER_SIGMA] * (m - 1)]
            stacks.append(Stack(np.column_stack([order, sigma, rho]), base.instr.copy()))
    return stacks


def _s2_start(env: FitEnv) -> FitEnv:
    """X2E.md 4.2: screen A1's step-3 state and the peak stacks with fit(20) (d and rho open),
    keep the lowest BIC."""
    candidates = [env.state.stack, *_s2_stacks(env)]
    tried = []
    for stack in candidates:
        fork = env.fork()
        open_layers = np.zeros((stack.n_layers, 3), bool)
        open_layers[:, [LAYER_KEYS.index("d"), LAYER_KEYS.index("rho")]] = True
        fork.set_stack(stack, open_layers)
        _screen(fork)
        tried.append(fork)
    best = int(np.argmin([f.bic for f in tried]))
    env.episode.s2_changed = best != 0
    return tried[best]


def run(env: FitEnv, p: Protocol = A1) -> FitEnv:
    """Run protocol p on a fresh episode. Returns the env of the kept path; env.result() of it (or
    of any fork, they share the episode) gives the returned model."""
    env.kept_path = p.kept_path
    # 1. scale and background
    _do(env, ("loss", p.first_loss()), ("open", "bkg"), ("fit", 20))
    # 2–3. critical edge -> SLD, Kiessig fringes -> thickness
    if p.estimate:
        _do(env, ("estimate_rho",), ("estimate_d",))
    if p.s2:
        env = _s2_start(env)
    if p.staged:
        # 4. structure, 5. roughness last, 6. instrument last with the finishing loss
        _do(env, ("open_all", "d"), ("open_all", "rho"), ("fit", 100), ("fit", 100))
        _do(env, ("open_all", "sigma"), ("fit", 100))
        _do(env, ("open", "dq"), ("open", "offset"), ("loss", p.finish_loss))
        _do(env, ("fit", 100), ("fit", 100))
    else:
        _open_all(env)
        _do(env, ("loss", "poisson"), *([("fit", 100)] * 5))

    # 7–8. layers only while a residual remains; keep a candidate only if BIC drops enough
    while not env.done and env.budget_left > 0:
        _do(env, ("revert",))  # continue from the best state so far
        if env.excess() <= EXCESS_SIGMAS:
            break
        tried = [f for a in _candidates(env, p) if (f := _try(env, a, p)) is not None]
        if not tried:
            break
        best = min(tried, key=lambda f: f.bic)
        if p.matched:
            base = _refit_current(env, p)
            env.episode.decisions.append(
                [env.state.stack.n_layers, best.bic - env.bic, best.bic - base.bic]
            )
            if not _accepts(best, base, p):
                env = base  # continue from the refit state (X2D.md section 4)
                break
        elif best.bic - env.bic > p.accept:
            break
        env = _merge_pass(best, p)

    # 9. finish
    _do(env, ("revert",))
    if env.fit_mask is not None:
        env.use_all_points()  # A1v: the inner block rejoins
    _open_all(env)
    _do(env, ("loss", p.finish_loss), ("fit", 100), ("stop",))
    return env


def _refit_current(env: FitEnv, p: Protocol) -> FitEnv:
    """The current state refit on a fork exactly as a candidate is (A1m)."""
    base = env.fork()
    _refit(base, p)
    return base


def _accepts(candidate: FitEnv, base: FitEnv, p: Protocol) -> bool:
    ok = candidate.bic - base.bic <= p.accept
    return ok and (not p.inner or candidate.dev_inner < base.dev_inner)


def _merge_pass(env: FitEnv, p: Protocol) -> FitEnv:
    """Step 8: merge adjacent layers of nearly equal SLD when BIC does not rise. Matched arms
    compare with a refit of the current state and continue from it on rejection (amendment 1e)."""
    i = 0
    while i < env.state.stack.n_layers - 1 and not env.done:
        rho = env.state.stack.layers[:, 2]
        if abs(rho[i] - rho[i + 1]) < MERGE_CONTRAST:
            merged = _try(env, ("merge", i), p)
            if merged is not None and not p.matched and merged.bic <= env.bic:
                env = merged
                continue
            if merged is not None and p.matched:
                base = _refit_current(env, p)
                keep = merged.bic <= base.bic and (
                    not p.inner or merged.dev_inner <= base.dev_inner
                )
                env = merged if keep else base
                if keep:
                    continue
        i += 1
    return env
