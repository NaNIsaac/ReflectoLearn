"""Baseline A: the scripted human protocol A1 and its variants A2–A4 (DESIGN.md section 5.1).

The script acts only through FitEnv actions. Candidates in step 7 run on forks that share the
episode's budget and best-state record (amendment 2).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reflectolearn.fit_policy.env import FitEnv
from reflectolearn.fit_policy.stack import LAYER_KEYS

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
    fork = env.fork()
    fork.step(action)
    _refit(fork, p)
    return fork


def run(env: FitEnv, p: Protocol = A1) -> FitEnv:
    """Run protocol p on a fresh episode. Returns the env of the kept path; env.result() of it (or
    of any fork, they share the episode) gives the returned model."""
    env.kept_path = p.kept_path
    # 1. scale and background
    _do(env, ("loss", p.first_loss()), ("open", "bkg"), ("fit", 20))
    # 2–3. critical edge -> SLD, Kiessig fringes -> thickness
    if p.estimate:
        _do(env, ("estimate_rho",), ("estimate_d",))
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
