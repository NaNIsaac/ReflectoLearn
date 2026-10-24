"""Experiment 001, R2c: why refine.mcmc intervals of d disagree with the profile likelihood.

Decision rules are pre-registered in docs/identifiability/mcmc_width.md. Per case, on the R2b curve
(drawn from refine's own forward model), under refine's model throughout:

    profile  Poisson profile likelihood of d on that curve: Δχ² = 1 crossings on both sides of the ML
    G        refine.mcmc.sample_posterior as it is (Gaussian in log10 R), 2000 steps
    G-long   the same, lengthened until the kept half is ≥ 50 integrated autocorrelation times
    P        emcee with the Poisson likelihood, same walkers, start and bounds, same convergence rule

Run:  uv run python experiments/001_identifiability/mcmc_width.py profile G G-long P
      uv run python experiments/001_identifiability/mcmc_width.py report
"""

from __future__ import annotations

import argparse
import json
import time

import emcee
import fim
import numpy as np
import torch
from nonunique import REFINE_INIT
from refine_causes import ALL_EFFECTS, BOUNDS, CASES, OUT, S_COUNTS, Q, curve, lsq_fit, model

from reflectolearn.refine.mcmc import LNSIGMA_BOUNDS, sample_posterior
from reflectolearn.refine.pipeline import OXIDE_KEYS, RefineConfig

RESULTS = OUT / "mcmc_width.json"
LENGTHS = (2000, 8000, 20000)
TAU_FACTOR = 50


def ml(case: str):
    _, free = CASES[case]
    p0, y, oxide = curve(case)
    p, dev, _ = lsq_fit(y, p0, free, oxide)
    return p, dev, y, oxide, free


# ------------------------------------------------------------------ profile on the noisy curve
def run_profile(case: str, ml_start: dict | None = None, rounds: int = 0) -> dict:
    """If a profile point falls more than 0.05 below the ML deviance, the least-squares ML was not
    the minimum: refit every parameter from that point and profile again from the new ML."""
    p_ml, d_ml, y, oxide, free = ml(case)
    if ml_start is not None:
        p_ml, d_ml, _ = lsq_fit(y, ml_start, free, oxide)
    others = tuple(k for k in free if k != "thickness")
    d_hat = p_ml["thickness"]
    points = [{"d": d_hat, "dchi2": 0.0}]
    lowest: list = [0.0, p_ml]

    def at(d: float, start: dict) -> tuple[dict, float]:
        """Lowest of three local fits at fixed d: continued from `start`, nuisances at refine's
        initial values, nuisances at their bound midpoints (amendment 1 in mcmc_width.md)."""
        nuis = [k for k in others if k in REFINE_INIT]
        best = None
        for s in (
            start,
            start | {k: REFINE_INIT[k] for k in nuis},
            start | {k: 0.5 * (BOUNDS[k][0] + BOUNDS[k][1]) for k in nuis},
        ):
            p, dev, _ = lsq_fit(y, s | {"thickness": d}, others, oxide)
            if best is None or dev < best[1]:
                best = (p, dev)
        p, dev = best
        points.append({"d": d, "dchi2": dev - d_ml})
        if dev - d_ml < lowest[0]:
            lowest[:] = [dev - d_ml, p]
        return p, dev - d_ml

    crossings = []
    for sign in (1, -1):
        step, inner, inner_p = 1e-4 * d_hat, 0.0, p_ml
        while True:  # step out by doubling until Δχ² > 1
            p, chi = at(d_hat + sign * step, inner_p)
            if chi > 1:
                outer = step
                break
            inner, inner_p, step = step, p, 2 * step
        for _ in range(6):  # bisect between the last point inside and the first outside
            mid = 0.5 * (inner + outer)
            p, chi = at(d_hat + sign * mid, inner_p)
            if chi <= 1:
                inner, inner_p = mid, p
            else:
                outer = mid
        crossings.append(0.5 * (inner + outer))
    if lowest[0] < -0.05 and rounds < 3:
        return run_profile(case, lowest[1], rounds + 1)
    points.sort(key=lambda r: r["d"])
    return {
        "ml_rounds": rounds,
        "ml_thickness": d_hat,
        "ml_deviance": d_ml,
        "ml_params": p_ml,
        "crossings": crossings,
        "half_width": float(np.mean(crossings)),
        "points": points,
    }


# ------------------------------------------------------------------ samplers
def tau_max(chain: np.ndarray) -> float:
    """Largest integrated autocorrelation time over parameters, chain (steps, walkers, dim)."""
    return float(np.max(emcee.autocorr.integrated_time(chain, quiet=True)))


def summary(chain: np.ndarray, keys: list[str], steps: int, seconds: float) -> dict:
    d = chain[:, :, keys.index("thickness")].reshape(-1)
    p16, p50, p84 = np.percentile(d, [16, 50, 84])
    tau = tau_max(chain)
    return {
        "steps": steps,
        "seconds": seconds,
        "tau_max": tau,
        "converged": chain.shape[0] >= TAU_FACTOR * tau,
        "d": {"lo": float(p16), "median": float(p50), "hi": float(p84)},
        "half_width": float((p84 - p16) / 2),
    }


def run_gaussian(case: str, steps: int) -> dict:
    p_ml, _, y, oxide, free = ml(case)
    mask = y > 0
    log_r = np.where(mask, np.log10(np.where(mask, y, 1.0) / S_COUNTS), -15.0)
    cfg = RefineConfig(fit_oxide=oxide)
    info = {
        "state": {k: torch.tensor([v], dtype=torch.float32) for k, v in p_ml.items()},
        "active": ALL_EFFECTS if oxide else ALL_EFFECTS - {"oxide"},
        "bounds": {k: BOUNDS[k] for k in free},
        "fitted": list(free),
    }
    t = time.perf_counter()
    post = sample_posterior(
        info,
        torch.tensor([[fim.SUB_SLD, fim.SUB_ROUGH]]),
        torch.as_tensor(Q, dtype=torch.float32),
        torch.as_tensor(log_r, dtype=torch.float32)[None],
        torch.as_tensor(mask, dtype=torch.float32)[None],
        wavelength=fim.WAVELENGTH,
        config=cfg,
        steps=steps,
    )
    seconds = time.perf_counter() - t
    # sample_posterior returns the kept half flattened step-major, without lnsigma
    chain = post["samples"].reshape(steps - steps // 2, post["walkers"], -1)
    return summary(chain, post["keys"], steps, seconds) | {"acceptance": post["acceptance"]}


def run_poisson(case: str, steps: int, seed: int = 0) -> dict:
    p_ml, _, y, oxide, free = ml(case)
    keys = [k for k in free if oxide or k not in OXIDE_KEYS]
    fwd = model(oxide)
    lo = np.array([BOUNDS[k][0] for k in keys])
    hi = np.array([BOUNDS[k][1] for k in keys])

    @torch.no_grad()
    def log_prob(theta: np.ndarray) -> np.ndarray:
        out = np.full(len(theta), -np.inf)
        ok = np.all((theta > lo) & (theta < hi), axis=1)
        if ok.any():
            rows = theta[ok]
            vals = {k: torch.full((len(rows),), v) for k, v in p_ml.items()}
            vals |= {
                k: torch.as_tensor(rows[:, i], dtype=torch.float32) for i, k in enumerate(keys)
            }
            lam = S_COUNTS * fwd(vals, ALL_EFFECTS).double().numpy()
            out[ok] = (y * np.log(lam) - lam).sum(1)
        return out

    # walkers and start ball as in refine.mcmc
    ndim = len(keys)
    nwalk = max(4 * ndim, 2 * ndim + 2)
    rng = np.random.default_rng(seed)
    x0 = np.array([p_ml[k] for k in keys])
    p0 = np.clip(
        x0 + (hi - lo) * 1e-3 * rng.standard_normal((nwalk, ndim)),
        lo + 1e-9 * (hi - lo),
        hi - 1e-9 * (hi - lo),
    )
    sampler = emcee.EnsembleSampler(nwalk, ndim, log_prob, vectorize=True)
    sampler.random_state = np.random.RandomState(seed)
    t = time.perf_counter()
    sampler.run_mcmc(p0, steps, progress=False)
    seconds = time.perf_counter() - t
    chain = sampler.get_chain(discard=steps // 2)
    return summary(chain, keys, steps, seconds) | {
        "acceptance": float(np.mean(sampler.acceptance_fraction))
    }


# ------------------------------------------------------------------ amendment 3: whitened DE chains
DE_LENGTHS = (5000, 15000)
DE_WALKERS = 64


def fisher_cholesky(p_ml: dict, keys: list[str], oxide: bool) -> np.ndarray:
    """Cholesky factor of F^-1, F the Poisson Fisher information at p_ml under refine's model,
    from a central-difference Jacobian of ln λ (one batched forward call)."""
    fwd = model(oxide)
    h = 3e-4 * np.array([BOUNDS[k][1] - BOUNDS[k][0] for k in keys])
    x0 = np.array([p_ml[k] for k in keys])
    rows = np.r_[x0 + np.diag(h), x0 - np.diag(h), x0[None]]
    vals = {k: torch.full((len(rows),), v) for k, v in p_ml.items()}
    vals |= {k: torch.as_tensor(rows[:, i], dtype=torch.float32) for i, k in enumerate(keys)}
    with torch.no_grad():
        lam = S_COUNTS * fwd(vals, ALL_EFFECTS).double().numpy()
    n = len(keys)
    J = ((np.log(lam[:n]) - np.log(lam[n : 2 * n])) / (2 * h[:, None])).T
    F = J.T @ (lam[-1][:, None] * J)
    return np.linalg.cholesky(np.linalg.inv(F))


def run_de(case: str, likelihood: str, steps: int, seed: int = 0) -> dict:
    """Amendment 3: emcee DE moves in coordinates whitened by the Poisson Fisher information."""
    _, _, y, oxide, free = ml(case)
    p_ml = load()[case]["profile"]["ml_params"]
    keys = [k for k in free if oxide or k not in OXIDE_KEYS]
    fwd = model(oxide)
    lo = np.array([BOUNDS[k][0] for k in keys])
    hi = np.array([BOUNDS[k][1] for k in keys])
    x_ml = np.array([p_ml[k] for k in keys])
    L = fisher_cholesky(p_ml, keys, oxide)
    mask = y > 0
    obs = np.log10(np.where(mask, y, 1.0) / S_COUNTS)
    n_valid = float(mask.sum())
    gauss = likelihood == "gaussian"

    @torch.no_grad()
    def r_of(theta: np.ndarray) -> np.ndarray:
        vals = {k: torch.full((len(theta),), v) for k, v in p_ml.items()}
        vals |= {k: torch.as_tensor(theta[:, i], dtype=torch.float32) for i, k in enumerate(keys)}
        return fwd(vals, ALL_EFFECTS).double().numpy()

    def log_prob(z: np.ndarray) -> np.ndarray:
        theta = x_ml + z[:, : len(keys)] @ L.T
        out = np.full(len(z), -np.inf)
        ok = np.all((theta > lo) & (theta < hi), axis=1)
        if gauss:
            ok &= (z[:, -1] > LNSIGMA_BOUNDS[0]) & (z[:, -1] < LNSIGMA_BOUNDS[1])
        if ok.any():
            r = r_of(theta[ok])
            if gauss:
                # refine.mcmc's likelihood: Gaussian in log10 R over the nonzero counts, spread e^lnσ
                sse = (((np.log10(np.clip(r, 1e-15, None)) - obs) ** 2) * mask).sum(1)
                sig = np.exp(z[ok, -1])
                out[ok] = -0.5 * sse / sig**2 - n_valid * np.log(sig)
            else:
                lam = S_COUNTS * r
                out[ok] = (y * np.log(lam) - lam).sum(1)
        return out

    ndim = len(keys) + gauss
    z0 = np.zeros(ndim)
    if gauss:
        sse0 = (((np.log10(np.clip(r_of(x_ml[None])[0], 1e-15, None)) - obs) ** 2) * mask).sum()
        z0[-1] = np.clip(np.log(max(np.sqrt(sse0 / n_valid), 1e-4)), *LNSIGMA_BOUNDS)
    rng = np.random.default_rng(seed)
    p0 = np.empty((DE_WALKERS, ndim))
    for i in range(DE_WALKERS):  # a ball of radius 0.1 in z, redrawn until inside the bounds
        while True:
            p0[i] = z0 + 0.1 * rng.standard_normal(ndim) / np.sqrt(ndim)
            if np.isfinite(log_prob(p0[i : i + 1])[0]):
                break
    sampler = emcee.EnsembleSampler(
        DE_WALKERS,
        ndim,
        log_prob,
        vectorize=True,
        moves=[(emcee.moves.DEMove(), 0.8), (emcee.moves.DESnookerMove(), 0.2)],
    )
    sampler.random_state = np.random.RandomState(seed)
    t = time.perf_counter()
    sampler.run_mcmc(p0, steps, progress=False)
    seconds = time.perf_counter() - t
    zc = sampler.get_chain(discard=steps // 2)
    chain = x_ml + zc[:, :, : len(keys)] @ L.T  # back to θ for τ and the interval
    return summary(chain, keys, steps, seconds) | {
        "acceptance": float(np.mean(sampler.acceptance_fraction))
    }


def until_converged(run, case: str, lengths=LENGTHS) -> dict:
    tries = []
    for steps in lengths:
        r = run(case, steps)
        tries.append(r)
        print(f"  {case} {steps} steps: tau {r['tau_max']:.0f}, h {r['half_width']:.3f} Å")
        if r["converged"]:
            break
    return tries[-1] | {
        "tries": [{k: t[k] for k in ("steps", "tau_max", "half_width")} for t in tries]
    }


# ------------------------------------------------------------------ report
def load() -> dict:
    return json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}


def report() -> None:
    data = load()
    lines = [
        "# R2c tables (generated by mcmc_width.py report; do not edit)",
        "",
        "h_p: profile Δχ² ≤ 1 half-width of d on the same noisy curve (mean of both sides). h_m:"
        " half the MCMC 16–84 % range of d. Ratio h_m / h_p; agreement 0.5–2. τ: largest integrated"
        f" autocorrelation time; converged if the kept half is ≥ {TAU_FACTOR} τ.",
        "",
        "| case | ML d (Å) | h_p (Å) | arm | steps | τ | converged | d 68 % (Å) | h_m (Å) | ratio"
        " | agrees |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for case in CASES:
        c = data.get(case, {})
        if "profile" not in c:
            continue
        pr = c["profile"]
        for arm in ("G", "G-long", "P", "G-DE", "P-DE"):
            if arm not in c:
                continue
            r = c[arm]
            ratio = r["half_width"] / pr["half_width"]
            lines.append(
                f"| {case} | {pr['ml_thickness']:.2f} | {pr['half_width']:.3f} | {arm} | {r['steps']}"
                f" | {r['tau_max']:.0f} | {'yes' if r['converged'] else 'no'}"
                f" | {r['d']['lo']:.2f}–{r['d']['hi']:.2f} | {r['half_width']:.3f} | {ratio:.2f}"
                f" | {'yes' if 0.5 <= ratio <= 2 else 'no'} |"
            )
    (OUT / "mcmc_width_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "parts", nargs="+", choices=["profile", "G", "G-long", "P", "G-DE", "P-DE", "report"]
    )
    ap.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    for part in args.parts:
        if part == "report":
            report()
            continue
        for case in args.cases:
            t = time.perf_counter()
            if part == "profile":
                result = run_profile(case)
            elif part == "G":
                result = run_gaussian(case, LENGTHS[0])
            elif part == "G-long":
                result = until_converged(run_gaussian, case)
            elif part == "P":
                result = until_converged(run_poisson, case)
            else:
                lik = "gaussian" if part == "G-DE" else "poisson"
                result = until_converged(lambda c, n, lik=lik: run_de(c, lik, n), case, DE_LENGTHS)
            data = load()
            data.setdefault(case, {})[part] = result
            RESULTS.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
            print(f"{case} {part}: {time.perf_counter() - t:.1f} s")


if __name__ == "__main__":
    main()
