"""Experiment 001, R1c: is a wrong film from a fit a second solution or a failed fit?

Decision rules are pre-registered in docs/identifiability/README.md ("R1c pre-registration"); this
script computes what those rules need, per case:

    profile  profile likelihood in d on noise-free data: Poisson deviance minimized over the other
             free parameters at fixed d (scipy least_squares on deviance residuals, autograd Jacobian)
    fits     one Poisson draw; refine.refine from 15 starts plus one from the truth; the Poisson
             deviance of every endpoint against the draw
    mcmc     refine.mcmc.sample_posterior from the lowest-deviance endpoint (needs the mcmc extra)

The forward model is fim.py's, with Cu Kα2 and the footprint switched on, on a lab-like q grid.

Run:  uv run python experiments/001_identifiability/nonunique.py profile fits --case mid_oxide
      uv run python experiments/001_identifiability/nonunique.py report
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import fim
import numpy as np
import torch
from fim import FILM, INSTR, OXIDE, Film
from scipy.optimize import least_squares

from reflectolearn.refine.pipeline import RefineConfig, make_forward, refine

S_COUNTS = 1e6
Q = np.linspace(fim.tth_to_q(np.array(0.2)), fim.tth_to_q(np.array(6.0)), 291)
EXTRA = {"kalpha2_ratio": 0.5, "footprint": 500.0}
REFINE_FREE = FILM + INSTR + ("kalpha2_ratio", "footprint") + OXIDE
PARAM_RANGES = SimpleNamespace(thickness=(10.0, 1200.0), roughness=(0.0, 15.0), sld=(5.0, 100.0))
_rc = RefineConfig()
BOUNDS = {
    "thickness": PARAM_RANGES.thickness,
    "roughness": PARAM_RANGES.roughness,
    "sld": PARAM_RANGES.sld,
    "i0": _rc.i0_bounds,
    "bkg": _rc.bkg_bounds,
    "tth_offset": _rc.tth_offset_bounds,
    "dq": _rc.dq_bounds,
    "kalpha2_ratio": _rc.kalpha2_ratio_bounds,
    "footprint": _rc.footprint_bounds,
    "oxide_thickness": _rc.oxide_thickness_bounds,
    "oxide_sld": _rc.oxide_sld_bounds,
    "oxide_roughness": _rc.oxide_roughness_bounds,
}
NO_OXIDE_FREE = tuple(k for k in REFINE_FREE if k not in OXIDE)
# name: (film, free parameters); the truth has the top oxide exactly when the oxide is free.
CASES = {
    "mid_oxide": (Film("mid", 240.0, 6.0, 45.0), REFINE_FREE),
    "mid_no_oxide": (Film("mid", 240.0, 6.0, 45.0), NO_OXIDE_FREE),
    "t2_low": (Film("t2_low", 200.0, 4.0, 12.0), REFINE_FREE),
}
REFINE_INIT = {
    "i0": 1.0,
    "bkg": _rc.bkg_init,
    "tth_offset": 0.0,
    "dq": _rc.dq_init,
    "kalpha2_ratio": _rc.kalpha2_ratio_init,
    "footprint": _rc.footprint_init,
    "oxide_thickness": _rc.oxide_thickness_init,
    "oxide_sld": _rc.oxide_sld_init,
    "oxide_roughness": _rc.oxide_roughness_init,
}
# Relative steps of d away from the truth: fine near it, coarse out to ±30 %.
DELTAS = np.r_[
    np.arange(0.0025, 0.05, 0.0025), np.arange(0.05, 0.15, 0.005), np.arange(0.15, 0.31, 0.02)
]
OUT = Path("docs/identifiability")


def case_truth(name: str) -> dict[str, float]:
    film, free = CASES[name]
    return fim.truth(film, "oxide" if "oxide_sld" in free else "instr") | EXTRA


def counts(p: dict[str, float], s: float = S_COUNTS) -> np.ndarray:
    pt = {k: torch.tensor(v) for k, v in p.items()}
    with torch.no_grad():
        return fim.expected_counts(torch.as_tensor(Q), pt, s, fim.effects_of(p)).numpy()


def deviance_terms(lam: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-point Poisson deviance 2 (y ln(y/lam) - (y - lam)), y = 0 allowed."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(y > 0, y * np.log(y / lam), 0.0) - (y - lam)
    return 2 * np.maximum(t, 0.0)


def deviance(p: dict[str, float], y: np.ndarray) -> float:
    return float(deviance_terms(counts(p), y).sum())


def fit(y: np.ndarray, p_start: dict[str, float], free: tuple[str, ...]) -> tuple[dict, float]:
    """Minimize the Poisson deviance against counts y over `free`, within BOUNDS."""
    lo = np.array([BOUNDS[k][0] for k in free])
    hi = np.array([BOUNDS[k][1] for k in free])
    x0 = np.clip([p_start[k] for k in free], lo + 1e-9 * (hi - lo), hi - 1e-9 * (hi - lo))
    cache: dict = {}

    def model(x):
        key = x.tobytes()
        if key not in cache:
            cache.clear()
            cache[key] = fim.jacobian(Q, p_start | dict(zip(free, x, strict=True)), free, S_COUNTS)
        return cache[key]

    def resid(x):
        lam, _ = model(x)
        return np.sign(lam - y) * np.sqrt(deviance_terms(lam, y))

    def jac(x):
        lam, jlog = model(x)
        r = np.sign(lam - y) * np.sqrt(deviance_terms(lam, y))
        near = np.abs(lam - y) <= 1e-6 * np.maximum(y, 1e-12)
        g = np.where(near, 1 / np.sqrt(lam), (1 - y / lam) / np.where(near, 1.0, r))
        return (g * lam)[:, None] * jlog

    res = least_squares(
        resid, x0, jac=jac, bounds=(lo, hi), x_scale="jac", xtol=1e-10, ftol=1e-10, max_nfev=100
    )
    p = p_start | dict(zip(free, res.x.tolist(), strict=True))
    return p, float(2 * res.cost)


# ------------------------------------------------------------------ profile likelihood
def run_profile(name: str) -> dict:
    film, free = CASES[name]
    p0 = case_truth(name)
    y = counts(p0)
    others = tuple(k for k in free if k != "thickness")
    lam, J = fim.jacobian(Q, p0, free, S_COUNTS)
    fi = fim.fisher(lam, J, free)
    points = [{"delta": 0.0, "dchi2": 0.0, "params": p0}]
    for sign in (1, -1):
        p = p0
        for delta in DELTAS:
            p, d = fit(y, p | {"thickness": film.thickness * (1 + sign * delta)}, others)
            points.append({"delta": float(sign * delta), "dchi2": d, "params": p})
    points.sort(key=lambda r: r["delta"])
    return {
        "case": name,
        "film": film.__dict__,
        "free": list(free),
        "s": S_COUNTS,
        "crlb_d_pct": 100 * fi.crlb[0] / film.thickness,
        "gamma": fi.gamma,
        "weakest": fi.weakest_str(),
        "profile": points,
        **label(points),
    }


def label(points: list[dict], threshold: float = 1.0) -> dict:
    """Pre-registered labels from a profile at s = 1e6."""
    deltas = np.array([r["delta"] for r in points])
    chi = np.array([r["dchi2"] for r in points])
    i0 = int(np.argmin(np.abs(deltas)))
    lo = hi = i0
    while lo > 0 and chi[lo - 1] <= threshold:
        lo -= 1
    while hi < len(chi) - 1 and chi[hi + 1] <= threshold:
        hi += 1
    second = [
        float(deltas[i])
        for i in range(1, len(chi) - 1)
        if not lo <= i <= hi
        and chi[i] <= threshold
        and chi[i] <= chi[i - 1]
        and chi[i] <= chi[i + 1]
    ]
    half_width = max(-deltas[lo], deltas[hi])
    if second:
        verdict = "multimodal"
    elif half_width >= 0.02:
        verdict = "ridge"
    else:
        verdict = "identifiable"
    far = lo if -deltas[lo] >= deltas[hi] else hi
    return {
        "interval": [float(deltas[lo]), float(deltas[hi])],
        "second_minima": second,
        "label": verdict,
        "pair_delta": float(deltas[far]),
    }


# ------------------------------------------------------------------ fits on one noisy curve
def run_fits(name: str, seed: int = 0) -> dict:
    film, free = CASES[name]
    p0 = case_truth(name)
    rng = np.random.default_rng(seed)
    y = rng.poisson(counts(p0)).astype(float)
    mask = y > 0
    log_r = np.where(mask, np.log10(np.where(mask, y, 1.0) / S_COUNTS), -15.0)

    starts = [
        (film.thickness * fd, 0.8 * film.roughness, film.sld * fr)
        for fd in (0.8, 0.9, 1.0, 1.1, 1.2)
        for fr in (0.85, 1.0, 1.15)
    ] + [(film.thickness, film.roughness, film.sld)]
    labels = [f"d×{fd:g} ρ×{fr:g}" for fd in (0.8, 0.9, 1.0, 1.1, 1.2) for fr in (0.85, 1.0, 1.15)]
    labels.append("truth")
    B = len(starts)
    cfg = RefineConfig()
    if not any(k in free for k in OXIDE):
        cfg.fit_oxide = False
    t = time.perf_counter()
    _, info = refine(
        torch.tensor(starts, dtype=torch.float32),
        torch.tensor([[fim.SUB_SLD, fim.SUB_ROUGH]] * B, dtype=torch.float32),
        torch.as_tensor(Q, dtype=torch.float32),
        torch.as_tensor(log_r, dtype=torch.float32).expand(B, -1),
        torch.as_tensor(mask, dtype=torch.float32).expand(B, -1),
        PARAM_RANGES,
        wavelength=fim.WAVELENGTH,
        config=cfg,
    )
    seconds = time.perf_counter() - t

    endpoints = []
    for b in range(B):
        p = dict(p0)
        for k in free:
            p[k] = float(info["state"][k][b])
        if not cfg.fit_oxide:
            p |= {k: p0[k] for k in OXIDE}
        endpoints.append({"start": labels[b], "params": p, "deviance": deviance(p, y)})

    # The Poisson maximum-likelihood fit from the truth, as a reference for D*.
    p_ml, d_ml = fit(y, p0, free)
    d_star = min([d_ml] + [e["deviance"] for e in endpoints])
    for e in endpoints:
        dd = e["deviance"] - d_star
        e["delta_D"] = dd
        e["class"] = "valley" if dd <= 4 else ("failure" if dd > 25 else "undecided")

    # The same points scored by refine's own forward model (float32, grid smearing) and its own
    # objective (log10 MSE over the nonzero counts), to tell a failed search from a model difference.
    fwd = make_forward(
        torch.as_tensor(Q, dtype=torch.float32),
        torch.tensor([[fim.SUB_SLD, fim.SUB_ROUGH]], dtype=torch.float32),
        wavelength=fim.WAVELENGTH,
        config=cfg,
        device=torch.device("cpu"),
    )
    points = [p0, p_ml] + [e["params"] for e in endpoints]
    vals = {k: torch.tensor([p[k] for p in points], dtype=torch.float32) for k in p0}
    with torch.no_grad():
        r = fwd(vals, info["active"]).double().numpy()
    log_mse = (((np.log10(r) - log_r) ** 2) * mask).sum(1) / mask.sum()
    d_refine = deviance_terms(r * S_COUNTS, y).sum(1)
    for e, lm, dr in zip(endpoints, log_mse[2:], d_refine[2:], strict=True):
        e["log_mse_refine_model"] = float(lm)
        e["deviance_refine_model"] = float(dr)
    return {
        "case": name,
        "seed": seed,
        "refine_seconds": seconds,
        "deviance_truth": deviance(p0, y),
        "refine_model": {
            "log_mse_truth": float(log_mse[0]),
            "log_mse_ml": float(log_mse[1]),
            "deviance_truth": float(d_refine[0]),
            "deviance_ml": float(d_refine[1]),
        },
        "ml": {"params": p_ml, "deviance": d_ml},
        "d_star": d_star,
        "endpoints": endpoints,
    }


def noisy_counts(name: str, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).poisson(counts(case_truth(name))).astype(float)


def fit_starts(film: Film) -> list[tuple[str, tuple[float, float, float]]]:
    """The registered starts: d0 x {0.8..1.2} x rho0 x {0.85, 1, 1.15}, sigma 0.8 sigma0."""
    return [
        (f"d×{fd:g} ρ×{fr:g}", (film.thickness * fd, 0.8 * film.roughness, film.sld * fr))
        for fd in (0.8, 0.9, 1.0, 1.1, 1.2)
        for fr in (0.85, 1.0, 1.15)
    ]


def run_lsq(name: str, seed: int = 0) -> dict:
    """Not registered: the Poisson least-squares fit from the same starts as refine, nuisances at
    refine's initial values."""
    film, free = CASES[name]
    p0 = case_truth(name)
    y = noisy_counts(name, seed)
    out = []
    for label_, (d, sigma, rho) in fit_starts(film):
        start = p0 | {k: v for k, v in REFINE_INIT.items() if k in free}
        start |= {"thickness": d, "roughness": sigma, "sld": rho}
        p, dev = fit(y, start, free)
        out.append({"start": label_, "params": p, "deviance": dev})
    return {"seed": seed, "endpoints": out}


# ------------------------------------------------------------------ MCMC
def run_mcmc(name: str, steps: int, start: str, seed: int = 0) -> dict:
    from reflectolearn.refine.mcmc import sample_posterior

    saved = json.loads((OUT / f"nonunique_{name}.json").read_text(encoding="utf-8"))
    fits = saved["fits"]
    film, free = CASES[name]
    p0 = case_truth(name)
    rng = np.random.default_rng(fits["seed"])
    y = rng.poisson(counts(p0)).astype(float)
    mask = y > 0
    log_r = np.where(mask, np.log10(np.where(mask, y, 1.0) / S_COUNTS), -15.0)
    # "refine": the lowest-deviance refine endpoint (as registered); "ml": the Poisson ML fit
    if start == "refine":
        best = min(fits["endpoints"], key=lambda e: e["deviance"])["params"]
    else:
        best = fits["ml"]["params"]

    cfg = RefineConfig()
    if not any(k in free for k in OXIDE):
        cfg.fit_oxide = False
    active = {"tth_offset", "dq", "kalpha2", "footprint"} | ({"oxide"} if cfg.fit_oxide else set())
    fitted = [k for k in REFINE_FREE if cfg.fit_oxide or k not in OXIDE]
    info = {
        "state": {k: torch.tensor([best[k]], dtype=torch.float32) for k in REFINE_FREE},
        "active": active,
        "bounds": {k: BOUNDS[k] for k in fitted},
        "fitted": fitted,
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
        seed=seed,
    )
    seconds = time.perf_counter() - t
    d = post["samples"][:, post["keys"].index("thickness")]
    return {
        "start": start,
        "start_thickness": best["thickness"],
        "steps": steps,
        "seconds": seconds,
        "walkers": post["walkers"],
        "acceptance": post["acceptance"],
        "params": post["params"],
        "d_68": [float(np.percentile(d, 16)), float(np.percentile(d, 84))],
        "log_prob_start": post["log_prob_start"],
        "log_prob_max": post["log_prob_max"],
    }


def half_width(points: list[dict], threshold: float = 1.0) -> tuple[float, float]:
    """Relative d at which the profile crosses `threshold` on each side: linear in Δχ² between grid
    points, quadratic from the truth when the first grid point is already above it."""
    out = []
    for sign in (-1, 1):
        side = sorted((r for r in points if sign * r["delta"] >= 0), key=lambda r: abs(r["delta"]))
        cross = float("nan")
        for a, b in zip(side, side[1:], strict=False):
            if a["dchi2"] <= threshold < b["dchi2"]:
                if a["delta"] == 0:
                    cross = abs(b["delta"]) * (threshold / b["dchi2"]) ** 0.5
                else:
                    t = (threshold - a["dchi2"]) / (b["dchi2"] - a["dchi2"])
                    cross = abs(a["delta"]) + t * (abs(b["delta"]) - abs(a["delta"]))
                break
        out.append(cross)
    return out[0], out[1]


def summary(name: str) -> dict:
    data = json.loads((OUT / f"nonunique_{name}.json").read_text(encoding="utf-8"))
    pr, fits, lsq = data["profile"], data["fits"], data["lsq"]
    d0 = pr["film"]["thickness"]
    d_star = min([fits["d_star"]] + [e["deviance"] for e in lsq["endpoints"]])
    lm_ml = fits["refine_model"]["log_mse_ml"]
    lo, hi = half_width(pr["profile"])
    row = {
        "case": name,
        "label": pr["label"],
        "profile_pct": (100 * lo, 100 * hi),
        "crlb_pct": pr["crlb_d_pct"],
        "gamma": pr["gamma"],
        "refine_failure": sum(e["class"] == "failure" for e in fits["endpoints"]),
        "refine_n": len(fits["endpoints"]),
        "refine_min_dD": min(e["delta_D"] for e in fits["endpoints"]),
        "refine_worse_own": sum(e["log_mse_refine_model"] > lm_ml for e in fits["endpoints"]),
        "refine_d_pct": [100 * (e["params"]["thickness"] / d0 - 1) for e in fits["endpoints"]],
        "lsq_at_ml": sum(e["deviance"] - d_star <= 4 for e in lsq["endpoints"]),
        "lsq_n": len(lsq["endpoints"]),
        "ml_d_pct": 100 * (fits["ml"]["params"]["thickness"] / d0 - 1),
    }
    for st in ("refine", "ml"):
        m = data.get(f"mcmc_{st}")
        if m:
            row[f"mcmc_{st}_pct"] = [100 * (x / d0 - 1) for x in m["d_68"]]
    return row


def report() -> None:
    rows = [summary(n) for n in CASES]
    lines = [
        "# R1c tables (generated by nonunique.py report; do not edit)",
        "",
        "d in % of the true value. Profile: where the noise-free profile crosses Δχ² = 1 at"
        " s = 1e6 (interpolated). CRLB: unbounded local bound, same model. refine failure:"
        " endpoints with ΔD > 25 / all. Worse on own objective: endpoints whose log10 MSE under"
        " refine's own model is above that of the Poisson ML parameters. LSQ at ML: Poisson"
        " least-squares fits from the same starts that end within ΔD ≤ 4 of the best. MCMC: 68 %"
        " interval of refine.mcmc, started from the best refine endpoint / from the ML fit.",
        "",
        "| case | label | profile Δχ²≤1 | CRLB | γ | refine failure | worse on own objective"
        " | refine d range | LSQ at ML | ML d | MCMC (refine start) | MCMC (ML start) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lo, hi = r["profile_pct"]
        mc = [r.get(f"mcmc_{st}_pct") for st in ("refine", "ml")]
        mc_s = [f"{m[0]:+.2f} to {m[1]:+.2f}" if m else "—" for m in mc]
        lines.append(
            f"| {r['case']} | {r['label']} | −{lo:.2f} / +{hi:.2f} | ±{r['crlb_pct']:.2f} "
            f"| {r['gamma']:.0f} | {r['refine_failure']}/{r['refine_n']} "
            f"(min ΔD {r['refine_min_dD']:.3g}) | {r['refine_worse_own']}/{r['refine_n']} "
            f"| {min(r['refine_d_pct']):+.1f} to {max(r['refine_d_pct']):+.1f} "
            f"| {r['lsq_at_ml']}/{r['lsq_n']} | {r['ml_d_pct']:+.2f} | {mc_s[0]} | {mc_s[1]} |"
        )
    (OUT / "nonunique_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    figure()


def figure() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    colors = {"mid_oxide": "#2a78d6", "mid_no_oxide": "#eb6834", "t2_low": "#1baf7a"}
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.edgecolor": ink2,
            "axes.labelcolor": ink,
            "xtick.color": ink2,
            "ytick.color": ink2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": grid,
            "grid.linewidth": 0.6,
            "figure.facecolor": surface,
            "axes.facecolor": surface,
        }
    )
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    data = {n: json.loads((OUT / f"nonunique_{n}.json").read_text(encoding="utf-8")) for n in CASES}

    # (a) profiles of d
    for n, d in data.items():
        pts = [r for r in d["profile"]["profile"] if abs(r["delta"]) <= 0.05]
        a.plot(
            [100 * r["delta"] for r in pts],
            [max(r["dchi2"], 1e-3) for r in pts],
            "-o",
            ms=2.5,
            lw=2,
            color=colors[n],
            label=n,
        )
    a.axhline(1, color=ink2, lw=0.8)
    a.text(4.9, 1.3, "Δχ² = 1", color=ink2, fontsize=7.5, ha="right")
    a.set_yscale("log")
    a.set_ylim(1e-3, 1e5)
    a.set_xlabel("d − d_true (%), other parameters re-fitted")
    a.set_ylabel("Δχ² (Poisson deviance, s = 10⁶)")
    a.set_title("(a) profile likelihood of d, noise-free data", loc="left", color=ink)
    a.legend(frameon=False, fontsize=7.5)

    # (b) endpoints on one noisy curve: refine vs Poisson LSQ from the same starts
    for n, d in data.items():
        d0 = d["profile"]["film"]["thickness"]
        d_star = min([d["fits"]["d_star"]] + [e["deviance"] for e in d["lsq"]["endpoints"]])
        for key, marker, filled in (("fits", "o", False), ("lsq", "^", True)):
            ends = d[key]["endpoints"]
            b.scatter(
                [100 * (e["params"]["thickness"] / d0 - 1) for e in ends],
                [e["deviance"] - d_star + 1 for e in ends],
                marker=marker,
                s=26,
                facecolors=colors[n] if filled else "none",
                edgecolors=colors[n],
                linewidths=1.2,
                label=f"{n}, {'refine' if key == 'fits' else 'Poisson LSQ'}",
            )
    b.axhline(5, color=ink2, lw=0.8)
    b.text(-38, 6.5, "ΔD = 4", color=ink2, fontsize=7.5)
    b.set_yscale("log")
    b.set_xlabel("d of the endpoint − d_true (%)")
    b.set_ylabel("ΔD + 1 against the best fit")
    b.set_title("(b) one noisy curve: refine (open) vs Poisson LSQ (filled)", loc="left", color=ink)
    b.legend(frameon=False, fontsize=6.5, ncol=2, loc="center")
    b.set_ylim(0.5, 1e8)

    # (c) normalized residuals on the noisy mid_oxide curve
    n = "mid_oxide"
    y = noisy_counts(n, data[n]["fits"]["seed"])
    ml = data[n]["fits"]["ml"]["params"]
    best_refine = min(data[n]["fits"]["endpoints"], key=lambda e: e["deviance"])
    far = max(
        (r for r in data[n]["profile"]["profile"] if r["dchi2"] <= 1),
        key=lambda r: abs(r["delta"]),
    )
    for label_, p, col in (
        (f"Poisson ML (d {ml['thickness']:.1f} Å)", ml, ink2),
        (
            f"profile point, Δχ² {far['dchi2']:.2f} (d {far['params']['thickness']:.1f} Å)",
            far["params"],
            "#1baf7a",
        ),
        (
            f"best refine endpoint (d {best_refine['params']['thickness']:.1f} Å)",
            best_refine["params"],
            "#e34948",
        ),
    ):
        lam = counts(p)
        c.plot(Q, (y - lam) / np.sqrt(np.maximum(lam, 1.0)), lw=1.1, color=col, label=label_)
    c.set_ylim(-40, 40)
    c.set_xlabel("q (Å⁻¹)")
    c.set_ylabel("(counts − model) / √model")
    c.set_title("(c) mid_oxide, one Poisson curve: residuals", loc="left", color=ink)
    c.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.savefig(OUT / "nonunique.png", dpi=150)
    plt.close(fig)


def save(name: str, part: str, result: dict) -> None:
    path = OUT / f"nonunique_{name}.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[part] = result
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("parts", nargs="+", choices=["profile", "fits", "lsq", "mcmc", "report"])
    ap.add_argument("--case", choices=list(CASES))
    ap.add_argument("--mcmc-steps", type=int, default=2000)
    ap.add_argument("--mcmc-start", choices=["refine", "ml"], default="refine")
    args = ap.parse_args()
    for part in args.parts:
        t = time.perf_counter()
        if part == "report":
            report()
            continue
        if part == "profile":
            result = run_profile(args.case)
        elif part == "fits":
            result = run_fits(args.case)
        elif part == "lsq":
            result = run_lsq(args.case)
        else:
            result = run_mcmc(args.case, args.mcmc_steps, args.mcmc_start)
            part = f"mcmc_{args.mcmc_start}"
        save(args.case, part, result)
        print(f"{args.case} {part}: {time.perf_counter() - t:.1f} s")


if __name__ == "__main__":
    main()
