"""Experiment 001, R2b: which setting makes refine stop short of the best fit (ReflectoLearn#25)?

Decision rules are pre-registered in docs/identifiability/refine_causes.md. Per case, one Poisson
curve is drawn from refine's own forward model at the truth; refine runs from the 15 R1c starts plus
the truth under a configuration built from factors A-F (budget, stage 2, objective, Occam prior,
trust region, step size), and every endpoint is scored by its Poisson deviance under the same model.

Configuration names: "base" (all off), a string of factor letters ("A", "ABCDEF"), "all-X" for all on
minus factor X, and an optional "@m" budget multiplier on top ("ABCDEF@20"). "ref" is ABCDEF@30
from the truth only.

Run:  uv run python experiments/001_identifiability/refine_causes.py run --configs base A
      uv run python experiments/001_identifiability/refine_causes.py report
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

import fim
import numpy as np
import torch
from fim import Film
from nonunique import (
    BOUNDS,
    CASES,
    EXTRA,
    OUT,
    PARAM_RANGES,
    REFINE_FREE,
    S_COUNTS,
    Q,
    deviance_terms,
    fit_starts,
)
from scipy.optimize import least_squares

from reflectolearn.refine.losses import LOSS_PRESETS
from reflectolearn.refine.pipeline import (
    OXIDE_KEYS,
    RefineConfig,
    RefineStage,
    _default_stages,
    make_forward,
    refine,
)

# fim sets float64 as the default dtype; refine is meant to run in float32 as in the package.
torch.set_default_dtype(torch.float32)

FACTORS = "ABCDEF"
SINGLES = ["base", *FACTORS, FACTORS, *(f"all-{f}" for f in FACTORS)]
RESULTS = OUT / "refine_causes.json"
CONFIRM = OUT / "refine_confirm.json"
# R2f (registered in refine_causes.md): a film R2b did not use and new seeds for two R2b films.
R2F = {
    "thin_oxide": (Film("thin_oxide", 90.0, 3.0, 28.0), REFINE_FREE, 21),
    "mid_oxide": (*CASES["mid_oxide"], 22),
    "t2_low": (*CASES["t2_low"], 23),
}
ALL_EFFECTS = {"tth_offset", "dq", "kalpha2", "footprint", "oxide"}


def poisson_loss(r_sim, log_r_obs, mask):
    """Poisson deviance per point at S_COUNTS counts for R = 1. Ignores the mask: points with zero
    counts (log_r_obs = -15) carry information too."""
    lam = S_COUNTS * r_sim.double().clamp_min(1e-15)
    y = S_COUNTS * torch.pow(10.0, log_r_obs.double())
    dev = 2 * (lam - y + y * torch.log(y / lam))
    return dev.mean().to(r_sim.dtype)


# Factor C names this loss by key in RefineStage.loss.
LOSS_PRESETS["poisson"] = poisson_loss


def factors_of(name: str) -> tuple[set[str], int]:
    name, _, mult = name.partition("@")
    if name == "base":
        on = set()
    elif name.startswith("all-"):
        on = set(FACTORS) - {name[4]}
    else:
        on = set(name)
    if not on <= set(FACTORS) or name.startswith("all-") and name[4:] not in FACTORS:
        raise ValueError(f"unknown configuration {name!r}")
    return on, int(mult or (10 if "A" in on else 1))


def build_config(on: set[str], mult: int, oxide: bool) -> RefineConfig:
    stages = [replace(s, steps=s.steps * mult) for s in _default_stages()]
    if "B" in on:
        stages = [s for s in stages if s.name != "2.structure"]
    fine = stages[-1]
    if "C" in on:
        stages[-1] = fine = replace(fine, loss="poisson")
    if "F" in on:
        stages += [
            RefineStage(f"4.polish{lr:g}", fine.optimize, fine.loss, 300 * mult, lr)
            for lr in (1e-3, 1e-4)
        ]
    return RefineConfig(
        fit_oxide=oxide,
        nuisance_reg=0.0 if "D" in on else RefineConfig.nuisance_reg,
        trust_margin=1.0 if "E" in on else RefineConfig.trust_margin,
        stages=stages,
    )


def curve(name: str, seed: int = 0):
    """One Poisson draw from refine's own forward model at the truth of an R2b case."""
    film, free = CASES[name]
    return draw(film, free, seed)


def draw(film: Film, free: tuple[str, ...], seed: int):
    oxide = any(k in free for k in OXIDE_KEYS)
    fwd = model(oxide)
    p0 = fim.truth(film, "oxide" if oxide else "instr") | EXTRA
    with torch.no_grad():
        r = fwd({k: torch.tensor([v]) for k, v in p0.items()}, ALL_EFFECTS).double().numpy()[0]
    y = np.random.default_rng(seed).poisson(S_COUNTS * r).astype(float)
    return p0, y, oxide


def model(oxide: bool):
    return make_forward(
        torch.as_tensor(Q, dtype=torch.float32),
        torch.tensor([[fim.SUB_SLD, fim.SUB_ROUGH]]),
        wavelength=fim.WAVELENGTH,
        config=RefineConfig(fit_oxide=oxide),
        device=torch.device("cpu"),
    )


def scores(states: dict[str, torch.Tensor], y: np.ndarray, oxide: bool) -> np.ndarray:
    with torch.no_grad():
        r = model(oxide)(states, ALL_EFFECTS).double().numpy()
    return deviance_terms(S_COUNTS * r, y).sum(1)


def lsq_fit(
    y: np.ndarray, p_base: dict[str, float], free: tuple[str, ...], oxide: bool
) -> tuple[dict[str, float], float, int]:
    """Minimize the Poisson deviance of counts y under refine's own model over `free` from p_base,
    by bounded least squares on deviance residuals, within the global bounds. The model is float32,
    so the Jacobian is central differences, all of them in one batched forward call. Returns the
    full parameter dict, the deviance and the number of evaluations."""
    fwd = model(oxide)
    lo = np.array([BOUNDS[k][0] for k in free])
    hi = np.array([BOUNDS[k][1] for k in free])
    h = 3e-4 * (hi - lo)
    n = len(free)

    def lam(rows: np.ndarray) -> np.ndarray:
        vals = {k: torch.full((len(rows),), v) for k, v in p_base.items()}
        vals |= {k: torch.as_tensor(rows[:, i], dtype=torch.float32) for i, k in enumerate(free)}
        with torch.no_grad():
            return S_COUNTS * fwd(vals, ALL_EFFECTS).double().numpy()

    def resid_of(lam_: np.ndarray) -> np.ndarray:
        return np.sign(lam_ - y) * np.sqrt(deviance_terms(lam_, y))

    def resid(x):
        return resid_of(lam(x[None])[0])

    def jac(x):
        xs = np.clip(np.r_[x + np.diag(h), x - np.diag(h)], lo, hi)
        r = resid_of(lam(xs))
        width = (xs[:n] - xs[n:]).diagonal()
        return ((r[:n] - r[n:]) / width[:, None]).T

    x0 = np.clip([p_base[k] for k in free], lo, hi)
    res = least_squares(resid, x0, jac=jac, bounds=(lo, hi), x_scale="jac", max_nfev=200)
    return p_base | dict(zip(free, res.x.tolist(), strict=True)), float(2 * res.cost), res.nfev


def reference_ml(case: str) -> dict:
    """The Poisson maximum likelihood under refine's own model, from the truth, independent of
    refine's optimizer."""
    _, free = CASES[case]
    p0, y, oxide = curve(case)
    p, dev, nfev = lsq_fit(y, p0, free, oxide)
    return {"deviance": dev, "params": {k: p[k] for k in free}, "nfev": int(nfev)}


def run_config(case: str, config_name: str) -> dict:
    film, free = CASES[case]
    return run_fits(film, free, 0, config_name)


def run_fits(film: Film, free: tuple[str, ...], seed: int, config_name: str) -> dict:
    p0, y, oxide = draw(film, free, seed)
    mask = y > 0
    log_r = np.where(mask, np.log10(np.where(mask, y, 1.0) / S_COUNTS), -15.0)
    if config_name == "ref":
        on, mult = set(FACTORS), 30
        starts = [("truth", (film.thickness, film.roughness, film.sld))]
    else:
        on, mult = factors_of(config_name)
        starts = [*fit_starts(film), ("truth", (film.thickness, film.roughness, film.sld))]
    cfg = build_config(on, mult, oxide)
    B = len(starts)
    t = time.perf_counter()
    _, info = refine(
        torch.tensor([s for _, s in starts]),
        torch.tensor([[fim.SUB_SLD, fim.SUB_ROUGH]] * B),
        torch.as_tensor(Q, dtype=torch.float32),
        torch.as_tensor(log_r, dtype=torch.float32).expand(B, -1),
        torch.as_tensor(mask, dtype=torch.float32).expand(B, -1),
        PARAM_RANGES,
        wavelength=fim.WAVELENGTH,
        config=cfg,
    )
    seconds = time.perf_counter() - t
    states = {k: v.float() for k, v in info["state"].items()}
    dev = scores(states, y, oxide)
    keys = [k for k in free if k in states]
    return {
        "seconds": seconds,
        "deviance_truth": float(scores({k: torch.tensor([v]) for k, v in p0.items()}, y, oxide)[0]),
        "endpoints": [
            {
                "start": label,
                "deviance": float(dev[b]),
                "params": {k: float(states[k][b]) for k in keys},
            }
            for b, (label, _) in enumerate(starts)
        ],
    }


def load() -> dict:
    return json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}


def report() -> None:
    data = load()
    lsq = data.get("lsq", {})
    d_star = {
        c: min(
            [e["deviance"] for runs in data.values() for e in runs.get(c, {}).get("endpoints", [])]
            + ([lsq[c]["deviance"]] if c in lsq else [])
        )
        for c in CASES
    }
    names = [n for n in [*SINGLES, f"{FACTORS}@20", "ref", "ABCF"] if n in data]
    lines = [
        "# R2b tables (generated by refine_causes.py report; do not edit)",
        "",
        "Reached: endpoints with D − D* ≤ 4 under refine's own forward model, D* the lowest"
        " deviance found on that curve by any configuration or by the reference least-squares fit"
        " (amendment 1). d range: endpoint thickness in % of the truth. Seconds: refine wall time"
        " for the 16 starts of all three cases, CPU, 4 threads.",
        "",
        "| case | D* | D of the truth | reference LSQ |",
        "|---|---|---|---|",
        *(
            f"| {c} | {d_star[c]:.1f} | {data['base'][c]['deviance_truth']:.1f}"
            f" | {lsq[c]['deviance']:.1f} |"
            for c in CASES
            if c in data.get("base", {}) and c in lsq
        ),
        "",
        "| config | factors on | "
        + " | ".join(f"{c} reached" for c in CASES)
        + " | total | median ΔD | d range (%) | seconds |",
        "|---|---|" + "---|" * len(CASES) + "---|---|---|---|",
    ]
    for n in names:
        runs = data[n]
        on, mult = factors_of(n) if n != "ref" else (set(FACTORS), 30)
        reached, total, dds, dpct, secs = [], 0, [], [], 0.0
        for c in CASES:
            if c not in runs:
                reached.append("—")
                continue
            d0 = CASES[c][0].thickness
            ends = runs[c]["endpoints"]
            dd = [e["deviance"] - d_star[c] for e in ends]
            k = sum(x <= 4 for x in dd)
            reached.append(f"{k}/{len(ends)}")
            total += k
            dds += dd
            dpct += [100 * (e["params"]["thickness"] / d0 - 1) for e in ends]
            secs += runs[c]["seconds"]
        n_all = sum(len(runs[c]["endpoints"]) for c in CASES if c in runs)
        lines.append(
            f"| {n} | {''.join(sorted(on)) or '—'} ×{mult} | "
            + " | ".join(reached)
            + f" | {total}/{n_all} | {np.median(dds):.3g} | {min(dpct):+.1f} to {max(dpct):+.1f}"
            f" | {secs:.0f} |"
        )
    (OUT / "refine_causes_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def run_confirm(case: str) -> dict:
    """R2f: the preset (ABCF) on a new curve; D* from the endpoints and two least-squares fits."""
    film, free, seed = R2F[case]
    result = run_fits(film, free, seed, "ABCF")
    p0, y, oxide = draw(film, free, seed)
    _, d_truth, _ = lsq_fit(y, p0, free, oxide)
    best = min(result["endpoints"], key=lambda e: e["deviance"])
    _, d_best, _ = lsq_fit(y, p0 | best["params"], free, oxide)
    d_star = min(d_truth, d_best, best["deviance"])
    for e in result["endpoints"]:
        e["delta_D"] = e["deviance"] - d_star
    return result | {
        "seed": seed,
        "d_star": d_star,
        "lsq_from_truth": d_truth,
        "lsq_from_best": d_best,
        "reached": sum(e["delta_D"] <= 4 for e in result["endpoints"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("part", choices=["run", "reference", "report", "confirm"])
    ap.add_argument("--configs", nargs="+", default=SINGLES)
    ap.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    if args.part == "report":
        report()
        return
    torch.set_num_threads(args.threads)
    if args.part == "confirm":
        total = 0
        for case in R2F:
            result = run_confirm(case)
            data = json.loads(CONFIRM.read_text(encoding="utf-8")) if CONFIRM.exists() else {}
            data[case] = result
            CONFIRM.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
            total += result["reached"]
            print(
                f"R2f {case}: reached {result['reached']}/16, D* {result['d_star']:.1f}"
                f" (lsq truth {result['lsq_from_truth']:.1f}, best {result['lsq_from_best']:.1f})",
                flush=True,
            )
        print(f"R2f total {total}/48 (confirmed if >= 43)")
        return
    if args.part == "reference":
        for case in args.cases:
            result = reference_ml(case)
            data = load()
            data.setdefault("lsq", {})[case] = result
            RESULTS.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
            print(f"lsq {case}: D {result['deviance']:.1f}, {result['nfev']} evaluations")
        return
    for name in args.configs:
        for case in args.cases:
            result = run_config(case, name)
            data = load()
            data.setdefault(name, {})[case] = result
            RESULTS.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
            dev = [e["deviance"] for e in result["endpoints"]]
            print(f"{name:>10} {case:<13} {result['seconds']:6.1f} s  min D {min(dev):.1f}")


if __name__ == "__main__":
    main()
