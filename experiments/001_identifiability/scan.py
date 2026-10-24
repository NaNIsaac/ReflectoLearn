"""Experiment 001, R2a: where does a single-film or bilayer XRR curve have several solutions?

Decision rules are pre-registered in docs/identifiability/generative_go_nogo.md ("Pre-registered
experiment for 2026-10-23: condition scan for multimodality"). Per condition, on noise-free data
(the expected counts at the truth):

    starts   64 Latin-hypercube starts over the full prior box; Poisson least squares from each
    modes    endpoints with Δχ² ≤ 4 grouped into modes by their film parameters
    profile  Δχ² = 1 half-widths of every film parameter at each mode's best endpoint

The forward model is fim.py's (abeles_core in complex128, top oxide, Kα2, footprint, 2θ offset,
constant-dq resolution, i0, background), extended to a second film under the first.

Run:  uv run python experiments/001_identifiability/scan.py check
      uv run python experiments/001_identifiability/scan.py run --conditions 0 1 2
      uv run python experiments/001_identifiability/scan.py report
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time

import fim
import numpy as np
import torch
from fim import SUB_ROUGH, SUB_SLD, WAVELENGTH, WAVELENGTH_KA2
from nonunique import BOUNDS, OUT, REFINE_INIT, deviance_terms
from scipy.optimize import least_squares
from scipy.stats import qmc
from torch import Tensor
from torch.func import jacfwd

from reflectolearn.benchmark import parratt
from reflectolearn.physics import abeles_core, apply_tth_offset

S_COUNTS = 1e6
N_STARTS = 64
MAX_NFEV = 200  # evaluations per global-start fit
TTH_MAX = (2.2, 3.0, 6.0)  # deg -> q_max 0.16, 0.21, 0.43 Å^-1
FILM_SLD = (12.0, 24.0, 45.0)
NUISANCE = ("refine", "wide")
STACK = ("single", "bilayer")
WIDE = {"oxide_sld": (5.0, 40.0), "oxide_thickness": (0.0, 60.0), "kalpha2_ratio": (0.0, 1.0)}
SECOND = {"thickness2": 120.0, "roughness2": 6.0, "sld_factor": 1.25}
FILM1 = ("thickness", "roughness", "sld")
FILM2 = ("thickness2", "roughness2", "sld2")
NUIS = (
    "i0",
    "bkg",
    "tth_offset",
    "dq",
    "kalpha2_ratio",
    "footprint",
    "oxide_thickness",
    "oxide_sld",
    "oxide_roughness",
)
SCAN_DIR = OUT / "scan"
# Resolution quadrature: Gauss-Hermite nodes for the Gaussian weight. 15 nodes differ from fim.py's
# 41-point ±5σ rule by a deviance of < 0.002 at 10^6 counts and cost a third as much.
_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(15)
GH_U, GH_W = torch.as_tensor(_GH_X), torch.as_tensor(_GH_W / _GH_W.sum())


# ------------------------------------------------------------------ conditions
# Named case (amendment 5): R1c's t2_low film, whose noisy curve has two Kα2 basins.
T2_LOW = {
    "tth_max": 6.0,
    "film_sld": 12.0,
    "nuisance": "refine",
    "stack": "single",
    "film": (200.0, 4.0),
}


def conditions() -> list[dict]:
    grid = [
        {"tth_max": t, "film_sld": s, "nuisance": n, "stack": k}
        for k, n, t, s in itertools.product(STACK, NUISANCE, TTH_MAX, FILM_SLD)
    ]
    return [*grid, T2_LOW]


def q_grid(tth_max: float) -> np.ndarray:
    """Uniform in q from 2θ 0.2° to tth_max, as many points as a 0.02° step."""
    n = round((tth_max - 0.2) / 0.02) + 1
    return np.linspace(fim.tth_to_q(np.array(0.2)), fim.tth_to_q(np.array(tth_max)), n)


def setup(cond: dict) -> tuple[dict[str, float], tuple[str, ...], dict[str, tuple[float, float]]]:
    """Truth, free parameters and bounds for a condition."""
    d, sigma = cond.get("film", (240.0, 6.0))
    p0 = {"thickness": d, "roughness": sigma, "sld": cond["film_sld"]}
    p0 |= fim.TRUE_INSTR | fim.TRUE_OXIDE | {"kalpha2_ratio": 0.5, "footprint": 500.0}
    free = FILM1
    if cond["stack"] == "bilayer":
        p0 |= {
            "thickness2": SECOND["thickness2"],
            "roughness2": SECOND["roughness2"],
            "sld2": SECOND["sld_factor"] * cond["film_sld"],
        }
        free += FILM2
    free += NUIS
    bounds = dict(BOUNDS)
    bounds |= {k2: BOUNDS[k1] for k1, k2 in zip(FILM1, FILM2, strict=True)}
    if cond["nuisance"] == "wide":
        bounds |= WIDE
    return p0, free, bounds


# ------------------------------------------------------------------ forward model (torch, float64)
def reflectivity(q: Tensor, p: dict[str, Tensor], absorption: float = 1e-9) -> Tensor:
    """R(q) of ambient / top oxide / film / [film 2] / substrate through abeles_core in complex128,
    prepared as fim.reflectivity."""
    zero = torch.zeros_like(p["sld"])
    sld = [zero, p["oxide_sld"], p["sld"]]
    thk = [zero, p["oxide_thickness"], p["thickness"]]
    rgh = [p["oxide_roughness"], p["roughness"]]
    if "thickness2" in p:
        sld.append(p["sld2"])
        thk.append(p["thickness2"])
        rgh.append(p["roughness2"])
    sld.append(zero + SUB_SLD)
    rgh.append(zero + SUB_ROUGH)
    sld_c = torch.stack(sld).to(torch.complex128) * 1e-6 + 1j * absorption
    q_c = q.to(torch.complex128).reshape(1, -1, 1)
    return abeles_core(
        q_c,
        torch.stack(thk).reshape(1, 1, -1),
        torch.stack(rgh).reshape(1, 1, -1),
        sld_c.reshape(1, 1, -1),
    )[0]


def expected_counts(q: Tensor, p: dict[str, Tensor], absorption: float = 1e-9) -> Tensor:
    """fim.expected_counts with every effect on, for this module's stacks, with Gauss-Hermite
    quadrature of the resolution."""
    q_eff = apply_tth_offset(q, p["tth_offset"].reshape(1, 1), WAVELENGTH)[0]
    qq = (q_eff[:, None] + p["dq"] * GH_U[None, :]).reshape(-1)
    r = reflectivity(qq, p, absorption)
    r2 = reflectivity(qq * (WAVELENGTH / WAVELENGTH_KA2), p, absorption)
    r = (r + p["kalpha2_ratio"] * r2) / (1 + p["kalpha2_ratio"])
    r = r * torch.erf(p["footprint"] * (qq * WAVELENGTH / (4 * math.pi)).clamp_min(0))
    r = (r.reshape(-1, len(GH_U)) * GH_W).sum(1)
    return S_COUNTS * (p["i0"] * r + 10.0 ** p["bkg"])


def model(q: np.ndarray, p_base: dict[str, float], free: tuple[str, ...]):
    """(counts, d ln counts / d theta) at theta as a function of theta (numpy in, numpy out)."""
    qt = torch.as_tensor(q)
    fixed = {k: torch.tensor(v) for k, v in p_base.items()}

    def log_counts(theta: Tensor) -> Tensor:
        return torch.log(expected_counts(qt, fixed | dict(zip(free, theta, strict=True))))

    jac = jacfwd(log_counts, has_aux=False)

    def f(x: np.ndarray):
        theta = torch.as_tensor(x)
        with torch.no_grad():
            lam = torch.exp(log_counts(theta)).numpy()
        return lam, jac(theta).numpy()

    return f


def fit(
    q: np.ndarray,
    y: np.ndarray,
    p_start: dict[str, float],
    free: tuple[str, ...],
    bounds: dict[str, tuple[float, float]],
    max_nfev: int = MAX_NFEV,
) -> tuple[dict[str, float], float, int]:
    """Poisson deviance minimized over `free` within `bounds` (as nonunique.fit)."""
    lo = np.array([bounds[k][0] for k in free])
    hi = np.array([bounds[k][1] for k in free])
    x0 = np.clip([p_start[k] for k in free], lo + 1e-9 * (hi - lo), hi - 1e-9 * (hi - lo))
    f = model(q, p_start, free)
    cache: dict = {}

    def m(x):
        key = x.tobytes()
        if key not in cache:
            cache.clear()
            cache[key] = f(x)
        return cache[key]

    def resid(x):
        lam, _ = m(x)
        return np.sign(lam - y) * np.sqrt(deviance_terms(lam, y))

    def jac(x):
        lam, jlog = m(x)
        r = np.sign(lam - y) * np.sqrt(deviance_terms(lam, y))
        near = np.abs(lam - y) <= 1e-6 * np.maximum(y, 1e-12)
        g = np.where(near, 1 / np.sqrt(lam), (1 - y / lam) / np.where(near, 1.0, r))
        return (g * lam)[:, None] * jlog

    res = least_squares(
        resid,
        x0,
        jac=jac,
        bounds=(lo, hi),
        x_scale="jac",
        xtol=1e-10,
        ftol=1e-10,
        max_nfev=max_nfev,
    )
    return p_start | dict(zip(free, res.x.tolist(), strict=True)), float(2 * res.cost), res.nfev


# ------------------------------------------------------------------ check
def check_vs_parratt() -> float:
    """Max relative difference of R between this module's bilayer stack (no absorption) and
    benchmark.parratt over 2θ 0.2–6°."""
    q = q_grid(6.0)
    p0, _, _ = setup({"tth_max": 6.0, "film_sld": 24.0, "nuisance": "refine", "stack": "bilayer"})
    p = {k: torch.tensor(v) for k, v in p0.items()}
    r_torch = reflectivity(torch.as_tensor(q), p, absorption=0.0).numpy()
    layers = [
        (0j, 0.0, 0.0),
        (complex(p0["oxide_sld"]), p0["oxide_thickness"], p0["oxide_roughness"]),
        (complex(p0["sld"]), p0["thickness"], p0["roughness"]),
        (complex(p0["sld2"]), p0["thickness2"], p0["roughness2"]),
        (complex(SUB_SLD), 0.0, SUB_ROUGH),
    ]
    return float(np.max(np.abs(r_torch / parratt(q, layers) - 1)))


# ------------------------------------------------------------------ one condition
def lhs_starts(p0, free, bounds, n: int = N_STARTS, seed: int = 0) -> list[dict[str, float]]:
    lo = np.array([bounds[k][0] for k in free])
    hi = np.array([bounds[k][1] for k in free])
    u = qmc.LatinHypercube(d=len(free), seed=seed).random(n)
    return [p0 | dict(zip(free, (lo + x * (hi - lo)).tolist(), strict=True)) for x in u]


def run_condition(index: int, n_starts: int = N_STARTS) -> dict:
    cond = conditions()[index]
    p0, free, bounds = setup(cond)
    q = q_grid(cond["tth_max"])
    y = expected_counts(torch.as_tensor(q), {k: torch.tensor(v) for k, v in p0.items()}).numpy()
    ends = []
    t = time.perf_counter()
    for i, start in enumerate(lhs_starts(p0, free, bounds, n_starts)):
        p, dev, nfev = fit(q, y, start, free, bounds)
        ends.append({"start": i, "deviance": dev, "nfev": int(nfev), "params": p})
    return {
        "index": index,
        "condition": cond,
        "truth": p0,
        "free": list(free),
        "n_q": len(q),
        "seconds_fits": time.perf_counter() - t,
        "endpoints": ends,
    }


# ------------------------------------------------------------------ modes and labels
def film_keys(free) -> tuple[str, ...]:
    """The film parameters that define modes and get labels: d and ρ of each layer."""
    return tuple(k for k in ("thickness", "sld", "thickness2", "sld2") if k in free)


def fit_fixed(q, y, p_cont, key, value, others, bounds) -> tuple[dict, float]:
    """Deviance minimized over `others` at `key` = value: the lowest of three local fits, from the
    previous profile point (continuation), with the nuisances at refine's initial values, and with
    them at the midpoints of their bounds. One continued local fit alone only bounds PL from above
    (physics review of PR #24)."""
    nuis = [k for k in others if k in NUIS]
    starts = [
        p_cont,
        p_cont | {k: REFINE_INIT[k] for k in nuis},
        p_cont | {k: 0.5 * (bounds[k][0] + bounds[k][1]) for k in nuis},
    ]
    best = None
    for start in starts:
        pp, dev, _ = fit(q, y, start | {key: value}, others, bounds, 100)
        if best is None or dev < best[1]:
            best = (pp, dev)
    return best


def half_width(q, y, p, d_p, key, free, bounds, threshold: float = 1.0) -> dict:
    """Profile Δχ² = 1 half-width of `key` at endpoint p (deviance d_p): step out from p doubling
    the step until Δχ² - (d_p) > threshold or the bound is hit, then bisect 6 times; mean of both
    sides. A side that reaches its bound inside the threshold counts as the distance to the bound
    and is flagged. lowest_dchi2 < 0 means a restart found a lower point than the endpoint itself."""
    others = tuple(k for k in free if k != key)
    lo_b, hi_b = bounds[key]
    x0 = p[key]
    sides, lowest = [], 0.0
    for sign in (1, -1):
        limit = (hi_b - x0) if sign > 0 else (x0 - lo_b)
        step, inner, inner_p, outer = 1e-4 * max(abs(x0), 1e-3), 0.0, p, None
        while step < limit:
            pp, dev = fit_fixed(q, y, inner_p, key, x0 + sign * step, others, bounds)
            lowest = min(lowest, dev - d_p)
            if dev - d_p > threshold:
                outer = step
                break
            inner, inner_p, step = step, pp, 2 * step
        if outer is None:
            sides.append({"width": limit, "at_bound": True})
            continue
        for _ in range(6):
            mid = 0.5 * (inner + outer)
            pp, dev = fit_fixed(q, y, inner_p, key, x0 + sign * mid, others, bounds)
            lowest = min(lowest, dev - d_p)
            if dev - d_p <= threshold:
                inner, inner_p = mid, pp
            else:
                outer = mid
        sides.append({"width": 0.5 * (inner + outer), "at_bound": False})
    return {
        "half_width": float(np.mean([s["width"] for s in sides])),
        "at_bound": any(s["at_bound"] for s in sides),
        "lowest_dchi2": float(lowest),
    }


def analyze(index: int) -> dict:
    """Group the endpoints with Δχ² ≤ 4 into modes (registered rule 2), profile each mode's film
    parameters, label each film parameter (rule 3); also the nearest distinct competitor above
    Δχ² 4 and the counting level at which it would come within Δχ² 1 (addition)."""
    data = json.loads((SCAN_DIR / f"fits_{index:02d}.json").read_text(encoding="utf-8"))
    cond, p0, free = data["condition"], data["truth"], tuple(data["free"])
    _, _, bounds = setup(cond)
    q = q_grid(cond["tth_max"])
    y = expected_counts(torch.as_tensor(q), {k: torch.tensor(v) for k, v in p0.items()}).numpy()
    keys = film_keys(free)
    ends = sorted(data["endpoints"], key=lambda e: e["deviance"])

    widths: dict[int, dict[str, dict]] = {}

    def hw(e) -> dict[str, dict]:
        if e["start"] not in widths:
            widths[e["start"]] = {
                k: half_width(q, y, e["params"], e["deviance"], k, free, bounds) for k in keys
            }
        return widths[e["start"]]

    def distinct(a, b) -> bool:
        """Different modes: some film parameter differs by more than 3 × the larger half-width.
        b's own widths are only computed when a's alone would call them distinct."""
        wa = hw(a)
        far = [k for k in keys if abs(a["params"][k] - b["params"][k]) > 3 * wa[k]["half_width"]]
        if not far:
            return False
        wb = hw(b)
        return any(
            abs(a["params"][k] - b["params"][k]) > 3 * max(wa[k]["half_width"], wb[k]["half_width"])
            for k in far
        )

    modes: list[dict] = []
    for e in (e for e in ends if e["deviance"] <= 4):
        for m in modes:
            if not distinct(m["best"], e):
                m["members"].append(e["start"])
                break
        else:
            modes.append({"best": e, "members": [e["start"]]})

    out_modes = [
        {
            "deviance": m["best"]["deviance"],
            "start": m["best"]["start"],
            "film": {k: m["best"]["params"][k] for k in keys},
            "half_widths": hw(m["best"]),
            "members": m["members"],
        }
        for m in modes
    ]
    labels = {}
    within = [m for m in out_modes if m["deviance"] <= 1]
    for k in keys:
        if not out_modes:
            # no endpoint within Δχ² ≤ 4 (step 2), so no mode and no label (amendment 7)
            labels[k] = "not reached"
        elif len(within) > 1 and any(
            abs(a["film"][k] - b["film"][k])
            > 3 * max(a["half_widths"][k]["half_width"], b["half_widths"][k]["half_width"])
            for a, b in itertools.combinations(within, 2)
        ):
            labels[k] = "multimodal"
        elif (best := (within or out_modes)[0])["half_widths"][k]["half_width"] >= 0.02 * abs(
            best["film"][k]
        ):
            labels[k] = "ridge"
        else:
            labels[k] = "identifiable"

    # Amendment 7 observation, post hoc and outside the decision: with no mode, a local fit from the
    # truth gives the width of the truth's basin. It sees only that basin, so it says nothing about
    # multimodality.
    seeded = None
    refs = [m["best"] for m in modes]
    if not modes:
        p_s, d_s, _ = fit(q, y, p0, free, bounds)
        e_s = {"start": -1, "deviance": d_s, "params": p_s}
        w = hw(e_s)
        seeded = {
            "deviance": d_s,
            "film": {k: p_s[k] for k in keys},
            "half_widths": w,
            "labels": {
                k: "ridge" if w[k]["half_width"] >= 0.02 * abs(p_s[k]) else "identifiable"
                for k in keys
            },
        }
        refs = [e_s]

    # Addition: the lowest endpoint above Δχ² 4 that is distinct from every mode (from the seeded
    # fit when there is no mode).
    competitor = None
    if refs:
        for e in (e for e in ends if e["deviance"] > 4):
            if all(distinct(r, e) for r in refs):
                competitor = {
                    "deviance": e["deviance"],
                    "film": {k: e["params"][k] for k in keys},
                    "s_star": S_COUNTS / e["deviance"],
                }
                break
    # Amendment 5 observation: endpoints within Δχ² ≤ 4 whose nuisances differ from the best one.
    basins = []
    if ends and ends[0]["deviance"] <= 4:
        b0 = ends[0]["params"]
        for e in (e for e in ends[1:] if e["deviance"] <= 4):
            p = e["params"]
            if abs(p["kalpha2_ratio"] - b0["kalpha2_ratio"]) > 0.1 or any(
                abs(p[k] / b0[k] - 1) > 0.1 for k in ("oxide_thickness", "oxide_sld") if b0[k]
            ):
                basins.append(
                    {"deviance": e["deviance"]}
                    | {k: p[k] for k in ("kalpha2_ratio", "oxide_thickness", "oxide_sld")}
                )
    return {
        "index": index,
        "condition": cond,
        "nuisance_basins": basins,
        "n_starts": len(ends),
        "n_within_4": sum(e["deviance"] <= 4 for e in ends),
        "modes": out_modes,
        "labels": labels,
        "seeded": seeded,
        "competitor": competitor,
    }


def report() -> None:
    fitted = sorted(SCAN_DIR.glob("fits_*.json"))
    lines = [
        "# R2a condition scan (generated by scan.py report; do not edit)",
        "",
        "Noise-free data at 10⁶ counts. Reached: starts ending within Δχ² ≤ 4 of the truth."
        " Modes: groups of those endpoints (registered rule 2); Δχ² of each mode's best endpoint."
        " Labels per film parameter (d, ρ; layer 2 after the slash). Competitor: the lowest endpoint"
        " above Δχ² 4 distinct from every mode, and s*, the counting level at which it would be"
        " within Δχ² 1 (not part of the registered decision). At cap: fits of the 64 that stopped at"
        " the 200-evaluation limit. Not reached: no start ended within Δχ² ≤ 4, so the condition has"
        " no mode and no label (amendment 7); it is evidence neither for nor against multimodality."
        " In brackets, the width labels of a local fit from the truth: post hoc, outside the"
        " decision, and blind to multimodality because it only sees the truth's basin (not run in"
        " R2a, so no brackets appear). Not analysed: fitted, but `analyze` was not run for the"
        " condition.",
        "",
        "| # | stack | nuisance | 2θ max | ρ film | reached | at cap | modes (Δχ²) | d | ρ"
        " | competitor ΔD | s* |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for path in fitted:
        fits = json.loads(path.read_text(encoding="utf-8"))
        n = len(fits["endpoints"])
        reach = sum(e["deviance"] <= 4 for e in fits["endpoints"])
        at_cap = sum(e["nfev"] >= MAX_NFEV for e in fits["endpoints"])
        c = fits["condition"]
        head = (
            f"| {fits['index']} | {c['stack']} | {c['nuisance']} | {c['tth_max']:g}° | {c['film_sld']:g}"
            f" | {reach}/{n} | {at_cap}/{n}"
        )
        modes_path = SCAN_DIR / path.name.replace("fits_", "modes_")
        if not modes_path.exists():
            state = "not reached, not analysed" if reach == 0 else "not analysed"
            lines.append(f"{head} | — | {state} | {state} | — | — |")
            continue
        r = json.loads(modes_path.read_text(encoding="utf-8"))
        lab, seeded = r["labels"], r.get("seeded")

        def cell(k1: str, k2: str, lab=lab, seeded=seeded) -> str:
            ks = [k for k in (k1, k2) if k in lab]
            text = " / ".join(lab[k] for k in ks)
            if seeded:
                text += " (" + " / ".join(seeded["labels"][k] for k in ks) + ")"
            return text

        d, rho = cell("thickness", "thickness2"), cell("sld", "sld2")
        comp = r["competitor"]
        modes = ", ".join(f"{m['deviance']:.2g}" for m in r["modes"]) or "—"
        comp_d = f"{comp['deviance']:.3g}" if comp else "—"
        comp_s = f"{comp['s_star']:.2g}" if comp else "—"
        lines.append(f"{head} | {modes} | {d} | {rho} | {comp_d} | {comp_s} |")
    (OUT / "scan_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("part", choices=["check", "run", "analyze", "report", "list"])
    ap.add_argument("--conditions", nargs="+", type=int, default=[])
    ap.add_argument("--starts", type=int, default=N_STARTS)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.part == "check":
        print(f"bilayer vs benchmark.parratt, max |ΔR/R|: {check_vs_parratt():.2e}")
        return
    if args.part == "list":
        for i, c in enumerate(conditions()):
            print(i, c)
        return
    if args.part == "report":
        report()
        return
    if args.part == "analyze":
        for i in args.conditions:
            t = time.perf_counter()
            r = analyze(i)
            (SCAN_DIR / f"modes_{i:02d}.json").write_text(
                json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8"
            )
            print(f"condition {i}: {time.perf_counter() - t:.0f} s, labels {r['labels']}")
        return
    SCAN_DIR.mkdir(exist_ok=True)
    for i in args.conditions:
        t = time.perf_counter()
        r = run_condition(i, args.starts)
        (SCAN_DIR / f"fits_{i:02d}.json").write_text(
            json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8"
        )
        best = sorted(e["deviance"] for e in r["endpoints"])[:5]
        print(f"condition {i}: {time.perf_counter() - t:.0f} s, lowest D {best}")


if __name__ == "__main__":
    main()
