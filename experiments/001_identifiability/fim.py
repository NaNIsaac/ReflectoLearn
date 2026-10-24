"""Experiment 001: local identifiability of a single film from one XRR curve.

Question: at the true structure, which combinations of film thickness d, roughness sigma and SLD rho
(and the nuisances refine fits alongside them) does one reflectivity curve fail to pin down, and how
does that change with the measured q range and the counting statistics?

Model (stated, not fitted):
    ambient / [top oxide] / film / Si substrate (SLD 20.07, roughness 3 Å, both known).
    counts(q) = s * (i0 * R_res(q_eff) + 10**bkg), Poisson, as in data.augmentations (step 3-5).
    q_eff: 2θ zero offset applied as in physics.apply_tth_offset.
    R_res: R convolved with a Gaussian of constant sigma dq in q (refine's `dq`).
    Not included: footprint (assumed corrected), Cu Kα2 (assumed monochromated), masked angles.
    The measured grid is uniform in 2θ (0.01° steps from 0.2°), as on a lab diffractometer.

Free-parameter sets (scenarios):
    film   d, sigma, rho; everything else known
    instr  film + i0, bkg, tth_offset, dq
    oxide  instr + oxide thickness, SLD, roughness (truth has a 10 Å top oxide, so this curve differs)

For each case the Jacobian J = d ln(counts) / d theta is taken by forward-mode autograd through
physics.abeles_core in complex128, and the Fisher information of Poisson counts is
F = J^T diag(counts) J. Reported:
    CRLB      sqrt(diag F^-1): the smallest standard deviation any unbiased estimator can reach
    corr      correlations of F^-1
    s_min     smallest singular value of the whitened Jacobian with unit-norm columns (scale-free)
    gamma     1 / s_min, the collinearity index of Brun et al. (2001); > 10-20 means near-degenerate
    weakest   right singular vector of s_min, in those normalized coordinates

All of this is local (one point, linearized): a small s_min is evidence of a weak direction, a large
one does not rule out a distant second solution (that is R1c).

Independent checks (also in tests/test_identifiability.py):
    the forward model against benchmark.parratt, a float64 numpy Parratt recursion;
    the autograd Jacobian against central differences of a numpy re-implementation of the counts;
    the forward model against refine.make_forward (the float32 model refine and MCMC fit with).

Run:  uv run python experiments/001_identifiability/fim.py [--out docs/identifiability]
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from scipy.special import erf
from torch import Tensor
from torch.func import jacfwd

from reflectolearn.benchmark import parratt
from reflectolearn.physics import abeles_core, apply_tth_offset

torch.set_default_dtype(torch.float64)

WAVELENGTH = 1.5406  # Å, Cu Kα1
WAVELENGTH_KA2 = 1.54439  # Å, Cu Kα2 (RefineConfig.kalpha2_wavelength)
SUB_SLD, SUB_ROUGH = 20.07, 3.0  # Si, known
TTH_MIN, TTH_STEP = 0.2, 0.01  # deg
N_RES, RES_WIDTH = 41, 5.0  # quadrature of the resolution Gaussian: points, half-width in sigma

FILM = ("thickness", "roughness", "sld")
INSTR = ("i0", "bkg", "tth_offset", "dq")
OXIDE = ("oxide_thickness", "oxide_sld", "oxide_roughness")
SCENARIOS = {"film": FILM, "instr": FILM + INSTR, "oxide": FILM + INSTR + OXIDE}
SYMBOL = {
    "thickness": "d",
    "roughness": "σ",
    "sld": "ρ",
    "i0": "i0",
    "bkg": "bkg",
    "tth_offset": "off",
    "dq": "dq",
    "oxide_thickness": "d_ox",
    "oxide_sld": "ρ_ox",
    "oxide_roughness": "σ_ox",
    "kalpha2_ratio": "Kα2",
    "footprint": "fp",
}


@dataclass(frozen=True)
class Film:
    name: str
    thickness: float  # Å
    roughness: float  # Å
    sld: float  # 1e-6 Å^-2


# README / benchmark fixed_mid film, benchmark fixed_thin film, and a film barely above the substrate.
FILMS = (
    Film("mid", 240.0, 6.0, 45.0),
    Film("thin", 90.0, 3.0, 28.0),
    Film("low_contrast", 240.0, 6.0, 24.0),
)
TRUE_INSTR = {"i0": 1.0, "bkg": -7.0, "tth_offset": 0.0, "dq": 0.002}
TRUE_OXIDE = {"oxide_thickness": 10.0, "oxide_sld": 18.9, "oxide_roughness": 3.0}
TTH_MAX = (3.0, 6.0, 12.0)  # deg -> q_max 0.21, 0.43, 0.85 Å^-1
COUNTS = (1e5, 1e6, 1e7)  # s: counts at R = 1 (augmentation.poisson_log_scale is 5-8)
BKG = (-8.0, -7.0, -6.0)  # log10 of the background relative to R = 1 (augmentation.bkg is -8 to -5)


def tth_grid(tth_max: float) -> np.ndarray:
    return np.arange(TTH_MIN, tth_max + TTH_STEP / 2, TTH_STEP)


def tth_to_q(tth: np.ndarray) -> np.ndarray:
    return 4 * np.pi / WAVELENGTH * np.sin(np.radians(tth / 2))


def truth(film: Film, scenario: str, bkg: float = TRUE_INSTR["bkg"]) -> dict[str, float]:
    p = {"thickness": film.thickness, "roughness": film.roughness, "sld": film.sld}
    p |= TRUE_INSTR | {"bkg": bkg}
    p |= TRUE_OXIDE if scenario == "oxide" else dict.fromkeys(OXIDE, 0.0)
    return p


def effects_of(p0: dict[str, float]) -> frozenset[str]:
    """Optional parts of the model switched on by the truth: a top oxide, Kα2, a footprint."""
    on = set()
    if p0["oxide_thickness"] != 0:
        on.add("oxide")
    if p0.get("kalpha2_ratio", 0.0) > 0:
        on.add("kalpha2")
    if "footprint" in p0:
        on.add("footprint")
    return frozenset(on)


# ------------------------------------------------------------------ forward model (torch, float64)
def reflectivity(q: Tensor, p: dict[str, Tensor], oxide: bool, absorption: float = 1e-9) -> Tensor:
    """R(q) of ambient / top oxide / film / substrate through abeles_core, prepared like AbelesMatrix
    (SLD relative to the ambient, 1e-6 units, +1e-9j Å^-2 absorption in every medium) but in
    complex128."""
    zero = torch.zeros_like(p["sld"])
    sld = torch.stack([zero, p["oxide_sld"], p["sld"], zero + SUB_SLD])
    thk = torch.stack([zero, p["oxide_thickness"], p["thickness"]])
    rgh = torch.stack([p["oxide_roughness"], p["roughness"], zero + SUB_ROUGH])
    if not oxide:
        # no oxide: same 3-media stack the generator builds for an oxide-free film
        sld, thk, rgh = sld[[0, 2, 3]], thk[[0, 2]], rgh[[1, 2]]
    sld_c = sld.to(torch.complex128) * 1e-6 + 1j * absorption
    q_c = q.to(torch.complex128).reshape(1, -1, 1)
    return abeles_core(q_c, thk.reshape(1, 1, -1), rgh.reshape(1, 1, -1), sld_c.reshape(1, 1, -1))[
        0
    ]


def expected_counts(
    q: Tensor,
    p: dict[str, Tensor],
    s: float,
    effects: frozenset[str],
    absorption: float = 1e-9,
) -> Tensor:
    """Mean Poisson counts at the measured q: offset, [Kα2, footprint,] constant-dq resolution, i0,
    background, in the order refine.make_forward applies them."""
    q_eff = apply_tth_offset(q, p["tth_offset"].reshape(1, 1), WAVELENGTH)[0]
    u = torch.linspace(-RES_WIDTH, RES_WIDTH, N_RES)
    w = torch.exp(-0.5 * u**2)
    w = w / w.sum()
    qq = (q_eff[:, None] + p["dq"] * u[None, :]).reshape(-1)
    oxide = "oxide" in effects
    r = reflectivity(qq, p, oxide, absorption)
    if "kalpha2" in effects:
        r2 = reflectivity(qq * (WAVELENGTH / WAVELENGTH_KA2), p, oxide, absorption)
        r = (r + p["kalpha2_ratio"] * r2) / (1 + p["kalpha2_ratio"])
    if "footprint" in effects:
        r = r * torch.erf(p["footprint"] * (qq * WAVELENGTH / (4 * math.pi)).clamp_min(0))
    r = (r.reshape(-1, N_RES) * w).sum(1)
    return s * (p["i0"] * r + 10.0 ** p["bkg"])


def jacobian(
    q: np.ndarray, p0: dict[str, float], free: tuple[str, ...], s: float, absorption: float = 1e-9
):
    """(counts (Q,), J = d ln counts / d theta (Q, P)) at p0, theta = the `free` parameters."""
    qt = torch.as_tensor(q)
    fixed = {k: torch.tensor(v) for k, v in p0.items()}
    effects = effects_of(p0)

    def log_counts(theta: Tensor) -> Tensor:
        p = fixed | dict(zip(free, theta, strict=True))
        return torch.log(expected_counts(qt, p, s, effects, absorption))

    theta0 = torch.tensor([p0[k] for k in free])
    lam = torch.exp(log_counts(theta0)).numpy()
    return lam, jacfwd(log_counts)(theta0).numpy()


# ------------------------------------------------------------------ numpy reference (finite differences)
def expected_counts_np(q: np.ndarray, p: dict[str, float], s: float) -> np.ndarray:
    """Same counts model on benchmark.parratt (numpy, float64), written without torch."""
    sin_th = q * WAVELENGTH / (4 * np.pi)
    tth = 2 * np.degrees(np.arcsin(sin_th)) + p["tth_offset"]
    q_eff = 4 * np.pi / WAVELENGTH * np.sin(np.radians(tth / 2))
    layers = [(0j, 0.0, 0.0)]
    if p["oxide_thickness"] != 0:
        layers.append((complex(p["oxide_sld"]), p["oxide_thickness"], p["oxide_roughness"]))
    layers += [
        (complex(p["sld"]), p["thickness"], p["roughness"]),
        (complex(SUB_SLD), 0.0, SUB_ROUGH),
    ]
    u = np.linspace(-RES_WIDTH, RES_WIDTH, N_RES)
    w = np.exp(-0.5 * u**2)
    w /= w.sum()
    qq = q_eff[:, None] + p["dq"] * u[None, :]
    r = parratt(qq.ravel(), layers).reshape(qq.shape)
    if p.get("kalpha2_ratio", 0.0) > 0:
        r2 = parratt(qq.ravel() * WAVELENGTH / WAVELENGTH_KA2, layers).reshape(qq.shape)
        r = (r + p["kalpha2_ratio"] * r2) / (1 + p["kalpha2_ratio"])
    if "footprint" in p:
        r = r * erf(p["footprint"] * np.clip(qq * WAVELENGTH / (4 * np.pi), 0, None))
    return s * (p["i0"] * (r * w).sum(1) + 10.0 ** p["bkg"])


FD_STEP = {
    "thickness": 1e-3,
    "roughness": 1e-4,
    "sld": 1e-4,
    "i0": 1e-6,
    "bkg": 1e-6,
    "tth_offset": 1e-6,
    "dq": 1e-7,
    "oxide_thickness": 1e-4,
    "oxide_sld": 1e-4,
    "oxide_roughness": 1e-4,
    "kalpha2_ratio": 1e-6,
    "footprint": 1e-3,
}


def jacobian_fd(q: np.ndarray, p0: dict[str, float], free: tuple[str, ...], s: float) -> np.ndarray:
    cols = []
    for k in free:
        h = FD_STEP[k]
        up = expected_counts_np(q, p0 | {k: p0[k] + h}, s)
        dn = expected_counts_np(q, p0 | {k: p0[k] - h}, s)
        cols.append((np.log(up) - np.log(dn)) / (2 * h))
    return np.stack(cols, 1)


# ------------------------------------------------------------------ Fisher analysis
@dataclass
class Fisher:
    free: tuple[str, ...]
    F: np.ndarray
    crlb: np.ndarray = field(init=False)
    corr: np.ndarray = field(init=False)
    s_min: float = field(init=False)
    weakest: np.ndarray = field(init=False)

    def __post_init__(self):
        dsq = np.sqrt(np.diag(self.F))
        N = self.F / np.outer(dsq, dsq)  # = Jn^T Jn, Jn = whitened J with unit-norm columns
        evals, evecs = np.linalg.eigh(N)
        self.s_min = math.sqrt(max(evals[0], 0.0))
        v = evecs[:, 0]
        self.weakest = v * np.sign(v[np.argmax(np.abs(v))])
        cov = np.linalg.inv(N) / np.outer(dsq, dsq)
        self.crlb = np.sqrt(np.diag(cov))
        self.corr = cov / np.outer(self.crlb, self.crlb)

    @property
    def gamma(self) -> float:
        return 1.0 / self.s_min

    def weakest_str(self, cut: float = 0.25) -> str:
        order = np.argsort(-np.abs(self.weakest))
        terms = [
            f"{self.weakest[i]:+.2f} {SYMBOL[self.free[i]]}"
            for i in order
            if abs(self.weakest[i]) >= cut
        ]
        return " ".join(terms)

    def max_corr(self) -> tuple[str, float]:
        c = np.abs(self.corr - np.eye(len(self.free)))
        i, j = np.unravel_index(np.argmax(c), c.shape)
        return f"{SYMBOL[self.free[i]]}–{SYMBOL[self.free[j]]}", float(self.corr[i, j])


def fisher(lam: np.ndarray, J: np.ndarray, free: tuple[str, ...]) -> Fisher:
    return Fisher(free, J.T @ (lam[:, None] * J))


def crlb_vs_qmax(q: np.ndarray, lam: np.ndarray, J: np.ndarray, free) -> np.ndarray:
    """Relative CRLB of the film parameters for every truncation q <= q[i]: (Q, 3), NaN where singular."""
    outer = np.cumsum(lam[:, None, None] * J[:, :, None] * J[:, None, :], axis=0)
    out = np.full((len(q), 3), np.nan)
    for i in range(len(q)):
        F = outer[i]
        dsq = np.sqrt(np.diag(F))
        if np.any(dsq == 0):
            continue
        N = F / np.outer(dsq, dsq)
        if np.linalg.cond(N) > 1e14:
            continue
        cov = np.linalg.inv(N) / np.outer(dsq, dsq)
        out[i] = np.sqrt(np.diag(cov)[:3])
    return out


# ------------------------------------------------------------------ checks
def check_forward_vs_parratt(film: Film = FILMS[0], absorption: float = 0.0) -> float:
    """Max relative difference of R between abeles_core (complex128) and benchmark.parratt.

    benchmark.parratt has no counterpart of the engine's 1e-9j absorption in the ambient, so the
    physics is compared with absorption=0; the default 1e-9 shows what that term alone changes."""
    q = tth_to_q(tth_grid(12.0))
    p = {k: torch.tensor(v) for k, v in truth(film, "film").items()}
    r_torch = reflectivity(torch.as_tensor(q), p, oxide=False, absorption=absorption).numpy()
    layers = [(0j, 0.0, 0.0), (complex(film.sld), film.thickness, film.roughness)]
    r_np = parratt(q, layers + [(complex(SUB_SLD), 0.0, SUB_ROUGH)])
    return float(np.max(np.abs(r_torch / r_np - 1)))


def check_jacobian_vs_fd(film: Film = FILMS[0], scenario: str = "oxide", tth_max: float = 6.0):
    """Per-parameter relative error ||J_autograd - J_fd|| / ||J_fd||, without absorption on both sides."""
    q = tth_to_q(tth_grid(tth_max))
    p0, free = truth(film, scenario), SCENARIOS[scenario]
    _, J = jacobian(q, p0, free, 1e6, absorption=0.0)
    J_fd = jacobian_fd(q, p0, free, 1e6)
    err = np.linalg.norm(J - J_fd, axis=0) / np.linalg.norm(J_fd, axis=0)
    return dict(zip(free, err.tolist(), strict=True))


def check_vs_refine_model(film: Film = FILMS[0]) -> float:
    """Max |Δ log10 R| against refine.make_forward (float32, grid-convolution smearing) on a q grid
    uniform in q, away from the grid edges where make_forward pads."""
    from reflectolearn.refine.pipeline import RefineConfig, make_forward

    cfg = RefineConfig(fit_kalpha2=False, fit_footprint=False, fit_oxide=False)
    q = np.linspace(tth_to_q(np.array([0.2]))[0], 0.43, 1500)
    p0 = truth(film, "instr") | {"tth_offset": 0.01}
    fwd = make_forward(
        torch.as_tensor(q, dtype=torch.float32),
        torch.tensor([[SUB_SLD, SUB_ROUGH]], dtype=torch.float32),
        wavelength=WAVELENGTH,
        config=cfg,
        device=torch.device("cpu"),
    )
    vals = {k: torch.tensor([v], dtype=torch.float32) for k, v in p0.items()}
    r_refine = fwd(vals, {"tth_offset", "dq"})[0].double().numpy()
    p = {k: torch.tensor(v) for k, v in p0.items()}
    r_here = expected_counts(torch.as_tensor(q), p, 1.0, frozenset()).numpy()
    edge = int(math.ceil(3 * cfg.dq_bounds[1] / (q[1] - q[0]))) + 1
    d = np.abs(np.log10(r_here) - np.log10(r_refine))[edge:-edge]
    return float(d.max())


# ------------------------------------------------------------------ experiment
def run(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rows = []
    cache = {}
    for film in FILMS:
        for scenario, free in SCENARIOS.items():
            for tth_max in TTH_MAX:
                q = tth_to_q(tth_grid(tth_max))
                for bkg in BKG:
                    p0 = truth(film, scenario, bkg)
                    for s in COUNTS:
                        lam, J = jacobian(q, p0, free, s)
                        fi = fisher(lam, J, free)
                        cache[film.name, scenario, tth_max, bkg, s] = (q, lam, J, fi)
                        pair, c = fi.max_corr()
                        rows.append(
                            {
                                "film": film.name,
                                "scenario": scenario,
                                "q_max": round(float(q[-1]), 3),
                                "log10_bkg": bkg,
                                "log10_counts": round(math.log10(s), 1),
                                "n_points": len(q),
                                **{
                                    f"crlb_{SYMBOL[k]}_pct": 100 * fi.crlb[i] / p0[k]
                                    for i, k in enumerate(FILM)
                                },
                                "s_min": fi.s_min,
                                "gamma": fi.gamma,
                                "weakest": fi.weakest_str(),
                                "max_corr_pair": pair,
                                "max_corr": c,
                                "crlb_offset_deg": (
                                    fi.crlb[free.index("tth_offset")]
                                    if "tth_offset" in free
                                    else ""
                                ),
                            }
                        )
    elapsed = time.perf_counter() - t0

    with open(out / "fim_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    checks = {
        "forward_vs_parratt": check_forward_vs_parratt(),
        "absorption_term": check_forward_vs_parratt(absorption=1e-9),
        "jacobian_vs_fd": check_jacobian_vs_fd(),
        "vs_refine_model": check_vs_refine_model(),
    }
    (out / "fim_tables.md").write_text(tables(rows, checks, elapsed, cache), encoding="utf-8")
    figure(cache, out / "fim.png")
    print((out / "fim_tables.md").read_text(encoding="utf-8"))


def tables(rows, checks, elapsed, cache) -> str:
    def fmt(r, cols):
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f"{v:.3g}" if isinstance(v, float) else str(v))
        return "| " + " | ".join(cells) + " |"

    cols = [
        "q_max",
        "log10_bkg",
        "crlb_d_pct",
        "crlb_σ_pct",
        "crlb_ρ_pct",
        "s_min",
        "gamma",
        "weakest",
        "max_corr_pair",
        "max_corr",
    ]
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols)
    jac = ", ".join(f"{SYMBOL[k]} {v:.1e}" for k, v in checks["jacobian_vs_fd"].items())
    lines = [
        "# Experiment 001 tables (generated by fim.py; do not edit)",
        "",
        f"Grid: {len(rows)} cases in {elapsed:.1f} s (CPU, float64).",
        "",
        "## Checks",
        "",
        f"- R(q), abeles_core complex128 vs benchmark.parratt (numpy), 2θ 0.2–12°: "
        f"max relative difference {checks['forward_vs_parratt']:.1e} without the engine's "
        f"1e-9j Å^-2 absorption term, {checks['absorption_term']:.1e} with it (the term is kept "
        "in the model, as in AbelesMatrix)",
        f"- J, autograd vs central differences of the numpy model (mid film, oxide scenario, "
        f"q_max 0.43): relative column error {jac}",
        f"- counts model vs refine.make_forward (float32), offset 0.01°, dq 0.002: max |Δ log10 R| "
        f"{checks['vs_refine_model']:.1e}",
        "",
        "CRLB in % of the true value, at s = 1e6 counts (CRLB scales exactly as s^-1/2 in this "
        "model; see the last table). s_min and gamma = 1/s_min are scale-free; weakest is the "
        "direction of s_min in normalized coordinates (components ≥ 0.25 shown).",
    ]
    for film in FILMS:
        for scenario in SCENARIOS:
            sel = [
                r
                for r in rows
                if r["film"] == film.name and r["scenario"] == scenario and r["log10_counts"] == 6.0
            ]
            lines += [
                "",
                f"## {film.name} (d {film.thickness:g} Å, σ {film.roughness:g} Å, "
                f"ρ {film.sld:g}), free: {scenario} ({', '.join(SYMBOL[k] for k in SCENARIOS[scenario])})",
                "",
                head,
                *(fmt(r, cols) for r in sel),
            ]
    sel = [
        r
        for r in rows
        if r["film"] == "mid" and r["scenario"] == "instr" and r["log10_bkg"] == -7.0
    ]
    lines += [
        "",
        "## Counting statistics (mid, instr, bkg 1e-7)",
        "",
        "| q_max | log10_counts | crlb_d_pct | crlb_σ_pct | crlb_ρ_pct | gamma | max_corr |",
        "|---|---|---|---|---|---|---|",
        *(
            fmt(
                r,
                [
                    "q_max",
                    "log10_counts",
                    "crlb_d_pct",
                    "crlb_σ_pct",
                    "crlb_ρ_pct",
                    "gamma",
                    "max_corr",
                ],
            )
            for r in sel
        ),
        "",
    ]
    return "\n".join(lines)


def figure(cache, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    ink, ink2, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    series = {"thickness": "#2a78d6", "roughness": "#eb6834", "sld": "#1baf7a"}
    films = {"mid": "#2a78d6", "thin": "#eb6834", "low_contrast": "#1baf7a"}
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

    # (a) CRLB of d, sigma, rho vs the upper end of the measured range, mid film, bkg 1e-7, s 1e6
    for scenario, ls in (("film", "-"), ("instr", "--")):
        q, lam, J, _ = cache["mid", scenario, 12.0, -7.0, 1e6]
        rel = crlb_vs_qmax(q, lam, J, SCENARIOS[scenario])
        p0 = truth(FILMS[0], scenario)
        for i, k in enumerate(FILM):
            a.plot(
                q,
                100 * rel[:, i] / p0[k],
                ls,
                color=series[k],
                lw=2 if ls == "-" else 1.6,
                label=f"{SYMBOL[k]}, {'film only' if scenario == 'film' else '+ i0, bkg, offset, dq'}",
            )
    a.set_yscale("log")
    a.set_ylim(1e-3, 1e2)
    a.set_xlabel("q_max of the measured range (Å⁻¹)")
    a.set_ylabel("CRLB (% of true value)")
    a.set_title("(a) mid film: bound vs measured range", loc="left", color=ink)
    a.legend(frameon=False, fontsize=7.5)

    # (b) correlation matrix, mid film, instr scenario, q_max 0.43
    fi = cache["mid", "instr", 6.0, -7.0, 1e6][3]
    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
    off_diag = np.where(np.eye(len(fi.free), dtype=bool), np.nan, fi.corr)
    cmap.set_bad(grid)
    b.imshow(off_diag, cmap=cmap, vmin=-1, vmax=1)
    labels = [SYMBOL[k] for k in fi.free]
    b.set_xticks(range(len(labels)), labels)
    b.set_yticks(range(len(labels)), labels)
    b.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j:
                b.text(
                    j, i, f"{fi.corr[i, j]:.2f}", ha="center", va="center", fontsize=7, color=ink
                )
    b.set_title("(b) mid film, q_max 0.43: corr(F⁻¹), blue −1 / red +1", loc="left", color=ink)

    # (c) collinearity index vs q_max for the three films
    for name, col in films.items():
        for scenario, ls in (("instr", "-"), ("oxide", ":")):
            q, lam, J, _ = cache[name, scenario, 12.0, -7.0, 1e6]
            g = []
            for i in range(len(q)):
                F = (lam[: i + 1, None] * J[: i + 1]).T @ J[: i + 1]
                dsq = np.sqrt(np.diag(F))
                ev = np.linalg.eigvalsh(F / np.outer(dsq, dsq))[0] if np.all(dsq > 0) else 0.0
                g.append(1 / math.sqrt(ev) if ev > 0 else np.nan)
            c.plot(q, g, ls, color=col, lw=2 if ls == "-" else 1.6, label=f"{name}, {scenario}")
    c.set_yscale("log")
    c.set_ylim(1, 1e4)
    c.axhline(20, color=ink2, lw=0.8)
    c.text(0.84, 22, "γ = 20", color=ink2, fontsize=7.5, ha="right")
    c.set_xlabel("q_max of the measured range (Å⁻¹)")
    c.set_ylabel("collinearity index γ = 1 / s_min")
    c.set_title("(c) near-degeneracy vs measured range", loc="left", color=ink)
    c.legend(frameon=False, fontsize=7.5)
    for ax in (a, c):
        ax.set_xlim(0.06, 0.86)

    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("docs/identifiability"))
    run(ap.parse_args().out)


if __name__ == "__main__":
    main()
