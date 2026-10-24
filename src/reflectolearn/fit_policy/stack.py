"""N-layer stack, counting data and fit kernel of the fit-policy environment.

Section numbers refer to docs/fit_policy/DESIGN.md. The stack is ambient / layers 1..L (top first) /
Si substrate with Névot–Croce roughness, computed by abeles_core in complex128, with intensity scale,
background, constant Gaussian resolution (15-node Gauss–Hermite) and a 2θ zero offset.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.special import xlogy
from scipy.stats import qmc
from torch import Tensor
from torch.func import jacfwd

from reflectolearn.physics import abeles_core, apply_tth_offset
from reflectolearn.refine import fft_thickness

WAVELENGTH = 1.5406  # Å, Cu Kα1
SUB_SLD, SUB_ROUGH = 20.07, 3.0  # Si, known
ABSORPTION = 1e-9
SEED = 20261024
L_MAX = 4

LAYER_KEYS = ("d", "sigma", "rho")
INSTR_KEYS = ("i0", "bkg", "offset", "dq")
LAYER_BOUNDS = {"d": (5.0, 400.0), "sigma": (0.0, 15.0), "rho": (0.0, 120.0)}
INSTR_BOUNDS = {"i0": (0.5, 2.0), "bkg": (-10.0, -4.0), "offset": (-0.08, 0.08), "dq": (0.0, 0.012)}
INSTR_START = np.array([1.0, -7.0, 0.0, 0.002])

NOISE_SCALES = (1e8, 1e6)  # counts at R = 1, noise level s = 0, 1
START_BOX = np.array([[20.0, 250.0], [2.0, 10.0], [5.0, 80.0]])  # d, sigma, rho of a start layer
N_STARTS = 4
TTH_HELD_MIN = 0.5  # deg; below it every point stays in training
HELD_BLOCKS = (2, 6)
N_BLOCKS = 10

QC_MARGIN = 1.05  # estimate_rho acts only if q_c exceeds the substrate's by this factor
KIESSIG_RANGE = (20.0, 800.0)  # Å; three-layer truths reach 750 Å (amendment 2)

LOSSES = ("log", "raw", "poisson")
LOG_FLOOR_COUNTS = 0.5

_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(15)
GH_U, GH_W = torch.as_tensor(_GH_X), torch.as_tensor(_GH_W / _GH_W.sum())


def tth_to_q(tth):
    return 4 * np.pi / WAVELENGTH * np.sin(np.radians(np.asarray(tth) / 2))


def q_grid() -> np.ndarray:
    """2θ 0.2–6° as 291 points uniform in q (the 0.02° step count)."""
    n = round((6.0 - 0.2) / 0.02) + 1
    return np.linspace(tth_to_q(0.2), tth_to_q(6.0), n)


INNER_BLOCK = 4  # X2d A1v and Cv: the training block midway between held-out blocks 2 and 6


def inner_mask(q: np.ndarray) -> np.ndarray:
    """X2d's inner validation block on the full grid (block INNER_BLOCK of the held-out split)."""
    above = np.flatnonzero(q >= tth_to_q(TTH_HELD_MIN))
    inner = np.zeros(q.shape, bool)
    inner[np.array_split(above, N_BLOCKS)[INNER_BLOCK]] = True
    return inner


def held_out_mask(q: np.ndarray) -> np.ndarray:
    """Contiguous held-out blocks (section 3.2): points above 2θ 0.5° in N_BLOCKS equal blocks,
    HELD_BLOCKS held out."""
    above = np.flatnonzero(q >= tth_to_q(TTH_HELD_MIN))
    held = np.zeros(q.shape, bool)
    for b in HELD_BLOCKS:
        held[np.array_split(above, N_BLOCKS)[b]] = True
    return held


# ------------------------------------------------------------------ stack
@dataclass(frozen=True)
class Stack:
    """layers: (L, 3) d [Å], sigma of the top interface [Å], rho [1e-6 Å^-2], top first.
    instr: (4,) i0, log10 background, 2θ offset [deg], resolution sigma_q [Å^-1]."""

    layers: np.ndarray
    instr: np.ndarray

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def n_params(self) -> int:
        return 3 * self.n_layers + len(INSTR_KEYS)

    def vector(self) -> np.ndarray:
        return np.concatenate([self.layers.ravel(), self.instr])

    def with_vector(self, x: np.ndarray) -> Stack:
        n = 3 * self.n_layers
        return Stack(
            np.asarray(x[:n], float).reshape(-1, 3).copy(), np.asarray(x[n:], float).copy()
        )

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lay = np.array([LAYER_BOUNDS[k] for k in LAYER_KEYS])
        ins = np.array([INSTR_BOUNDS[k] for k in INSTR_KEYS])
        lo = np.concatenate([np.tile(lay[:, 0], self.n_layers), ins[:, 0]])
        hi = np.concatenate([np.tile(lay[:, 1], self.n_layers), ins[:, 1]])
        return lo, hi

    def clipped(self) -> Stack:
        lo, hi = self.bounds()
        return self.with_vector(np.clip(self.vector(), lo, hi))


# ------------------------------------------------------------------ forward model
def reflectivity(q: Tensor, layers: Tensor, absorption: float = ABSORPTION) -> Tensor:
    """R(q) of the stack; layers (L, 3) as Stack.layers. q real, any shape (flattened)."""
    zero = layers.new_zeros(1)
    sld = torch.cat([zero, layers[:, 2], zero + SUB_SLD]).to(torch.complex128) * 1e-6
    thk = torch.cat([zero, layers[:, 0]])
    rgh = torch.cat([layers[:, 1], zero + SUB_ROUGH])
    q_c = q.to(torch.complex128).reshape(1, -1, 1)
    return abeles_core(
        q_c, thk.reshape(1, 1, -1), rgh.reshape(1, 1, -1), (sld + 1j * absorption).reshape(1, 1, -1)
    )[0]


def expected_counts(q: Tensor, x: Tensor, n_layers: int, scale: float) -> Tensor:
    """lambda(q) for the packed vector x = Stack.vector()."""
    layers = x[: 3 * n_layers].reshape(n_layers, 3)
    i0, bkg, offset, dq = x[3 * n_layers :]
    q_eff = apply_tth_offset(q, offset.reshape(1, 1), WAVELENGTH)[0]
    qq = (q_eff[:, None] + dq * GH_U[None, :]).reshape(-1)
    r = (reflectivity(qq, layers).reshape(-1, len(GH_U)) * GH_W).sum(1)
    return scale * (i0 * r + 10.0**bkg)


def counts(stack: Stack, q: np.ndarray, scale: float) -> np.ndarray:
    with torch.no_grad():
        x = torch.as_tensor(stack.vector(), dtype=torch.float64)
        return expected_counts(torch.as_tensor(q), x, stack.n_layers, scale).numpy()


# ------------------------------------------------------------------ fit quality
def deviance_terms(lam: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Poisson deviance per point, zero counts included.

    Clamped at 0, its exact lower bound: for lam within ~1e-8 (relative) of y, rounding leaves an
    error of order y * 1e-16 that can be negative, and the Poisson residual's sqrt would be NaN (#43).
    """
    return np.maximum(2.0 * (lam - y + xlogy(y, y / lam)), 0.0)


def deviance(lam: np.ndarray, y: np.ndarray) -> float:
    return float(deviance_terms(lam, y).sum())


def bic(dev: float, n_layers: int, n: int) -> float:
    """BIC = D + k ln n with k = 3L + 4: every layer and instrument parameter (section 4.6)."""
    return dev + (3 * n_layers + len(INSTR_KEYS)) * math.log(n)


def aic(dev: float, n_layers: int) -> float:
    return dev + 2 * (3 * n_layers + len(INSTR_KEYS))


def deviance_excess(lam: np.ndarray, y: np.ndarray, draws: int = 200, seed: int = 0) -> float:
    """(D - E[D]) / SD[D], with E and SD from Poisson draws at lam (section 5.1 step 7)."""
    rng = np.random.default_rng((SEED, 99, seed))
    sim = rng.poisson(lam, size=(draws, len(lam)))
    d_sim = deviance_terms(lam[None, :], sim).sum(1)
    return (deviance(lam, y) - d_sim.mean()) / max(d_sim.std(), 1e-12)


def residuals(loss: str, lam: np.ndarray, y: np.ndarray, scale: float) -> np.ndarray:
    """Residuals whose sum of squares the fit minimizes (section 4.3)."""
    match loss:
        case "log":
            return np.log10(lam) - np.log10(np.maximum(y, LOG_FLOOR_COUNTS))
        case "raw":
            return (lam - y) / scale
        case "poisson":
            return np.sign(lam - y) * np.sqrt(deviance_terms(lam, y))
    raise ValueError(loss)


def residual_jacobian(
    loss: str, lam: np.ndarray, y: np.ndarray, scale: float, jlog: np.ndarray
) -> np.ndarray:
    """d residual / d theta from jlog = d ln lambda / d theta."""
    match loss:
        case "log":
            return jlog / math.log(10)
        case "raw":
            return (lam / scale)[:, None] * jlog
        case "poisson":
            r = np.sign(lam - y) * np.sqrt(deviance_terms(lam, y))
            near = np.abs(lam - y) <= 1e-6 * np.maximum(y, 1e-12)
            g = np.where(near, 1 / np.sqrt(lam), (1 - y / lam) / np.where(near, 1.0, r))
            return (g * lam)[:, None] * jlog
    raise ValueError(loss)


# ------------------------------------------------------------------ fit kernel
def fit(
    stack: Stack,
    free: np.ndarray,
    q: np.ndarray,
    y: np.ndarray,
    scale: float,
    loss: str,
    max_nfev: int,
) -> tuple[Stack, int, int, int]:
    """Trust-region least squares over the parameters where `free` (bool, Stack.vector() order) is
    set, at most max_nfev evaluations. Returns (stack, nfev, njev, least_squares status); status 0
    means nfev reached max_nfev. Every residual evaluation also computes the Jacobian at that point."""
    idx = np.flatnonzero(free)
    if idx.size == 0 or max_nfev <= 0:
        return stack, 0, 0, 0
    lo_all, hi_all = stack.bounds()
    lo, hi = lo_all[idx], hi_all[idx]
    x_all = stack.vector()
    x0 = np.clip(x_all[idx], lo + 1e-9 * (hi - lo), hi - 1e-9 * (hi - lo))
    qt = torch.as_tensor(q)
    base = torch.as_tensor(x_all, dtype=torch.float64)
    idx_t = torch.as_tensor(idx)

    def log_counts(theta: Tensor) -> Tensor:
        return torch.log(
            expected_counts(qt, base.index_put((idx_t,), theta), stack.n_layers, scale)
        )

    jac_fn = jacfwd(log_counts)
    cache: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}

    def model(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        key = x.tobytes()
        if key not in cache:
            cache.clear()
            theta = torch.as_tensor(x, dtype=torch.float64)
            with torch.no_grad():
                lam = torch.exp(log_counts(theta)).numpy()
            cache[key] = lam, jac_fn(theta).numpy()
        return cache[key]

    def resid(x: np.ndarray) -> np.ndarray:
        return residuals(loss, model(x)[0], y, scale)

    def jac(x: np.ndarray) -> np.ndarray:
        lam, jlog = model(x)
        return residual_jacobian(loss, lam, y, scale, jlog)

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
    x_new = x_all.copy()
    x_new[idx] = res.x
    return stack.with_vector(x_new), int(res.nfev), int(res.njev or 0), int(res.status)


# ------------------------------------------------------------------ estimators
def critical_q(q: np.ndarray, y: np.ndarray) -> float:
    """q where the counts (3-point mean) first fall to half the plateau after it."""
    ys = np.convolve(y, np.ones(3) / 3, mode="same")
    ys[0], ys[-1] = y[0], y[-1]
    top = int(np.argmax(ys[: max(len(ys) // 10, 3)]))
    below = np.flatnonzero(ys[top:] < 0.5 * ys[top])
    if below.size == 0:
        return float("nan")
    k = top + int(below[0])
    if k == 0:
        return float(q[0])
    f = (ys[k - 1] - 0.5 * ys[top]) / (ys[k - 1] - ys[k])
    return float(q[k - 1] + f * (q[k] - q[k - 1]))


def kiessig_thickness(
    q: np.ndarray, y: np.ndarray, scale: float, rho: float, d_range: tuple[float, float]
) -> float:
    """Total thickness from the fringe period (refine.fft_thickness over the film wavevector)."""
    ok = y > 0
    log_r = np.where(ok, np.log10(np.maximum(y, 1) / scale), 0.0)
    d = fft_thickness(
        torch.as_tensor(q),
        torch.as_tensor(log_r)[None],
        torch.as_tensor(ok, dtype=torch.float64)[None],
        d_range,
        sld=torch.tensor([rho]),
        d_min=d_range[0],
    )
    return float(d[0])


def estimate_rho(stack: Stack, q: np.ndarray, y: np.ndarray) -> Stack:
    """Protocol A step 2 (`estimate_rho`): top-layer rho = q_c^2 / 16 pi from the data's critical q,
    unless q_c is within QC_MARGIN of the substrate's."""
    qc = critical_q(q, y)
    if not (math.isfinite(qc) and qc > QC_MARGIN * 4 * math.sqrt(math.pi * SUB_SLD * 1e-6)):
        return stack
    layers = stack.layers.copy()
    layers[0, 2] = qc**2 / (16 * math.pi) * 1e6
    return Stack(layers, stack.instr.copy()).clipped()


def estimate_d(stack: Stack, q: np.ndarray, y: np.ndarray, scale: float) -> Stack:
    """Protocol A step 3 (`estimate_d`): scale every layer's d so that their sum is the Kiessig
    thickness (computed with the top layer's rho); unchanged if there is none."""
    total = kiessig_thickness(q, y, scale, stack.layers[0, 2], KIESSIG_RANGE)
    if not math.isfinite(total):
        return stack
    layers = stack.layers.copy()
    layers[:, 0] *= total / layers[:, 0].sum()
    return Stack(layers, stack.instr.copy()).clipped()


def fringe_spectrum(
    q: np.ndarray, values: np.ndarray, q_min: float, z: np.ndarray
) -> tuple[float, float]:
    """(z of the strongest local maximum, its relative amplitude) of the detrended, Hann-windowed
    transform of `values` over q > q_min; (nan, 0) if there are fewer than 40 points or no maximum."""
    keep = q > q_min
    if keep.sum() < 40:
        return float("nan"), 0.0
    qq, vv = q[keep], values[keep]
    vv = vv - np.polyval(np.polyfit(qq, vv, 5), qq)
    vv = vv * np.hanning(vv.size)
    amp = np.abs(np.exp(-1j * np.outer(z, qq)) @ vv)
    peak = (amp[1:-1] > amp[:-2]) & (amp[1:-1] >= amp[2:])
    if not peak.any():
        return float("nan"), 0.0
    i = int(np.argmax(np.where(peak, amp[1:-1], -np.inf)))
    return float(z[1:-1][i]), float(amp[1:-1][i] / max(np.abs(vv).sum(), 1e-12))


PEAK_REL = 0.25  # X2e 4.2: a peak needs this fraction of the largest one
PEAK_MAX = 4


def fringe_peaks(
    q: np.ndarray, values: np.ndarray, q_min: float, z: np.ndarray, k: int = PEAK_MAX
) -> list[float]:
    """X2e's peak rule (X2E.md 4.2) on `fringe_spectrum`'s transform: local maxima with at least
    PEAK_REL of the largest one, taken in decreasing amplitude and kept if at least
    2 pi / (q_max - q_min) from every peak already kept, at most k. Returns their z, strongest
    first (so the first is `fringe_spectrum`'s peak); [] with fewer than 40 points or no maximum."""
    keep = q > q_min
    if keep.sum() < 40:
        return []
    qq, vv = q[keep], values[keep]
    vv = vv - np.polyval(np.polyfit(qq, vv, 5), qq)
    vv = vv * np.hanning(vv.size)
    amp = np.abs(np.exp(-1j * np.outer(z, qq)) @ vv)
    inner = np.flatnonzero((amp[1:-1] > amp[:-2]) & (amp[1:-1] >= amp[2:])) + 1
    if inner.size == 0:
        return []
    inner = inner[np.argsort(-amp[inner], kind="stable")]
    floor = PEAK_REL * amp[inner[0]]
    separation = 2 * np.pi / (qq.max() - qq.min())
    kept: list[float] = []
    for i in inner:
        if amp[i] < floor or len(kept) == k:
            break
        if all(abs(z[i] - zk) >= separation for zk in kept):
            kept.append(float(z[i]))
    return kept


# ------------------------------------------------------------------ registered data (section 3.1, 5.4)
def sample_truth(n_layers: int, j: int, root: int = SEED) -> Stack:
    """Truth j of class n_layers; adjacent media differ by |delta rho| >= 4 (redrawn otherwise)."""
    rng = np.random.default_rng((root, n_layers, j))
    while True:
        d = rng.uniform(20.0, 250.0, n_layers)
        sigma = rng.uniform(2.0, np.minimum(10.0, d / 4))
        rho = rng.uniform(5.0, 80.0, n_layers)
        if np.all(np.abs(np.diff(np.append(rho, SUB_SLD))) >= 4.0):
            break
    instr = np.array(
        [
            rng.uniform(0.8, 1.2),
            rng.uniform(-8.5, -7.0),
            rng.uniform(-0.01, 0.01),
            rng.uniform(0.0005, 0.003),
        ]
    )
    return Stack(np.column_stack([d, sigma, rho]), instr)


def noisy_counts(
    lam: np.ndarray, n_layers: int, j: int, s: int, r: int = 0, root: int = SEED
) -> np.ndarray:
    """Realisation r of truth (n_layers, j) at noise level s; r = 0 is the fitted one."""
    return np.random.default_rng((root, n_layers, j, s, r)).poisson(lam).astype(float)


def _box(u: np.ndarray) -> np.ndarray:
    return START_BOX[:, 0] + u * (START_BOX[:, 1] - START_BOX[:, 0])


def starts(n_layers: int, j: int, root: int = SEED) -> list[Stack]:
    """The N_STARTS one-layer starts of truth (n_layers, j) (amendment 1 seed)."""
    u = qmc.LatinHypercube(d=3, rng=np.random.default_rng((root, n_layers, j, 2))).random(N_STARTS)
    return [Stack(_box(row)[None, :], INSTR_START.copy()) for row in u]


def extra_layers(n_layers: int, j: int, root: int = SEED) -> np.ndarray:
    """(N_STARTS, 3, 3): the layers baseline C appends below start k for L = 2..4."""
    u = qmc.LatinHypercube(d=3, rng=np.random.default_rng((root, n_layers, j, 1))).random(12)
    return _box(u).reshape(N_STARTS, 3, 3)
