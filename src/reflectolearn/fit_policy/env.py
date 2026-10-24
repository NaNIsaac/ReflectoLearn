"""The fit-policy environment: state, actions, transition and reward (DESIGN.md section 4 and
amendment 2).

An episode starts from a one-layer stack with only the intensity scale open and the log loss. Every
fit consumes evaluations from the episode's budget; the episode keeps the lowest-BIC state it has
seen, which is the returned model. `FitEnv.fork` makes a copy of the state that shares the episode
(budget and best state), for scripted protocols that try candidates (amendment 2).

X2d (X2D.md section 4) adds two options that leave the default unchanged. With `kept_path`, `revert`
and the returned model use the lowest-BIC state created on this env's own lineage (itself and the
forks it was made from), not the episode-wide best. With `fit_mask`, fits, BIC and every quantity a
policy reads use only the masked training points; the rest (A1v's inner block) is scored separately.

Held-out points are stored on the episode for the terminal reward and the final score only; nothing
a policy reads (observation, BIC, residual features) uses them.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from reflectolearn.fit_policy import stack as st
from reflectolearn.fit_policy.stack import L_MAX, LAYER_BOUNDS, LAYER_KEYS, LOSSES, Stack

Action = tuple

INSTR_GROUPS = ("bkg", "offset", "dq")  # i0 is always open
FIT_STEPS = (20, 100)
BUDGET = 3000
MAX_ACTIONS = 60
N_BANDS = 8
NEW_LAYER_SIGMA = 3.0
NEW_LAYER_D = 50.0
SPLIT_CONTRAST = 0.05
RESIDUAL_Z = np.arange(20.0, 401.0, 1.0)


def _actions() -> tuple[Action, ...]:
    acts: list[Action] = [("add", p) for p in range(L_MAX)]
    acts += [("merge", i) for i in range(L_MAX - 1)]
    acts += [("split", i) for i in range(L_MAX)]
    acts += [(op, g, ell) for op in ("open", "close") for g in LAYER_KEYS for ell in range(L_MAX)]
    acts += [(op, g) for op in ("open_all", "close_all") for g in LAYER_KEYS]
    acts += [(op, g) for op in ("open", "close") for g in INSTR_GROUPS]
    acts += [("loss", name) for name in LOSSES]
    acts += [("fit", k) for k in FIT_STEPS]
    acts += [("estimate_rho",), ("estimate_d",), ("revert",), ("stop",)]
    return tuple(acts)


ACTIONS = _actions()
ACTION_INDEX = {a: i for i, a in enumerate(ACTIONS)}
LAYER_CHANGES = ("add", "merge", "split", "revert")
_ENV_IDS = itertools.count()


@dataclass
class FitState:
    stack: Stack
    open_layers: np.ndarray  # (L, 3) bool, LAYER_KEYS order
    open_instr: np.ndarray  # (3,) bool, INSTR_GROUPS order
    loss: str = "log"
    origin: int = -1  # id of the env whose action created this state

    def free_mask(self) -> np.ndarray:
        """Bool mask in Stack.vector() order: layers, then i0 (always), bkg, offset, dq."""
        return np.concatenate([self.open_layers.ravel(), [True], self.open_instr])

    def copy(self) -> FitState:
        return FitState(
            self.stack, self.open_layers.copy(), self.open_instr.copy(), self.loss, self.origin
        )


@dataclass
class Episode:
    """What forks of one episode share."""

    q: np.ndarray
    y: np.ndarray
    scale: float
    q_ho: np.ndarray | None
    y_ho: np.ndarray | None
    budget: int
    used: int = 0  # nfev, the budget unit
    jacobians: int = 0  # njev, reported
    best: FitState | None = None
    best_bic: float = math.inf
    kiessig: float = field(default=math.nan)
    qc_data: float = field(default=math.nan)
    decisions: list = field(default_factory=list)  # X2d A1m step 7: [L, dBIC unmatched, matched]
    s1_changed: list = field(default_factory=list)  # X2e S1: chosen start differs from A1's
    s2_changed: bool | None = None  # X2e S2: chosen start is not A1's step-3 state


class FitEnv:
    def __init__(
        self,
        q: np.ndarray,
        y: np.ndarray,
        scale: float,
        start: Stack,
        *,
        q_ho: np.ndarray | None = None,
        y_ho: np.ndarray | None = None,
        budget: int = BUDGET,
        max_actions: int = MAX_ACTIONS,
        kept_path: bool = False,
        fit_mask: np.ndarray | None = None,
    ):
        self.episode = Episode(q, y, scale, q_ho, y_ho, budget)
        self.episode.qc_data = st.critical_q(q, y)
        self.max_actions = max_actions
        self.kept_path = kept_path
        self.fit_mask = None if fit_mask is None else np.asarray(fit_mask, bool)
        self.uid = next(_ENV_IDS)
        self.lineage = frozenset({self.uid})
        self.path_best: FitState | None = None
        self.path_best_bic = math.inf
        self.path_best_all: FitState | None = None
        self.path_best_all_bic = math.inf
        # (origin, L, deviance on all training, stack) of every evaluated state
        self.history: list[tuple[int, int, float, Stack]] = []
        self.state = FitState(
            start.clipped(),
            np.zeros((start.n_layers, 3), bool),
            np.zeros(len(INSTR_GROUPS), bool),
            origin=self.uid,
        )
        top_rho = start.layers[0, 2]
        self.episode.kiessig = st.kiessig_thickness(q, y, scale, top_rho, st.KIESSIG_RANGE)
        self.trace: list[Action] = []
        self.layer_changes = 0
        self.done = False
        self._evaluate()

    # ------------------------------------------------------------ shared data
    @property
    def q(self) -> np.ndarray:
        """The points fits and policies see: all training points, or the fit mask's."""
        return self.episode.q if self.fit_mask is None else self.episode.q[self.fit_mask]

    @property
    def y(self) -> np.ndarray:
        return self.episode.y if self.fit_mask is None else self.episode.y[self.fit_mask]

    @property
    def scale(self) -> float:
        return self.episode.scale

    @property
    def budget_left(self) -> int:
        return self.episode.budget - self.episode.used

    @property
    def best(self) -> FitState:
        """The lowest-BIC state of the episode (set from the start state on)."""
        assert self.episode.best is not None
        return self.episode.best

    def fork(self) -> FitEnv:
        other = object.__new__(FitEnv)
        other.__dict__ |= self.__dict__
        other.state = self.state.copy()
        other.trace = list(self.trace)
        other.history = list(self.history)
        other.uid = next(_ENV_IDS)
        other.lineage = self.lineage | {other.uid}
        return other

    def use_all_points(self) -> None:
        """Drop the fit mask (A1v step 9: the inner block rejoins the fits)."""
        self.fit_mask = None
        self._evaluate()

    def set_stack(self, stack: Stack, open_layers: np.ndarray | None = None) -> None:
        """Replace the current stack (X2e S1 and S2 starts). A scripted-protocol operation, like
        amendment 2's estimates, and not part of the action space. New layers are open by default."""
        s = self.state
        s.stack = stack.clipped()
        s.open_layers = (
            np.ones((stack.n_layers, 3), bool) if open_layers is None else open_layers.copy()
        )
        s.origin = self.uid
        self._evaluate()

    def kept_state_at(self, n_layers: int) -> Stack | None:
        """The last state on this env's kept path with the given layer count (X2e D1)."""
        for origin, L, _, stack in reversed(self.history):
            if origin in self.lineage and L == n_layers:
                return stack
        return None

    # ------------------------------------------------------------ quality
    def _evaluate(self) -> None:
        s = self.state.stack
        if self.fit_mask is None:
            self.lam = st.counts(s, self.q, self.scale)
            self.dev = st.deviance(self.lam, self.y)
            self.dev_all, self.dev_inner = self.dev, math.nan
        else:
            lam_all = st.counts(s, self.episode.q, self.scale)
            self.lam = lam_all[self.fit_mask]
            self.dev = st.deviance(self.lam, self.y)
            self.dev_all = st.deviance(lam_all, self.episode.y)
            self.dev_inner = st.deviance(lam_all[~self.fit_mask], self.episode.y[~self.fit_mask])
        self.bic = st.bic(self.dev, s.n_layers, len(self.q))
        self.bic_all = st.bic(self.dev_all, s.n_layers, len(self.episode.q))
        if self.bic < self.episode.best_bic:
            self.episode.best_bic = self.bic
            self.episode.best = self.state.copy()
        if self.state.origin in self.lineage:
            if self.bic < self.path_best_bic:
                self.path_best_bic, self.path_best = self.bic, self.state.copy()
            if self.bic_all < self.path_best_all_bic:
                self.path_best_all_bic, self.path_best_all = self.bic_all, self.state.copy()
        self.history.append((self.state.origin, s.n_layers, self.dev_all, s))

    def excess(self) -> float:
        """(D - E[D]) / SD[D] at the current model (protocol A step 7)."""
        return st.deviance_excess(self.lam, self.y)

    def _residual_period(self) -> tuple[float, float]:
        resid = np.log10(np.maximum(self.y, st.LOG_FLOOR_COUNTS)) - np.log10(self.lam)
        return st.fringe_spectrum(self.q, resid, 1.1 * self._qc(), RESIDUAL_Z)

    def _qc(self) -> float:
        qc = self.episode.qc_data
        return qc if math.isfinite(qc) else 4 * math.sqrt(math.pi * st.SUB_SLD * 1e-6)

    # ------------------------------------------------------------ actions
    def valid(self, action: Action) -> bool:
        if self.done:
            return False
        s, L = self.state, self.state.stack.n_layers
        match action:
            case ("add", p):
                return L < L_MAX and p <= L
            case ("merge", i):
                return L >= 2 and i <= L - 2
            case ("split", i):
                return L < L_MAX and i < L and s.stack.layers[i, 0] >= 2 * LAYER_BOUNDS["d"][0]
            case ("open" | "close" as op, g, ell):
                return ell < L and bool(s.open_layers[ell, LAYER_KEYS.index(g)]) != (op == "open")
            case ("open_all" | "close_all" as op, g):
                return bool(np.any(s.open_layers[:, LAYER_KEYS.index(g)] != (op == "open_all")))
            case ("open" | "close" as op, g):
                return bool(s.open_instr[INSTR_GROUPS.index(g)]) != (op == "open")
            case ("loss", name):
                return name != s.loss
            case ("fit", _):
                return self.budget_left > 0
            case ("estimate_rho",) | ("estimate_d",) | ("stop",):
                return True
            case ("revert",):
                target = self._revert_target()
                return target is not None and not _same_stack(target.stack, s.stack)
        raise ValueError(action)

    def _revert_target(self) -> FitState | None:
        """The episode-wide best, or with kept_path the best on this env's lineage."""
        return self.path_best if self.kept_path else self.episode.best

    def returned(self) -> FitState:
        """The returned model: the episode-wide lowest BIC (DESIGN 4.4), or with kept_path the
        lowest BIC on all training points among the states of this env's lineage."""
        if self.kept_path:
            assert self.path_best_all is not None
            return self.path_best_all
        return self.best

    def valid_actions(self) -> list[Action]:
        return [a for a in ACTIONS if self.valid(a)]

    def mask(self) -> np.ndarray:
        return np.array([self.valid(a) for a in ACTIONS])

    def step(self, action: Action) -> tuple[float, bool]:
        """Apply an action. Returns (reward, done)."""
        if not self.valid(action):
            raise ValueError(f"invalid action {action}")
        phi = -self.episode.best_bic / len(self.q)
        self._apply(action)
        if action != ("revert",):
            self.state.origin = self.uid
        self.trace.append(action)
        if action[0] in LAYER_CHANGES:
            self.layer_changes += 1
        self._evaluate()
        reward = -self.episode.best_bic / len(self.q) - phi  # gamma = 1 until X2c sets it
        if action == ("stop",) or self.budget_left <= 0 or len(self.trace) >= self.max_actions:
            self.done = True
            reward += self.terminal_reward()
        return reward, self.done

    def _apply(self, action: Action) -> None:
        s = self.state
        lay = s.stack.layers.copy()
        match action:
            case ("add", p):
                period, _ = self._residual_period()
                d = period if math.isfinite(period) else NEW_LAYER_D
                below = lay[p, 2] if p < len(lay) else st.SUB_SLD
                rho = lay[0, 2] if p == 0 else 0.5 * (lay[p - 1, 2] + below)
                lay = np.insert(lay, p, [d, NEW_LAYER_SIGMA, rho], axis=0)
                s.open_layers = np.insert(s.open_layers, p, True, axis=0)
            case ("merge", i):
                d1, d2 = lay[i, 0], lay[i + 1, 0]
                merged = [d1 + d2, lay[i, 1], (d1 * lay[i, 2] + d2 * lay[i + 1, 2]) / (d1 + d2)]
                lay = np.vstack([lay[:i], [merged], lay[i + 2 :]])
                opened = s.open_layers[i] | s.open_layers[i + 1]
                s.open_layers = np.vstack([s.open_layers[:i], [opened], s.open_layers[i + 2 :]])
            case ("split", i):
                d, sigma, rho = lay[i]
                top = [d / 2, sigma, rho * (1 + SPLIT_CONTRAST)]
                bottom = [d / 2, NEW_LAYER_SIGMA, rho * (1 - SPLIT_CONTRAST)]
                lay = np.vstack([lay[:i], [top, bottom], lay[i + 1 :]])
                s.open_layers = np.insert(s.open_layers, i, s.open_layers[i], axis=0)
            case ("open" | "close" as op, g, ell):
                s.open_layers[ell, LAYER_KEYS.index(g)] = op == "open"
            case ("open_all" | "close_all" as op, g):
                s.open_layers[:, LAYER_KEYS.index(g)] = op == "open_all"
            case ("open" | "close" as op, g):
                s.open_instr[INSTR_GROUPS.index(g)] = op == "open"
            case ("loss", name):
                s.loss = name
            case ("fit", k):
                stack, nfev, njev, _ = st.fit(
                    s.stack,
                    s.free_mask(),
                    self.q,
                    self.y,
                    self.scale,
                    s.loss,
                    min(k, self.budget_left),
                )
                self.episode.used += nfev
                self.episode.jacobians += njev
                s.stack = stack
                return
            case ("estimate_rho",):
                s.stack = st.estimate_rho(s.stack, self.q, self.y)
                return
            case ("estimate_d",):
                s.stack = st.estimate_d(s.stack, self.q, self.y, self.scale)
                return
            case ("revert",):
                target = self._revert_target()
                assert target is not None
                self.state = target.copy()
                return
            case ("stop",):
                return
        s.stack = Stack(lay, s.stack.instr.copy()).clipped()

    # ------------------------------------------------------------ reward and result
    def terminal_reward(self) -> float:
        """-D_ho / n_ho of the returned model; 0 without held-out data (section 4.6)."""
        if self.episode.q_ho is None:
            return 0.0
        return -self.held_out_deviance() / len(self.episode.q_ho)

    def held_out_deviance(self) -> float:
        lam = st.counts(self.returned().stack, self.episode.q_ho, self.scale)
        return st.deviance(lam, self.episode.y_ho)

    def result(self) -> dict:
        """The returned model, its fit quality on all training points, and the episode's cost.
        `on_kept_path` and `kept_path_states` (L, training deviance) are X2d's instrumentation."""
        state = self.returned()
        best = state.stack
        n = len(self.episode.q)
        dev = st.deviance(st.counts(best, self.episode.q, self.scale), self.episode.y)
        out = {
            "layers": best.layers.tolist(),
            "instr": best.instr.tolist(),
            "n_layers": best.n_layers,
            "bic": st.bic(dev, best.n_layers, n),
            "aic": st.aic(dev, best.n_layers),
            "deviance": dev,
            "evaluations": self.episode.used,
            "jacobians": self.episode.jacobians,
            "actions": len(self.trace),
            "trace": [list(a) for a in self.trace],
            "on_kept_path": state.origin in self.lineage,
            "kept_path_states": [
                [n_layers, d] for origin, n_layers, d, _ in self.history if origin in self.lineage
            ],
        }
        if self.episode.q_ho is not None:
            out["deviance_ho"] = self.held_out_deviance()
        return out

    # ------------------------------------------------------------ observation
    def observe(self) -> np.ndarray:
        """Section 4.1 as one float vector; training points only."""
        s = self.state
        lay_lo = np.array([LAYER_BOUNDS[k][0] for k in LAYER_KEYS])
        lay_hi = np.array([LAYER_BOUNDS[k][1] for k in LAYER_KEYS])
        layers = np.zeros((L_MAX, 4))
        layers[: s.stack.n_layers, :3] = (s.stack.layers - lay_lo) / (lay_hi - lay_lo)
        layers[: s.stack.n_layers, 3] = 1.0
        ins_b = np.array([st.INSTR_BOUNDS[k] for k in st.INSTR_KEYS])
        instr = (s.stack.instr - ins_b[:, 0]) / (ins_b[:, 1] - ins_b[:, 0])
        open_layers = np.zeros((L_MAX, 3))
        open_layers[: s.stack.n_layers] = s.open_layers
        loss = np.array([s.loss == name for name in LOSSES], float)

        n = len(self.q)
        qc = self._qc()
        pois = st.residuals("poisson", self.lam, self.y, self.scale)
        logr = st.residuals("log", self.lam, self.y, self.scale)
        edges = np.linspace(1.1 * qc, self.q.max(), N_BANDS + 1)
        bands = np.zeros((N_BANDS, 3))
        for b in range(N_BANDS):
            sel = (self.q >= edges[b]) & (self.q <= edges[b + 1])
            if sel.any():
                bands[b] = pois[sel].mean(), np.sqrt((pois[sel] ** 2).mean()), logr[sel].mean()
        period, amp = self._residual_period()
        best_bic = self.episode.best_bic
        qmax = self.q.max()
        scalars = [
            s.stack.n_layers / L_MAX,
            np.nan_to_num(period / RESIDUAL_Z[-1]),
            amp,
            np.nan_to_num(self.episode.kiessig / st.KIESSIG_RANGE[1]),
            np.nan_to_num(self.episode.qc_data / qmax),
            np.nan_to_num(st.critical_q(self.q, self.lam) / qmax),
            self.dev / n,
            self.excess(),
            self.bic / n,
            best_bic / n,
            len(self.trace) / self.max_actions,
            self.episode.used / self.episode.budget,
            (ACTION_INDEX[self.trace[-1]] + 1) / len(ACTIONS) if self.trace else 0.0,
            self.layer_changes / 10,
        ]
        return np.concatenate(
            [
                layers.ravel(),
                instr,
                open_layers.ravel(),
                s.open_instr.astype(float),
                loss,
                bands.ravel(),
                np.asarray(scalars, float),
            ]
        )


def _same_stack(a: Stack, b: Stack) -> bool:
    return a.n_layers == b.n_layers and np.array_equal(a.vector(), b.vector())
