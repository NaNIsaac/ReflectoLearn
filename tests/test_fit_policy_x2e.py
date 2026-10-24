"""X2e additions (docs/fit_policy/X2E.md section 4): the peak rule, S1 and S2 starts, the true-L
data start, and the D2 search."""

import numpy as np
import pytest

from reflectolearn.fit_policy import baselines
from reflectolearn.fit_policy import stack as st
from reflectolearn.fit_policy.baselines import de_fit, step3_state, true_l_data_start
from reflectolearn.fit_policy.env import RESIDUAL_Z, FitEnv
from reflectolearn.fit_policy.protocol import S2_Z, S12, _s2_stacks, run

TRUTH = st.Stack(
    np.array([[120.0, 4.0, 40.0], [60.0, 3.0, 12.0]]), np.array([1.0, -7.5, 0.0, 0.002])
)


def _data():
    q = st.q_grid()
    held = st.held_out_mask(q)
    y = np.random.default_rng(13).poisson(st.counts(TRUTH, q, 1e6)).astype(float)
    return q[~held], y[~held], q[held], y[held]


def test_peak_rule():
    q, y, _, _ = _data()
    values = np.log10(np.maximum(y, st.LOG_FLOOR_COUNTS))
    qc = st.critical_q(q, y)
    peaks = st.fringe_peaks(q, values, 1.1 * qc, S2_Z)
    assert 1 <= len(peaks) <= st.PEAK_MAX
    assert peaks[0] == st.fringe_spectrum(q, values, 1.1 * qc, S2_Z)[0]
    keep = q > 1.1 * qc
    separation = 2 * np.pi / (q[keep].max() - q[keep].min())
    assert all(abs(a - b) >= separation for a in peaks for b in peaks if a != b)
    assert st.fringe_peaks(q, values, 1.1 * qc, RESIDUAL_Z, k=3)[:1] == [
        st.fringe_spectrum(q, values, 1.1 * qc, RESIDUAL_Z)[0]
    ]


def test_s2_stacks_follow_the_registered_rules():
    q, y, q_ho, y_ho = _data()
    env = FitEnv(q, y, 1e6, st.starts(9, 0)[0], q_ho=q_ho, y_ho=y_ho)
    for a in [("open", "bkg"), ("fit", 20), ("estimate_rho",), ("estimate_d",)]:
        env.step(a)
    stacks = _s2_stacks(env)
    assert 1 <= len(stacks) <= 7
    base = env.state.stack
    for s in stacks:
        assert 1 <= s.n_layers <= st.L_MAX
        assert s.layers[:, 0].min() >= 5.0
        assert s.layers[0, 2] == base.layers[0, 2]
        for i in range(1, s.n_layers):
            assert s.layers[i, 2] == pytest.approx(0.5 * (s.layers[i - 1, 2] + st.SUB_SLD))


def test_s12_episode_records_its_exposure_and_stays_in_budget():
    q, y, q_ho, y_ho = _data()
    env = FitEnv(q, y, 1e6, st.starts(9, 1)[0], q_ho=q_ho, y_ho=y_ho, budget=800)
    final = run(env, S12)
    res = final.result()
    assert res["evaluations"] <= 800
    assert isinstance(final.episode.s2_changed, bool)
    assert all(isinstance(c, bool) for c in final.episode.s1_changed)


def test_true_l_data_start():
    q, y, _, _ = _data()
    step3 = step3_state(st.starts(9, 2)[0], q, y, 1e6)
    start = true_l_data_start(step3, 3)
    assert start.n_layers == 3
    assert start.layers[:, 0].sum() == pytest.approx(step3.layers[0, 0])
    assert start.layers[1, 2] == pytest.approx(0.5 * (start.layers[0, 2] + st.SUB_SLD))


def test_de_fit_runs_inside_its_bounds(monkeypatch):
    monkeypatch.setattr(baselines, "DE_MAXITER", 2)
    q, y, q_ho, y_ho = _data()
    start = true_l_data_start(step3_state(st.starts(9, 3)[0], q, y, 1e6), 2)
    res = de_fit(start, q, y, 1e6, (1, 2, 3), q_ho=q_ho, y_ho=y_ho)
    lo, hi = start.bounds()
    x = np.concatenate([np.ravel(res["layers"]), res["instr"]])
    assert np.all(x >= lo - 1e-9) and np.all(x <= hi + 1e-9)
    assert res["de_evaluations"] > 0 and res["n_layers"] == 2
