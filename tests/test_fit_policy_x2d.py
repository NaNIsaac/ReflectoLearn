"""X2d additions to the fit-policy environment (docs/fit_policy/X2D.md section 4): kept-path
revert and return, the inner block, matched refits, Cv and TF."""

import numpy as np
import pytest

from reflectolearn.fit_policy import stack as st
from reflectolearn.fit_policy.baselines import inner_full_fit, truth_fit
from reflectolearn.fit_policy.env import FitEnv
from reflectolearn.fit_policy.protocol import A1mk, A1v, run

TRUTH = st.Stack(
    np.array([[120.0, 4.0, 40.0], [60.0, 3.0, 12.0]]), np.array([1.0, -7.5, 0.0, 0.002])
)


def _data():
    q = st.q_grid()
    held = st.held_out_mask(q)
    y = np.random.default_rng(11).poisson(st.counts(TRUTH, q, 1e6)).astype(float)
    return q[~held], y[~held], q[held], y[held], ~st.inner_mask(q)[~held]


def test_inner_block_is_registered():
    q = st.q_grid()
    inner = st.inner_mask(q)
    assert inner.sum() == 28
    assert q[inner].min() == pytest.approx(0.1964, abs=1e-4)
    assert q[inner].max() == pytest.approx(0.2348, abs=1e-4)
    assert not (inner & st.held_out_mask(q)).any()
    assert (~st.inner_mask(q)[~st.held_out_mask(q)]).sum() == 208


def test_seed_root_changes_every_stream_and_default_is_x2b():
    assert np.array_equal(st.sample_truth(2, 0).vector(), st.sample_truth(2, 0, st.SEED).vector())
    assert not np.array_equal(
        st.sample_truth(2, 0).vector(), st.sample_truth(2, 0, 20261025).vector()
    )
    assert not np.array_equal(st.starts(2, 0)[0].vector(), st.starts(2, 0, 20261025)[0].vector())


def test_kept_path_revert_never_reaches_a_rejected_fork():
    q, y, q_ho, y_ho, _ = _data()
    start = st.Stack(np.array([[150.0, 5.0, 35.0]]), st.INSTR_START.copy())
    env = FitEnv(q, y, 1e6, start, q_ho=q_ho, y_ho=y_ho, kept_path=True)
    fork = env.fork()
    for a in [("open_all", "d"), ("open_all", "rho"), ("open", "bkg"), ("loss", "poisson")]:
        fork.step(a)
    fork.step(("fit", 100))  # the fork's state is better than anything on env's path
    assert env.episode.best_bic < env.path_best_bic
    env.step(("open", "bkg"))
    if env.valid(("revert",)):
        env.step(("revert",))
    assert env.state.origin in env.lineage
    assert env.returned().origin in env.lineage
    assert env.result()["on_kept_path"]


def test_matched_arm_records_decisions_and_returns_on_its_path():
    q, y, q_ho, y_ho, _ = _data()
    env = FitEnv(q, y, 1e6, st.starts(9, 1)[0], q_ho=q_ho, y_ho=y_ho, budget=600)
    final = run(env, A1mk)
    res = final.result()
    assert res["on_kept_path"]
    assert res["evaluations"] <= 600
    for n_layers, unmatched, matched in final.episode.decisions:
        assert 1 <= n_layers <= st.L_MAX
        assert np.isfinite(unmatched) and np.isfinite(matched)


def test_a1v_fits_without_the_inner_block_until_step_9():
    q, y, q_ho, y_ho, fit_mask = _data()
    env = FitEnv(q, y, 1e6, st.starts(9, 2)[0], q_ho=q_ho, y_ho=y_ho, fit_mask=fit_mask)
    assert len(env.q) == 208 and np.isfinite(env.dev_inner)
    final = run(env, A1v)
    assert final.fit_mask is None  # step 9 rejoined the inner block
    res = final.result()
    assert res["on_kept_path"] and res["evaluations"] <= 3000


def test_cv_and_tf_stay_inside_the_budget():
    q, y, q_ho, y_ho, fit_mask = _data()
    cv = inner_full_fit(
        st.starts(9, 3)[0],
        st.extra_layers(9, 3)[0],
        q,
        y,
        1e6,
        fit_mask,
        q_ho=q_ho,
        y_ho=y_ho,
        budget=400,
    )
    assert cv["evaluations"] <= 400 and len(cv["inner_by_L"]) == st.L_MAX
    tf = truth_fit(TRUTH, q, y, 1e6, q_ho=q_ho, y_ho=y_ho, budget=200)
    assert tf["evaluations"] <= 200 and tf["n_layers"] == 2
