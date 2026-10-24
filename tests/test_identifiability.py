"""Checks of experiments/001_identifiability/fim.py against references that do not share its code.

The forward model is compared with benchmark.parratt (numpy Parratt recursion) and the autograd
Jacobian with central differences of a numpy re-implementation of the counts model. The Fisher
analysis is checked on its two exact properties: Poisson information is linear in the counting time,
and a parameter that cannot change the curve leaves the information matrix singular.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

FIM_PATH = Path(__file__).parents[1] / "experiments" / "001_identifiability" / "fim.py"


@pytest.fixture(scope="module")
def fim():
    spec = importlib.util.spec_from_file_location("fim_001", FIM_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["fim_001"] = module
    spec.loader.exec_module(module)
    yield module
    del sys.modules["fim_001"]


def test_forward_model_matches_numpy_parratt(fim):
    assert fim.check_forward_vs_parratt() < 1e-12


def test_autograd_jacobian_matches_finite_differences(fim):
    errors = fim.check_jacobian_vs_fd(tth_max=3.0)
    assert max(errors.values()) < 1e-6, errors


def test_crlb_scales_as_inverse_sqrt_counts(fim):
    q = fim.tth_to_q(fim.tth_grid(3.0))
    p0, free = fim.truth(fim.FILMS[0], "instr"), fim.SCENARIOS["instr"]
    low = fim.fisher(*fim.jacobian(q, p0, free, 1e5), free)
    high = fim.fisher(*fim.jacobian(q, p0, free, 1e7), free)
    np.testing.assert_allclose(low.crlb / high.crlb, 10.0, rtol=1e-9)
    np.testing.assert_allclose(low.corr, high.corr, atol=1e-9)


def test_a_parameter_without_effect_makes_the_information_singular(fim):
    # With no oxide in the stack the oxide SLD does not enter the curve: its column of J is zero.
    q = fim.tth_to_q(fim.tth_grid(3.0))
    p0 = fim.truth(fim.FILMS[0], "film")
    lam, J = fim.jacobian(q, p0, (*fim.FILM, "oxide_sld"), 1e6)
    assert np.all(J[:, 3] == 0)
    F = J.T @ (lam[:, None] * J)
    assert np.linalg.matrix_rank(F) == 3


FULL = {"kalpha2_ratio": 0.5, "footprint": 500.0}
FULL_FREE = (
    "thickness",
    "roughness",
    "sld",
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


def test_jacobian_with_kalpha2_and_footprint_matches_finite_differences(fim):
    q = fim.tth_to_q(fim.tth_grid(3.0))
    p0 = fim.truth(fim.FILMS[0], "oxide") | FULL
    _, J = fim.jacobian(q, p0, FULL_FREE, 1e6, absorption=0.0)
    J_fd = fim.jacobian_fd(q, p0, FULL_FREE, 1e6)
    err = np.linalg.norm(J - J_fd, axis=0) / np.linalg.norm(J_fd, axis=0)
    assert err.max() < 1e-6, dict(zip(FULL_FREE, err, strict=True))


def test_counts_model_matches_refine_forward_with_every_effect(fim):
    import torch

    from reflectolearn.refine.pipeline import RefineConfig, make_forward

    q = np.linspace(fim.tth_to_q(np.array(0.2)), fim.tth_to_q(np.array(6.0)), 291)
    p0 = fim.truth(fim.FILMS[0], "oxide") | FULL | {"tth_offset": 0.01}
    fwd = make_forward(
        torch.as_tensor(q, dtype=torch.float32),
        torch.tensor([[fim.SUB_SLD, fim.SUB_ROUGH]]),
        wavelength=fim.WAVELENGTH,
        config=RefineConfig(),
        device=torch.device("cpu"),
    )
    vals = {k: torch.tensor([v], dtype=torch.float32) for k, v in p0.items()}
    r_refine = fwd(vals, {"tth_offset", "dq", "oxide", "kalpha2", "footprint"})[0].double().numpy()
    p = {k: torch.tensor(v, dtype=torch.float64) for k, v in p0.items()}
    r_here = fim.expected_counts(torch.as_tensor(q), p, 1.0, fim.effects_of(p0)).numpy()
    # make_forward pads the grid edges before smearing; compare away from them
    diff = np.abs(np.log10(r_here / r_refine))[5:-5]
    assert diff.max() < 1e-4
