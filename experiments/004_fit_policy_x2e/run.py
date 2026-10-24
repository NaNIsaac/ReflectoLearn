"""Experiment 004, X2e: the registered run of docs/fit_policy/X2E.md.

Seed root 20261027 for every data stream (X2E.md section 3). Episodes are paired units (class n,
truth j, noise level s, start k), 3 × 16 × 2 × 4 = 384 per arm; A1o, D2 and D3 run once per truth
and noise level (k = 0). D1 runs inside every A1 episode. Each finished episode is one JSON line in
docs/fit_policy/x2e_runs/<arm>.jsonl; every line carries its commit, and episodes refuse to start
with uncommitted changes (freeze). An episode that raises goes to x2e_runs/errors.jsonl instead, is
not counted as done, and makes report.py refuse a verdict (amendment 2).

Run:  uv run python experiments/004_fit_policy_x2e/run.py noise
      uv run python experiments/004_fit_policy_x2e/run.py episodes --arms A1 S12 C --workers 4
      uv run python experiments/004_fit_policy_x2e/run.py timing
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from reflectolearn.fit_policy import stack as st
from reflectolearn.fit_policy.baselines import (
    de_fit,
    free_fit_deviance,
    full_fit,
    random_policy,
    step3_state,
    true_l_data_start,
)
from reflectolearn.fit_policy.env import BUDGET, FitEnv
from reflectolearn.fit_policy.protocol import X2E_ARMS
from reflectolearn.fit_policy.protocol import run as run_protocol

ROOT_DIR = Path(__file__).resolve().parents[2]
RUNS = ROOT_DIR / "docs" / "fit_policy" / "x2e_runs"
ERRORS = RUNS / "errors.jsonl"
SEED_ROOT = 20261027
CLASSES = (1, 2, 3)
N_TRUTHS = 16
N_NOISE_REALISATIONS = 20
ARMS = ("A1", "S1", "S2", "S12", "B", "C", "C2E", "A1o", "D2", "D3")
PER_TRUTH = ("A1o", "D2", "D3")
TIMING_CLASS = 9  # timing pilot: truths (n, 1000 + n), starts and random seeds of class 9


def units(arm: str, n_truths: int = N_TRUTHS) -> list[tuple[int, int, int, int]]:
    starts = (0,) if arm in PER_TRUTH else range(st.N_STARTS)
    return list(itertools.product(CLASSES, range(n_truths), range(len(st.NOISE_SCALES)), starts))


def data(n: int, j: int, s: int, root: int):
    truth = st.sample_truth(n, j, root)
    q = st.q_grid()
    held = st.held_out_mask(q)
    y = st.noisy_counts(st.counts(truth, q, st.NOISE_SCALES[s]), n, j, s, root=root)
    return truth, q[~held], y[~held], q[held], y[held]


def frozen_commit() -> str:
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", ".", ":(exclude)docs/fit_policy/x2e_runs"],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if dirty:
        raise SystemExit(f"uncommitted changes, not a frozen commit:\n{dirty}")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, capture_output=True, text=True, check=True
    ).stdout.strip()


def _kept_true_l_dev(res: dict, n: int) -> float | None:
    devs = [d for n_layers, d in res["kept_path_states"] if n_layers == n]
    return devs[-1] if devs else None


def episode(
    arm: str, n: int, j: int, s: int, k: int, root: int = SEED_ROOT, seed_class: int | None = None
) -> dict:
    """One episode. seed_class replaces n in the start and random seeds (timing pilot only)."""
    torch.set_num_threads(1)
    truth, q, y, q_ho, y_ho = data(n, j, s, root)
    sc = n if seed_class is None else seed_class
    start = st.starts(sc, j, root)[k]
    extra = st.extra_layers(sc, j, root)[k]
    scale = st.NOISE_SCALES[s]
    t0 = time.perf_counter()
    extra_fields: dict = {}
    if arm in X2E_ARMS:
        first = start
        if arm == "A1o":
            first = truth
        elif arm == "D3":
            first = true_l_data_start(step3_state(start, q, y, scale), n)
        env = FitEnv(q, y, scale, first, q_ho=q_ho, y_ho=y_ho)
        final = run_protocol(env, X2E_ARMS[arm])
        res = final.result()
        extra_fields = {
            "s1_changed": final.episode.s1_changed,
            "s2_changed": final.episode.s2_changed,
            "kept_true_l_dev": _kept_true_l_dev(res, n),
        }
        if arm == "A1":
            state = final.kept_state_at(n)
            if state is not None:
                d1, d1_nfev = free_fit_deviance(state, q, y, scale)
                extra_fields |= {"d1_dev": d1, "d1_evaluations": d1_nfev}
            else:
                extra_fields["d1_dev"] = None
    elif arm == "B":
        env = FitEnv(q, y, scale, start, q_ho=q_ho, y_ho=y_ho)
        res = random_policy(env, np.random.default_rng((root, sc, j, s, k, 7))).result()
    elif arm in ("C", "C2E"):
        budget = BUDGET * (2 if arm == "C2E" else 1)
        res = full_fit(start, extra, q, y, scale, q_ho=q_ho, y_ho=y_ho, budget=budget)
    elif arm == "D2":
        first = true_l_data_start(step3_state(start, q, y, scale), n)
        res = de_fit(first, q, y, scale, (root, sc, j, s, 0, 8), q_ho=q_ho, y_ho=y_ho)
    else:
        raise ValueError(arm)
    return (
        {
            "arm": arm,
            "n": n,
            "j": j,
            "s": s,
            "k": k,
            "root": root,
            "seconds": time.perf_counter() - t0,
            "exhausted": res.get("evaluations", 0) >= BUDGET,
        }
        | res
        | extra_fields
    )


def _episode(args) -> dict:
    """An episode that raises becomes an error record and the run goes on (amendment 2, rule 4)."""
    try:
        return episode(*args)
    except Exception as exc:  # noqa: BLE001 - the rule covers any exception an episode raises
        arm, n, j, s, k = args
        return {"arm": arm, "n": n, "j": j, "s": s, "k": k, "error": f"{type(exc).__name__}: {exc}"}


def _done(path: Path) -> set[tuple[int, int, int, int]]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {(r["n"], r["j"], r["s"], r["k"]) for r in map(json.loads, f)}


def run_episodes(arms: list[str], workers: int, classes: list[int]) -> None:
    commit = frozen_commit()
    print(f"commit {commit}")
    RUNS.mkdir(parents=True, exist_ok=True)
    jobs = []
    for arm in arms:
        done = _done(RUNS / f"{arm}.jsonl")
        jobs += [(arm, *u) for u in units(arm) if u[0] in classes and u not in done]
    print(f"{len(jobs)} episodes to run")
    errors = 0
    with Pool(workers) as pool:
        for i, rec in enumerate(pool.imap_unordered(_episode, jobs), 1):
            path = ERRORS if "error" in rec else RUNS / f"{rec['arm']}.jsonl"
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec | {"commit": commit}) + "\n")
            if "error" in rec:
                errors += 1
                print(
                    f"error {rec['arm']} n{rec['n']} j{rec['j']} s{rec['s']} k{rec['k']}: {rec['error']}",
                    flush=True,
                )
            elif i % 20 == 0 or i == len(jobs):
                print(
                    f"{i}/{len(jobs)} {rec['arm']} n{rec['n']} j{rec['j']} {rec['seconds']:.0f}s",
                    flush=True,
                )
    print(f"done, {errors} error records", flush=True)


def noise_scale() -> None:
    out = {}
    q = st.q_grid()
    held = st.held_out_mask(q)
    for n, j, s in itertools.product(CLASSES, range(N_TRUTHS), range(len(st.NOISE_SCALES))):
        lam = st.counts(st.sample_truth(n, j, SEED_ROOT), q, st.NOISE_SCALES[s])[held]
        devs = [
            st.deviance(lam, st.noisy_counts(lam, n, j, s, r, root=SEED_ROOT))
            for r in range(1, N_NOISE_REALISATIONS + 1)
        ]
        out[f"{n},{j},{s}"] = {"s_ho": float(np.std(devs, ddof=1)), "mean": float(np.mean(devs))}
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "noise.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"{len(out)} noise scales written")


def timing() -> None:
    """One episode per arm on unregistered truths (n, 1000 + n); time and evaluations only."""
    for n in CLASSES:
        for arm in ARMS:
            if arm == "C2E":
                continue
            rec = episode(arm, n, 1000 + n, 1, 0, seed_class=TIMING_CLASS)
            extra = f"  D1 evaluations {rec.get('d1_evaluations')}" if arm == "A1" else ""
            print(
                f"n{n} {arm:4s} {rec['seconds']:6.1f} s  evaluations {rec['evaluations']}{extra}",
                flush=True,
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("episodes")
    e.add_argument("--arms", nargs="+", choices=ARMS, required=True)
    e.add_argument("--workers", type=int, default=4)
    e.add_argument("--classes", type=int, nargs="+", default=list(CLASSES))
    sub.add_parser("noise")
    sub.add_parser("timing")
    args = ap.parse_args()
    match args.cmd:
        case "episodes":
            run_episodes(args.arms, args.workers, args.classes)
        case "noise":
            noise_scale()
        case "timing":
            timing()


if __name__ == "__main__":
    main()
