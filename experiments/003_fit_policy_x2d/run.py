"""Experiment 003, X2d: the registered run of docs/fit_policy/X2D.md.

Seed root 20261025 for every data stream (X2D.md section 3). Episodes are paired units (class n,
truth j, noise level s, start k), 3 × 16 × 2 × 4 = 384 per arm; the diagnostics A1o and TF run once
per truth and noise level (k = 0). Each finished episode is one JSON line in
docs/fit_policy/x2d_runs/<arm>.jsonl, so a run can be stopped and resumed; every line carries the
commit it ran at, and episodes refuse to start with uncommitted changes (freeze, section 3).

Run:  uv run python experiments/003_fit_policy_x2d/run.py noise
      uv run python experiments/003_fit_policy_x2d/run.py episodes --arms A1 A1mk C --workers 4
      uv run python experiments/003_fit_policy_x2d/run.py timing
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
    full_fit,
    inner_full_fit,
    random_policy,
    truth_fit,
)
from reflectolearn.fit_policy.env import BUDGET, FitEnv
from reflectolearn.fit_policy.protocol import X2D_ARMS
from reflectolearn.fit_policy.protocol import run as run_protocol

ROOT_DIR = Path(__file__).resolve().parents[2]
RUNS = ROOT_DIR / "docs" / "fit_policy" / "x2d_runs"
SEED_ROOT = 20261025
CLASSES = (1, 2, 3)
N_TRUTHS = 16
N_NOISE_REALISATIONS = 20
PROTOCOL_ARMS = ("A1", "A1m", "A1k", "A1mk", "A1v")
ARMS = (*PROTOCOL_ARMS, "B", "C", "Cv", "C2E", "A1o", "TF")
PER_TRUTH = ("A1o", "TF")  # one episode per truth and noise level
TIMING_CLASS = 9  # timing pilot: truths (n, 1000 + n), starts and random seeds of class 9


def units(arm: str, n_truths: int = N_TRUTHS) -> list[tuple[int, int, int, int]]:
    starts = (0,) if arm in PER_TRUTH else range(st.N_STARTS)
    return list(itertools.product(CLASSES, range(n_truths), range(len(st.NOISE_SCALES)), starts))


def data(n: int, j: int, s: int, root: int):
    """The truth, and training and held-out (q, y) of truth (n, j) at noise level s."""
    truth = st.sample_truth(n, j, root)
    q = st.q_grid()
    held = st.held_out_mask(q)
    y = st.noisy_counts(st.counts(truth, q, st.NOISE_SCALES[s]), n, j, s, root=root)
    return truth, q[~held], y[~held], q[held], y[held], ~st.inner_mask(q)[~held]


def frozen_commit() -> str:
    """The commit the registered episodes run at. Refuses to run with uncommitted changes outside
    the run records."""
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", ".", ":(exclude)docs/fit_policy/x2d_runs"],
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


def episode(
    arm: str, n: int, j: int, s: int, k: int, root: int = SEED_ROOT, seed_class: int | None = None
) -> dict:
    """One episode. seed_class replaces n in the start and random seeds (timing pilot only)."""
    torch.set_num_threads(1)
    truth, q, y, q_ho, y_ho, fit_mask = data(n, j, s, root)
    sc = n if seed_class is None else seed_class
    start = st.starts(sc, j, root)[k]
    extra = st.extra_layers(sc, j, root)[k]
    scale = st.NOISE_SCALES[s]
    t0 = time.perf_counter()
    extra_fields: dict = {}
    if arm in X2D_ARMS:
        protocol = X2D_ARMS[arm]
        env = FitEnv(
            q,
            y,
            scale,
            truth if arm == "A1o" else start,
            q_ho=q_ho,
            y_ho=y_ho,
            fit_mask=fit_mask if protocol.inner else None,
        )
        final = run_protocol(env, protocol)
        res = final.result()
        extra_fields["decisions"] = final.episode.decisions
    elif arm == "B":
        env = FitEnv(q, y, scale, start, q_ho=q_ho, y_ho=y_ho)
        res = random_policy(env, np.random.default_rng((root, sc, j, s, k, 7))).result()
    elif arm in ("C", "C2E"):
        budget = BUDGET * (2 if arm == "C2E" else 1)
        res = full_fit(start, extra, q, y, scale, q_ho=q_ho, y_ho=y_ho, budget=budget)
    elif arm == "Cv":
        res = inner_full_fit(start, extra, q, y, scale, fit_mask, q_ho=q_ho, y_ho=y_ho)
    elif arm == "TF":
        res = truth_fit(truth, q, y, scale, q_ho=q_ho, y_ho=y_ho)
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
        }
        | res
        | extra_fields
    )


def _episode(args) -> dict:
    return episode(*args)


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
    with Pool(workers) as pool:
        for i, rec in enumerate(pool.imap_unordered(_episode, jobs), 1):
            with (RUNS / f"{rec['arm']}.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec | {"commit": commit}) + "\n")
            if i % 20 == 0 or i == len(jobs):
                print(
                    f"{i}/{len(jobs)} {rec['arm']} n{rec['n']} j{rec['j']} {rec['seconds']:.0f}s",
                    flush=True,
                )


def noise_scale() -> None:
    """s_ho per (n, j, s): SD of the truth model's held-out deviance over 20 extra realisations."""
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
            print(
                f"n{n} {arm:4s} {rec['seconds']:6.1f} s  evaluations {rec['evaluations']}",
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
