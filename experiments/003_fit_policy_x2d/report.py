"""Experiment 003, X2d: apply docs/fit_policy/X2D.md sections 5–7 (with amendment 1) to the episodes
in docs/fit_policy/x2d_runs/ and write docs/fit_policy/x2d_results.md and x2d_verdict.json. The rules
are applied mechanically. The per-cell protocol tests are X2b's own functions (DESIGN 6.2–6.5), taken
from experiments/002_fit_policy/report.py unchanged.

Run:  uv run python experiments/003_fit_policy_x2d/report.py
"""

from __future__ import annotations

import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

from reflectolearn.fit_policy import stack as st

HERE = Path(__file__).resolve().parent
ROOT_DIR = HERE.parents[1]
OUT = ROOT_DIR / "docs" / "fit_policy"
RUNS = OUT / "x2d_runs"
SEED_ROOT = 20261025
MULTI = [(n, s) for n in (2, 3) for s in range(len(st.NOISE_SCALES))]
ARMS_FULL = ("A1", "A1m", "A1k", "A1mk", "A1v", "B", "C", "Cv")
ARMS_PER_TRUTH = ("A1o", "TF")
BIC_MARGIN = 10 + 3 * np.log(236)  # 26.4, H2 (i)
SIGN_P = 0.01

_spec = importlib.util.spec_from_file_location(
    "x2b_report", HERE.parent / "002_fit_policy" / "report.py"
)
x2b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(x2b)


def load(arm: str) -> dict[tuple, dict]:
    path = RUNS / f"{arm}.jsonl"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {(r["n"], r["j"], r["s"], r["k"]): r for r in map(json.loads, f)}


def cell_keys(n: int, s: int, per_truth: bool = False) -> list[tuple]:
    starts = (0,) if per_truth else range(st.N_STARTS)
    return [(n, j, s, k) for j in range(16) for k in starts]


def all_keys(per_truth: bool = False) -> set[tuple]:
    return {k for n, s in itertools.product((1, 2, 3), (0, 1)) for k in cell_keys(n, s, per_truth)}


# ------------------------------------------------------------------ section 3 checks
def check(arms: dict[str, dict]) -> None:
    """Refuse a mixed commit, a missing episode, or a budget-limited cell without C(2E)."""
    commits = {r["commit"] for recs in arms.values() for r in recs.values()}
    if len(commits) != 1:
        raise SystemExit(f"records from more than one commit: {sorted(commits)}")
    for arm in ARMS_FULL:
        missing = all_keys() - arms[arm].keys()
        if missing:
            raise SystemExit(f"{arm}: {len(missing)} episodes missing")
    for arm in ARMS_PER_TRUTH:
        missing = all_keys(per_truth=True) - arms[arm].keys()
        if missing:
            raise SystemExit(f"{arm}: {len(missing)} episodes missing")
    for n, s in itertools.product((1, 2, 3), (0, 1)):
        keys = cell_keys(n, s)
        if x2b.capped_frac(arms["C"], keys) > x2b.CAP_FRAC and not set(keys) <= arms["C2E"].keys():
            raise SystemExit(f"cell n{n}_s{s} is budget-limited and C(2E) is missing")


def reference(arms: dict, n: int, s: int) -> tuple[str, dict]:
    keys = cell_keys(n, s)
    name = "C2E" if x2b.capped_frac(arms["C"], keys) > x2b.CAP_FRAC else "C"
    return name, arms[name]


# ------------------------------------------------------------------ section 5 unit
def median_l(recs: dict, n: int, j: int, s: int) -> float:
    return float(np.median([r["n_layers"] for k, r in recs.items() if k[:3] == (n, j, s)]))


def overshoot(recs: dict, n: int, j: int, s: int) -> float:
    return max(median_l(recs, n, j, s) - n, 0.0)


def e_count(recs: dict, n: int, s: int) -> int:
    """E_X: truths whose median returned L exceeds the true L."""
    return sum(median_l(recs, n, j, s) > n for j in range(16))


def overshoot_sign_test(arm: dict, a1: dict, n: int, s: int) -> dict:
    """Amendment 1d: per truth, arm's overshoot below A1's is for, above is against."""
    diffs = [overshoot(arm, n, j, s) - overshoot(a1, n, j, s) for j in range(16)]
    wins, losses = sum(d < 0 for d in diffs), sum(d > 0 for d in diffs)
    p = binomtest(wins, wins + losses, 0.5, alternative="greater").pvalue if wins + losses else 1.0
    return {"for": wins, "against": losses, "p": float(p), "holds": bool(p < SIGN_P)}


def grade(refuted: bool, supported: bool) -> str:
    """Amendment 1g: refutation first, then support, otherwise unresolved."""
    return "refuted" if refuted else "supported" if supported else "unresolved"


# ------------------------------------------------------------------ section 6
def replication(arms: dict, noise: dict) -> dict:
    cells = {}
    for n, s in MULTI:
        _, ref = reference(arms, n, s)
        b = x2b.beats(arms["A1"], ref, cell_keys(n, s), noise)
        cells[f"n{n}_s{s}"] = {
            "rule3_fails": b["ho_losses"] / b["truths"] > x2b.HO_LOSS_FRAC,
            "E_A1": e_count(arms["A1"], n, s),
            "A1_vs_C": b,
        }
    failing = [c for c in cells.values() if c["rule3_fails"]]
    if len(failing) <= 1:
        status = "not replicated"
    elif len(failing) >= 3 and all(c["E_A1"] >= 4 for c in failing):
        status = "replicated"
    else:
        status = "partial"  # two failing cells, or amendment 1c
    return {"status": status, "cells": cells}


def h1(arms: dict) -> dict:
    decisions = [
        d for k, r in arms["A1m"].items() if k[0] in (2, 3) for d in r.get("decisions", [])
    ]
    would_accept = [d for d in decisions if d[1] <= -10]
    frac = float(np.mean([d[2] > -10 for d in would_accept])) if would_accept else float("nan")
    tests = {f"n{n}_s{s}": overshoot_sign_test(arms["A1m"], arms["A1"], n, s) for n, s in MULTI}
    cells = sum(t["holds"] for t in tests.values())
    refuted = (would_accept and frac < 0.10) or cells == 0
    supported = bool(would_accept) and frac >= 0.50 and cells >= 2
    return {
        "decisions_unmatched_accept": len(would_accept),
        "matched_reject_frac": frac,
        "sign_tests": tests,
        "grade": grade(bool(refuted), supported),
    }


def _kept_at_true_l(rec: dict, n: int) -> float | None:
    devs = [d for n_layers, d in rec["kept_path_states"] if n_layers == n]
    return devs[-1] if devs else None


def h2(arms: dict) -> dict:
    a1, a1o = arms["A1"], arms["A1o"]
    over, counted, no_state = 0, 0, 0
    for n, s in MULTI:
        for j in range(16):
            if median_l(a1, n, j, s) <= n:
                continue
            a1_devs = [
                v
                for k in range(st.N_STARTS)
                if (v := _kept_at_true_l(a1[(n, j, s, k)], n)) is not None
            ]
            o = _kept_at_true_l(a1o[(n, j, s, 0)], n)
            if not a1_devs or o is None:
                no_state += 1  # amendment 1f
                continue
            counted += 1
            over += float(np.median(a1_devs)) - o > BIC_MARGIN
    frac = over / counted if counted else float("nan")
    e_o = {f"n{n}_s{s}": e_count(a1o, n, s) for n, s in MULTI}
    refuted = sum(e >= 4 for e in e_o.values()) >= 2 or (counted > 0 and frac <= 0.25)
    supported = counted > 0 and frac >= 0.75 and all(e <= 1 for e in e_o.values())
    return {
        "overshooting_truths_counted": counted,
        "left_out_no_true_l_state": no_state,
        "frac_above_margin": frac,
        "E_A1o": e_o,
        "grade": grade(refuted, supported),
    }


def h3(arms: dict, noise: dict) -> dict:
    tf, c = arms["TF"], arms["C"]
    losses, tf_above = {}, {}
    for n, s in MULTI:
        lost, above = 0, 0
        for j in range(16):
            nz = noise[f"{n},{j},{s}"]
            d_tf = tf[(n, j, s, 0)]["deviance_ho"]
            d_c = np.median([c[(n, j, s, k)]["deviance_ho"] for k in range(st.N_STARTS)])
            lost += d_tf - d_c >= x2b.HO_BAND * nz["s_ho"]
            above += d_tf > nz["mean"] + 2 * nz["s_ho"]
        losses[f"n{n}_s{s}"], tf_above[f"n{n}_s{s}"] = lost, above
    supported = sum(v >= 2 for v in losses.values()) >= 2
    refuted = all(v <= 1 for v in losses.values())
    return {
        "TF_heldout_losses": losses,
        "TF_above_noise_mean_2s": tf_above,
        "grade": grade(refuted, supported),
    }


def h4(arms: dict) -> dict:
    over = [r for k, r in arms["A1"].items() if k[0] in (2, 3) and r["n_layers"] > k[0]]
    frac = float(np.mean([not r["on_kept_path"] for r in over])) if over else float("nan")
    tests = {f"n{n}_s{s}": overshoot_sign_test(arms["A1k"], arms["A1"], n, s) for n, s in MULTI}
    cells = sum(t["holds"] for t in tests.values())
    refuted = bool(over) and frac <= 0.10
    supported = bool(over) and frac >= 0.50 and cells >= 2
    return {
        "A1_overshooting_episodes": len(over),
        "off_kept_path_frac": frac,
        "sign_tests": tests,
        "grade": grade(refuted, supported),
    }


# ------------------------------------------------------------------ section 7
def protocol_cells(arms: dict, arm: str, noise: dict, against: tuple[str, ...] = ("C",)) -> dict:
    out = {}
    for n, s in MULTI:
        keys = cell_keys(n, s)
        row = {}
        for ref_name in against:
            ref = reference(arms, n, s)[1] if ref_name == "C" else arms[ref_name]
            row[ref_name] = x2b.beats(arms[arm], ref, keys, noise)
        row["beats"] = all(row[r]["beats"] for r in against)
        out[f"n{n}_s{s}"] = row
    return out


def count_beats(cells: dict) -> int:
    return sum(c["beats"] for c in cells.values())


def main() -> None:
    noise = json.loads((RUNS / "noise.json").read_text(encoding="utf-8"))
    arms = {a: load(a) for a in (*ARMS_FULL, "C2E", *ARMS_PER_TRUTH)}
    check(arms)
    commit = next(iter(arms["A1"].values()))["commit"]

    rep = replication(arms, noise)
    if rep["status"] == "not replicated":
        hyps = {h: {"grade": "untested"} for h in ("H1", "H2", "H3", "H4")}
    else:
        hyps = {"H1": h1(arms), "H2": h2(arms), "H3": h3(arms, noise), "H4": h4(arms)}
    grades = {h: v["grade"] for h, v in hyps.items()}
    causes = [
        h
        for h, g in grades.items()
        if g == "supported" and all(g2 == "refuted" for h2_, g2 in grades.items() if h2_ != h)
    ]

    primary = protocol_cells(arms, "A1mk", noise)
    secondary = {
        "A1v": protocol_cells(arms, "A1v", noise, ("C", "Cv")),
        "A1m": protocol_cells(arms, "A1m", noise),
        "A1k": protocol_cells(arms, "A1k", noise),
    }
    n_primary = count_beats(primary)
    guards = {
        "random_floor_cells": count_beats(protocol_cells(arms, "A1mk", noise, ("B",))),
        "reward_alignment": {
            name: x2b.mcnemar_less(
                arms["A1mk"],
                reference(arms, n, s)[1],
                cell_keys(n, s),
                {(n, j): st.sample_truth(n, j, SEED_ROOT) for j in range(16)},
            )
            for (n, s), name in zip(MULTI, primary, strict=True)
            if primary[name]["beats"]
        },
    }
    misaligned = any(v["misaligned"] for v in guards["reward_alignment"].values())
    secondary_pass = [a for a, cells in secondary.items() if count_beats(cells) >= 2]
    if n_primary >= 2 and not misaligned:
        verdict = "primary passes: confirmation run on seed root 20261026; X2c stays unregistered"
    elif n_primary >= 2:
        verdict = "primary passes but the reward is misaligned: no confirmation run"
    elif secondary_pass or n_primary == 1:
        verdict = "inconclusive, confirm with new seeds"
    else:
        verdict = "no protocol change supported"

    summary = {
        "commit": commit,
        "replication": rep["status"],
        "hypotheses": grades,
        "cause": causes[0] if len(causes) == 1 else None,
        "primary_A1mk_cells": n_primary,
        "secondary_passing": secondary_pass,
        "verdict": verdict,
    }
    (OUT / "x2d_verdict.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "replication": rep,
                "hypotheses": hyps,
                "primary": primary,
                "secondary": secondary,
                "guards": guards,
            },
            indent=1,
            default=float,
        ),
        encoding="utf-8",
    )
    write_markdown(summary, rep, hyps, primary, secondary)
    print(json.dumps(summary, indent=1))


def write_markdown(summary, rep, hyps, primary, secondary) -> None:
    lines = [
        "# Experiment 003 (X2d) results (generated by experiments/003_fit_policy_x2d/report.py)",
        "",
        "Rules: docs/fit_policy/X2D.md sections 5–7 with amendment 1. Generated; do not edit by hand.",
        "",
        "## Summary",
        "",
        *(f"- {k}: {v}" for k, v in summary.items()),
        "",
        "## Replication (A1 against C)",
        "",
        "| cell | rule 3 fails | E_A1 | truth BIC win / tie / loss | held-out loss |",
        "|---|---|---:|---|---:|",
    ]
    for name, c in rep["cells"].items():
        b = c["A1_vs_C"]
        lines.append(
            f"| {name} | {c['rule3_fails']} | {c['E_A1']} | "
            f"{b['bic_wins']} / {b['bic_ties']} / {b['bic_losses']} | {b['ho_losses']} |"
        )
    lines += ["", "## Hypotheses", ""]
    for h, v in hyps.items():
        lines.append(f"- {h}: `{json.dumps(v, default=float)}`")
    lines += ["", "## Protocols against C (and Cv for A1v)", ""]
    lines += [
        "| arm | cell | beats | truth BIC win / loss vs C | held-out loss vs C |",
        "|---|---|---|---|---:|",
    ]
    for arm, cells in {"A1mk": primary, **secondary}.items():
        for name, c in cells.items():
            b = c["C"]
            lines.append(
                f"| {arm} | {name} | {c['beats']} | {b['bic_wins']} / {b['bic_losses']} | "
                f"{b['ho_losses']} |"
            )
    (OUT / "x2d_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
