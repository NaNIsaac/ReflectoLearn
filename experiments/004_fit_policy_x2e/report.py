"""Experiment 004, X2e: apply docs/fit_policy/X2E.md sections 5–6 to the episodes in
docs/fit_policy/x2e_runs/ and write docs/fit_policy/x2e_results.md and x2e_verdict.json. The rules
are applied mechanically. Per-cell protocol tests are X2b's (DESIGN 6.2–6.5), and the overshoot
unit and sign test are X2d's (X2D section 5, amendment 1d), both imported unchanged.

Run:  uv run python experiments/004_fit_policy_x2e/report.py
"""

from __future__ import annotations

import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np

from reflectolearn.fit_policy import stack as st

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "docs" / "fit_policy"
RUNS = OUT / "x2e_runs"
SEED_ROOT = 20261027
MULTI = [(n, s) for n in (2, 3) for s in range(len(st.NOISE_SCALES))]
ARMS_FULL = ("A1", "S1", "S2", "S12", "B", "C")
ARMS_PER_TRUTH = ("A1o", "D2", "D3")
MARGIN = 10 + 3 * np.log(236)  # 26.4
EXPOSURE_MIN = 0.10
EXHAUST_MAX = 0.25


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


x2b = _load_module("x2b_report", HERE.parent / "002_fit_policy" / "report.py")
x2d = _load_module("x2d_report", HERE.parent / "003_fit_policy_x2d" / "report.py")


def load(arm: str) -> dict[tuple, dict]:
    path = RUNS / f"{arm}.jsonl"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {(r["n"], r["j"], r["s"], r["k"]): r for r in map(json.loads, f)}


def check(arms: dict) -> None:
    """Refuse error records, mixed commits, missing episodes, and a budget-limited cell without
    C(2E)."""
    errors = RUNS / "errors.jsonl"
    if errors.exists() and errors.read_text(encoding="utf-8").strip():
        raise SystemExit(f"{errors.name} has error records: no verdict (amendment 2)")
    commits = {r["commit"] for recs in arms.values() for r in recs.values()}
    if len(commits) != 1:
        raise SystemExit(f"records from more than one commit: {sorted(commits)}")
    for arm in ARMS_FULL:
        missing = x2d.all_keys() - arms[arm].keys()
        if missing:
            raise SystemExit(f"{arm}: {len(missing)} episodes missing")
    for arm in ARMS_PER_TRUTH:
        missing = x2d.all_keys(per_truth=True) - arms[arm].keys()
        if missing:
            raise SystemExit(f"{arm}: {len(missing)} episodes missing")
    for n, s in itertools.product((1, 2, 3), (0, 1)):
        keys = x2d.cell_keys(n, s)
        if x2b.capped_frac(arms["C"], keys) > x2b.CAP_FRAC and not set(keys) <= arms["C2E"].keys():
            raise SystemExit(f"cell n{n}_s{s} is budget-limited and C(2E) is missing")


# ------------------------------------------------------------------ section 5
def overshooting(arms: dict) -> list[tuple[int, int, int]]:
    """Multilayer truths where X2e's own A1 overshoots (median returned L > true L)."""
    return [(n, j, s) for n, s in MULTI for j in range(16) if x2d.median_l(arms["A1"], n, j, s) > n]


def _median(values) -> float | None:
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else None


def gap(arms: dict, arm: str, key: str, n: int, j: int, s: int) -> float | None:
    """Median over starts of `key` minus A1o's kept-path state at the true L (None if missing)."""
    ref = arms["A1o"][(n, j, s, 0)]["kept_true_l_dev"]
    value = _median(r[key] for k, r in arms[arm].items() if k[:3] == (n, j, s))
    return None if value is None or ref is None else value - ref


def within(gaps: list) -> tuple[float, int]:
    gaps = [g for g in gaps if g is not None]
    return (float(np.mean([g <= MARGIN for g in gaps])) if gaps else float("nan")), len(gaps)


def d1(arms: dict, over) -> dict:
    gaps = [gap(arms, "A1", "d1_dev", *t) for t in over]
    frac_within, counted = within(gaps)
    refuted = counted > 0 and frac_within >= 0.50
    supported = counted > 0 and 1 - frac_within >= 0.75
    return {
        "counted": counted,
        "frac_within_margin": frac_within,
        "basin_reading": "refuted" if refuted else "supported" if supported else "unresolved",
    }


def exposure(arms: dict, arm: str) -> float:
    recs = [r for k, r in arms[arm].items() if k[0] in (2, 3)]
    if arm == "S1":
        flags = [c for r in recs for c in r["s1_changed"]]
    else:
        flags = [bool(r["s2_changed"]) for r in recs]
    return float(np.mean(flags)) if flags else 0.0


def exhausted(arms: dict, arm: str) -> float:
    return float(np.mean([r["exhausted"] for k, r in arms[arm].items() if k[0] in (2, 3)]))


def grade(not_applicable: bool, untested: bool, confounded: bool, refuted: bool, supported: bool):
    """X2E.md section 5 grading order."""
    for label, flag in (
        ("not applicable", not_applicable),
        ("untested", untested),
        ("budget-confounded", confounded),
        ("refuted", refuted),
        ("supported", supported),
    ):
        if flag:
            return label
    return "unresolved"


def source_s(arms: dict, arm: str, over, basin: str) -> dict:
    tests = {f"n{n}_s{s}": x2d.overshoot_sign_test(arms[arm], arms["A1"], n, s) for n, s in MULTI}
    cells = sum(t["holds"] for t in tests.values())
    frac, counted = within([gap(arms, arm, "kept_true_l_dev", *t) for t in over])
    exp, exh = exposure(arms, arm), exhausted(arms, arm)
    refuted = cells == 0 and counted > 0 and frac <= 0.10
    supported = cells >= 2 and counted > 0 and frac >= 0.50
    return {
        "sign_tests": tests,
        "frac_gap_within_margin": frac,
        "counted": counted,
        "exposure": exp,
        "exhausted_frac": exh,
        "grade": grade(
            basin == "refuted", exp < EXPOSURE_MIN, exh > EXHAUST_MAX, refuted, supported
        ),
    }


def source_b3(arms: dict, over, basin: str) -> dict:
    deep = [t for t in over if (g := gap(arms, "A1", "d1_dev", *t)) is not None and g > MARGIN]
    gaps = []
    for n, j, s in deep:
        ref = arms["A1o"][(n, j, s, 0)]["kept_true_l_dev"]
        gaps.append(None if ref is None else arms["D2"][(n, j, s, 0)]["deviance"] - ref)
    frac, counted = within(gaps)
    return {
        "truths_where_D1_stays_out": len(deep),
        "frac_D2_within_margin": frac,
        "grade": grade(
            basin == "refuted",
            False,
            False,
            counted > 0 and frac <= 0.10,
            counted > 0 and frac >= 0.50,
        ),
    }


def d3_reading(arms: dict) -> dict:
    e = {f"n{n}_s{s}": x2d.e_count(arms["D3"], n, s) for n, s in MULTI}
    if all(v <= 1 for v in e.values()):
        reading = "points to the one-layer path"
    elif sum(v >= 4 for v in e.values()) >= 2:
        reading = "points to the start values at a fixed L"
    else:
        reading = "unclear"
    return {"E_D3": e, "reading": reading}


# ------------------------------------------------------------------ section 6
def attributable_cells(arms: dict, arm: str, noise: dict) -> dict:
    out = {}
    for n, s in MULTI:
        keys = x2d.cell_keys(n, s)
        ref = x2d.reference(arms, n, s)[1]
        b = x2b.beats(arms[arm], ref, keys, noise)
        guard = x2d.overshoot_sign_test(arms[arm], arms["A1"], n, s)
        out[f"n{n}_s{s}"] = {
            "vs_C": b,
            "beats_C": b["beats"],
            "overshoot_vs_A1": guard,
            "attributable": b["beats"] and guard["holds"],
        }
    return out


def main() -> None:
    noise = json.loads((RUNS / "noise.json").read_text(encoding="utf-8"))
    arms = {a: load(a) for a in (*ARMS_FULL, "C2E", *ARMS_PER_TRUTH)}
    check(arms)
    commit = next(iter(arms["A1"].values()))["commit"]
    truths = {(n, j): st.sample_truth(n, j, SEED_ROOT) for n in (1, 2, 3) for j in range(16)}

    over = overshooting(arms)
    diag1 = d1(arms, over)
    sources = {
        "B1": source_s(arms, "S1", over, diag1["basin_reading"]),
        "B2": source_s(arms, "S2", over, diag1["basin_reading"]),
        "B3": source_b3(arms, over, diag1["basin_reading"]),
    }
    grades = {b: v["grade"] for b, v in sources.items()}
    causes = [
        b
        for b, g in grades.items()
        if g == "supported" and all(g2 == "refuted" for b2, g2 in grades.items() if b2 != b)
    ]

    primary = attributable_cells(arms, "S12", noise)
    secondary = {a: attributable_cells(arms, a, noise) for a in ("S1", "S2")}
    n_beats = sum(c["beats_C"] for c in primary.values())
    n_attr = sum(c["attributable"] for c in primary.values())
    passing_cells = [x for x, c in primary.items() if c["beats_C"]]
    reward = {
        name: x2b.mcnemar_less(
            arms["S12"], x2d.reference(arms, n, s)[1], x2d.cell_keys(n, s), truths
        )
        for (n, s), name in zip(MULTI, primary, strict=True)
        if name in passing_cells
    }
    misaligned = any(v["misaligned"] for v in reward.values())
    random_floor = sum(
        x2b.beats(arms["S12"], arms["B"], x2d.cell_keys(n, s), noise)["beats"] for n, s in MULTI
    )
    secondary_pass = [
        a for a, cells in secondary.items() if sum(c["attributable"] for c in cells.values()) >= 2
    ]
    if n_attr >= 2 and not misaligned:
        verdict = "passes, attributable: confirmation run on root 20261028; X2c stays unregistered"
    elif n_attr >= 2:
        verdict = "passes, attributable, but the reward is misaligned: no confirmation run"
    elif n_beats >= 2:
        verdict = "passes, not attributable to the start change: no confirmation run"
    elif secondary_pass:
        verdict = "inconclusive, confirm with new seeds"
    else:
        verdict = "fail"

    third_sample = {
        f"n{n}_s{s}": x2b.beats(
            arms["A1"], x2d.reference(arms, n, s)[1], x2d.cell_keys(n, s), noise
        )
        for n, s in MULTI
    }
    summary = {
        "commit": commit,
        "basin_reading_D1": diag1["basin_reading"],
        "sources": grades,
        "cause": causes[0] if len(causes) == 1 else None,
        "D3": d3_reading(arms)["reading"],
        "S12_beats_C_cells": n_beats,
        "S12_attributable_cells": n_attr,
        "random_floor_cells": random_floor,
        "verdict": verdict,
        "A1_vs_C_cells_beaten (third X2b sample)": sum(b["beats"] for b in third_sample.values()),
    }
    (OUT / "x2e_verdict.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "D1": diag1,
                "sources": sources,
                "D3": d3_reading(arms),
                "primary": primary,
                "secondary": secondary,
                "reward_alignment": reward,
                "A1_vs_C": third_sample,
            },
            indent=1,
            default=float,
        ),
        encoding="utf-8",
    )
    write_markdown(summary, diag1, sources, primary, secondary, third_sample)
    print(json.dumps(summary, indent=1))


def write_markdown(summary, diag1, sources, primary, secondary, third_sample) -> None:
    lines = [
        "# Experiment 004 (X2e) results (generated by experiments/004_fit_policy_x2e/report.py)",
        "",
        "Rules: docs/fit_policy/X2E.md sections 5–6. Generated; do not edit by hand.",
        "",
        "## Summary",
        "",
        *(f"- {k}: {v}" for k, v in summary.items()),
        "",
        "## D1 and sources",
        "",
        f"- D1: `{json.dumps(diag1, default=float)}`",
        *(f"- {b}: `{json.dumps(v, default=float)}`" for b, v in sources.items()),
        "",
        "## Protocols against C, with the attribution guard (overshoot against A1)",
        "",
        (
            "| arm | cell | beats C | truth BIC win / loss | held-out loss "
            "| overshoot for / against A1, p | attributable |"
        ),
        "|---|---|---|---|---:|---|---|",
    ]
    for arm, cells in {"S12": primary, **secondary}.items():
        for name, c in cells.items():
            b, g = c["vs_C"], c["overshoot_vs_A1"]
            lines.append(
                f"| {arm} | {name} | {c['beats_C']} | {b['bic_wins']} / {b['bic_losses']} | "
                f"{b['ho_losses']} | {g['for']} / {g['against']}, {g['p']:.2g} | "
                f"{c['attributable']} |"
            )
    lines += ["", "## A1 against C (third sample of the X2b question)", ""]
    lines += ["| cell | truth BIC win / loss | held-out loss | beats |", "|---|---|---:|---|"]
    for name, b in third_sample.items():
        lines.append(
            f"| {name} | {b['bic_wins']} / {b['bic_losses']} | {b['ho_losses']} | {b['beats']} |"
        )
    (OUT / "x2e_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
