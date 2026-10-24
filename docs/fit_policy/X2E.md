# X2e: does a better start basin remove A1's extra layers? (registration)

Registered 2026-10-24, before any X2e code or run. It replaces `X2E_DRAFT.md`. The physics
conditions on ReflectoLearn#41 (issue comment 6378899745, items 1–10) are part of this registration
and govern where they differ from this text. DECISIONS #26: the X2d confirmation run on root
20261026 is not run, and X2e is registered instead. Section numbers without a prefix refer to
`DESIGN.md` (frozen at 887c4da); "X2d" refers to `X2D.md` with its amendment 1.

## 1. What is known, and from where

All of it is stated in #39. The X2b and X2d records are not analysed further (condition 2).

- [registered, X2d] H2 (a wrong basin compensated by layers) is supported, and is the cause under
  X2d's rules. In 42 of 42 overshooting truths, A1's kept-path state at the true L is more than 26.4
  above A1o's (the gap is at least 1777). A1o never overshoots.
- [registered, X2d] H1, H3 and H4 are refuted. H4's window never opened (0 of 168 overshooting
  episodes returned off the kept path).
- [registered, X2d] The X2d primary A1mk **passed, but the pass cannot be attributed to the
  protocol changes, and it was not confirmed**. A1, A1m, A1k and A1mk had the same truth-level
  outcomes against C. The confirmation run was dropped (DECISIONS #26).
- [registered, X2d] X2b's held-out failure replicated in 2 of 4 multilayer cells. X2e's A1 against C
  is a third sample of the X2b question. It is reported as such, and it does not revise X2b or X2d.

## 2. Sources of the wrong basin

- **B1, the new layer's initial values** (`add`: dominant residual period, neighbours' mean SLD,
  σ = 3 Å).
- **B2, the one-layer path.** Every later fit starts from a one-layer compromise.
- **B3, the reach of local fits.** At a fixed L, local fits may not leave the basin they start in.

All three presuppose that the X2d gap is a basin and not slow convergence. Diagnostic D1 (section 5)
tests that first.

## 3. Data, seeds, freeze (conditions 1–3)

Carried over unchanged: classes (1, 2, 3 layers), M = 16, noise levels S = 10⁸ and 10⁶, grid,
held-out blocks 2 and 6, bounds, E = 3000, L_max = 4, every threshold of DESIGN 6.2–6.5, C as frozen,
the X2d unit (X2D section 5), and X2d amendments 1d (overshoot sign test) and 1f (truths without a
true-L state).

- **Seeds.** Root 20261027 in every data stream: truths `(20261027, n, j)`, fitted noise
  `(20261027, n, j, s, 0)`, noise-scale realisations `(20261027, n, j, s, r)`, starts
  `(20261027, n, j, 2)`, C's extra layers `(20261027, n, j, 1)`, B `(20261027, n, j, s, k, 7)`, D2
  `(20261027, n, j, s, 0, 8)`. The step-7 Monte Carlo seed `(20261024, 99, 0)` stays. The
  confirmation root is named now: **20261028**.
- **Pilots** only on unregistered seeds (class 9 or truth 1000 + n), reading only time, evaluations
  and memory.
- **Freeze** at the commit of the first registered episode; every record carries it. The report
  refuses mixed commits and missing episodes. No optional stopping.

## 4. Arms

All arms on the same truths, noise and starts. Each S arm is A1 with the named part replaced; the
steps, losses, the −10 margin, the step-7 trigger and the returned-model rule (DESIGN 4.4) are A1's.
Every evaluation, screening fits included, is charged to E.

### 4.1 S1, several starts for a new layer (B1)

For each `add(p)` candidate of step 7, the new layer gets up to 9 starts instead of one. Each start
takes one of the three strongest peaks of the residual's real-space transform (the procedure of
`stack.fringe_spectrum` with the peak rule of 4.2, z in [20, 400] Å; fewer if fewer peaks), and ρ at
1.0, 0.75 or 1.25 times A1's neighbour mean, with σ = 3 Å. **Screening:** each start gets the new
layer's parameters open, `loss(log)` and `fit(20)`. The start with the lowest BIC is refit as A1
refits a candidate. If no peak exists, A1's single start (50 Å) is used. `split` candidates are
unchanged.

Exposure: the fraction of `add` decisions in which the chosen start differs from A1's start (the
strongest peak, mean ρ).

### 4.2 S2, stack starts from the fringe peaks (B2)

After A1's steps 1–3, before step 4:

- **Transform.** Training points only (the outer held-out blocks are never used). Take the
  log10-counts with the floor 0.5, at q > 1.1 q_c of the data. Fit and subtract a quintic in q,
  apply a Hann window over the point index, and take the transform |Σ v_i exp(−i z q_i)| at the
  points' own q, over z = 20, 21, …, 800 Å. The held-out gaps are not filled; the transform runs
  over the training q only. This is `stack.fringe_spectrum`'s procedure.
- **Peak rule.** Local maxima with amplitude ≥ 0.25 of the largest one. Taken in decreasing
  amplitude, a peak is kept if it lies at least Δz = 2π / (q_max − q_min) from every peak already
  kept, with q_max and q_min the ends of the transformed range. At most k = 4 peaks.
- **Peaks to stacks.** For m = 1, …, number of kept peaks (≤ 4 = L_max): take the m strongest kept
  peaks and sort them by z: z₁ < … < z_m. Read them as interface depths below the surface, which
  gives layer thicknesses (z₁, z₂ − z₁, …, z_m − z_{m−1}); the same thicknesses in reverse order are
  the second reading. L is therefore derived from the peaks, never enumerated from the classes. This
  gives at most 1 + 2 · 3 = 7 stacks, plus A1's own one-layer state after step 3, so 8 candidates in
  all. A stack with a thickness below 5 Å is dropped.
- **Start values.** SLD never comes from peak amplitudes. The top layer's ρ follows A1 step 2 (the
  critical edge; the step-3 state's ρ if the edge rule does not act). Each deeper layer i gets
  ρ_i = (ρ_{i−1} + ρ_Si) / 2. σ is the step-3 state's for the top interface and 3 Å for the others.
  The instrument values are the step-3 state's.
- **Screening and choice.** Each candidate gets d and ρ of every layer open, `loss(log)` and
  `fit(20)`. The lowest BIC is kept, and A1 continues with step 4 from it.

Exposure: the fraction of episodes whose chosen start is not A1's one-layer state.

### 4.3 S12, primary

S2's start, then S1's layer starts. **Named primary before the run** (condition 4). S1 and S2 are
its single-change arms.

### 4.4 Controls and diagnostics

- **A1**, unchanged: replication control, and the reference of the attribution guard.
- **C**, as frozen, and **C(2E)** where the budget guard requires it. **B**, as X2b.
- **A1o.** A1 from the truth with steps 2–3 skipped, one episode per truth and noise level (as X2d).
- **D1, basin or slow convergence** (condition 7). In every A1 episode, record the full parameters of
  the last kept-path state with L = true L. From it, run one fit with every parameter free, the
  Poisson loss and `max_nfev` = E. This is outside A1's budget and never changes A1's result.
- **True-L data start** (for D2 and D3). Uses no truth values except L. Take A1's step-3 one-layer
  state and split its d into L equal layers. The top ρ comes from the edge rule, each deeper
  ρ_i = (ρ_{i−1} + ρ_Si) / 2, σ is the step-3 state's on top and 3 Å below, and the instrument values
  are the step-3 state's.
- **D2, reach at the true L** (condition 8). `scipy.optimize.differential_evolution` over every
  parameter of the true-L stack within the fit bounds. Objective: the Poisson deviance on the training
  points. Settings: popsize 15, maxiter 50, `init="latinhypercube"`, the true-L data start as `x0`,
  `polish=False`, seed as in section 3. Then one Poisson `fit(100)` with every parameter free. One
  episode per truth and noise level.
- **D3, true L with data values** (condition 9). A1 from the true-L data start with steps 2–3
  skipped. One episode per truth and noise level.
- **Instrumentation.** Exposure counts (4.1, 4.2); whether each S-arm and A1 episode exhausted E;
  the kept-path true-L state of every A1-family episode (S1, S2, S12, A1, A1o, D3).

## 5. Predictions and refutations (condition 10)

Unit as X2d section 5. A "gap" is the training deviance of an arm's kept-path state at the true L
(median over starts) minus A1o's. It is evaluated over the truths where X2e's own A1 overshoots.

- **D1 first.** Basin reading supported if D1's gap exceeds 26.4 in ≥ 75 % of the overshooting
  truths. Refuted if it is within 26.4 in ≥ 50 %; B1–B3 are then "not applicable" (the problem is the
  fit allowance at step 7), and section 6 still applies.
- **B1 (S1) and B2 (S2), each against A1 on the same truths.** Prediction: the overshoot sign test
  (X2d 1d) holds in ≥ 2 of 4 multilayer cells, and the arm's gap is within 26.4 in ≥ 50 % of the
  overshooting truths. Refuted if the sign test holds in no cell and that fraction is ≤ 10 %.
  Anything between is unresolved.
- **B3 (D2).** Among the overshooting truths where D1's gap exceeds 26.4, supported if D2's gap is
  within 26.4 in ≥ 50 %, refuted if in ≤ 10 %, unresolved between.
- **D3 (reading only, no grade).** E_D3 ≤ 1 in every multilayer cell points to the one-layer path.
  E_D3 ≥ 4 in ≥ 2 cells points to the start values at a fixed L. Anything else is reported as
  unclear.
- **Grading order** for each of B1–B3: "not applicable" (D1 refuted), then "untested" (exposure
  < 10 %, condition 6), then "budget-confounded" (more than 25 % of the arm's multilayer episodes
  exhaust E), then refuted, then supported, then unresolved. A source is the cause only if it is
  supported and the other two are refuted.

## 6. Protocol verdict

- S12 against C under DESIGN 6.3 unchanged, with the 6.5 guards (budget, reward alignment, random
  floor against B).
- **Attribution guard** (condition 5). In each cell where S12 beats C, it must also beat A1 by the
  overshoot sign test (one-sided p < 0.01). The verdict counts the cells that pass both:
  - ≥ 2 cells: passes, attributable. Confirmation run on root 20261028 with the same arms. X2c
    stays unregistered.
  - S12 beats C in ≥ 2 cells, but fewer than 2 pass the guard: "passes, not attributable to the
    start change". No confirmation run.
  - Otherwise the result is a fail. A pass of S1 or S2 under the same two tests while S12 fails is
    "inconclusive, confirm with new seeds".
- A1 against C is reported as the third sample of the X2b question (section 1).

## 7. Cost and order

Episodes: A1, S1, S2, S12, B, C at 384 each (2304). A1o, D2 and D3 at 96 each. D1 runs inside each of
A1's 384 episodes. [unverified] S1's screening adds up to 9 × `fit(20)` per `add` candidate.
D2 costs about 10⁴ forward evaluations per episode. The timing pilot on unregistered seeds comes
first, then the registered run in a [HEAVY] slot.

## Amendment 2: full re-run after #43 (before any outcome is read)

Physics points from the #41 comment of 2026-10-24 11:08 UTC and the #44 review. There is no X2e
amendment 1; the number follows the name used in #41 and #44.

1. **Frozen commit.** The commit that adds this amendment. Its hash is in the #41 description and in
   every record. Against 1c38e61, the only change under `src/reflectolearn/fit_policy` is
   `stack.deviance_terms`, clamped at 0 (#44, fixes #43), and `tests/test_fit_policy.py` gains its two
   tests. The merge of #44 also brings #42 (b214ac4) into this branch: comment and docstring edits
   to ten files outside `fit_policy` (`config.py`, `data/augmentations.py`, `evaluate.py`,
   `physics/abeles.py`, `predict.py`, `refine/pipeline.py`, `report/figures.py` and three test
   files). With docstrings removed, each of them has the same AST before and after. Outside the
   measured code: `run.py` writes error records (item 4), `report.py` refuses them, and the run
   wrapper (not in the repository) now writes the exit code and the last stderr line.
2. **The aborted run.** At 1c38e61, started 2026-10-24 13:09 KST, it stopped at 18:02:05 when A1o
   unit (3, 13, 0, 0) raised `ValueError: Residuals are not finite in the initial point.` (#43). That
   was before the machine's 18:07 bugcheck. 2392 of 2592 records, all at 1c38e61, no duplicate
   units: A1, S1, S2, S12, B and C 384 each, A1o 88 of 96 (missing n = 3, j = 12–15), D2 and D3 none.
   What was looked at: per-arm unit keys, commits and duplicates; C's `capped` flag per cell, for the
   C(2E) guard (at most 0.172, so C(2E) is not needed); the traceback; and a re-run of the 8 missing
   A1o units at 1c38e61, printing only whether each raised (only (3, 13, 0, 0) did). No outcome
   metric was read. The records are kept in `x2e_runs_aborted_1c38e61/` and are not read for any
   result. `report.py` does not read that directory. The re-run starts from an empty `x2e_runs/`.
   The stop was forced by an exception, so it is not optional stopping.
3. **Everything else unchanged.** Root 20261027, the units, the arms, every threshold, the grading
   order, the rules in `report.py`, and the confirmation root 20261028. A1o (3, 13, 0, 0) is re-run
   like every other unit and gets no special handling.
4. **The next exception.** A unit that raises is written to `x2e_runs/errors.jsonl` with its error
   and commit, and the run goes on. It does not count as done. `report.py` refuses a verdict while
   that file has a record, as it does for a missing episode. A fix then goes in as a further
   amendment with a full re-run, the same path as this one.
5. **Comparability with X2b and X2d.** X2e's A1 is the A1 of X2b and X2d plus the clamp. The "third
   sample of the X2b question" (section 6) is reported with that difference named. It does not say
   that X2b or X2d would have come out the same. The 26-episode check in #44 does not show that: its
   counter did not separate least-squares trials, returned deviances and the `deviance_excess`
   draws.
6. **Clamp exposure.** Not counted (physics item 6 is optional). So the run cannot say in how many
   episodes the clamp acted. That is a stated limitation of every X2e comparison with X2b or X2d,
   and it does not affect comparisons inside X2e, where every arm runs the same code.

Cost: all 2592 episodes again. The aborted run did 2392 in 4.9 h on 4 processes; the remaining 200,
mostly D2, add about 0.2 h. [HEAVY] slot: 4 processes, BelowNormal, 6 GB, 8 h.
