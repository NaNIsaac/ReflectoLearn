# X2d: why protocol A1 adds layers that do not hold out (registration)

Registered 2026-10-24, before any X2d code or run. It replaces `X2D_DRAFT.md`. The HARKing
conditions written by physics on ReflectoLearn#39 (issue comment 6374107811, "section 5") are part of
this registration and govern wherever this text and that comment differ. Section numbers 3–6 without
a prefix refer to `DESIGN.md` (X2a/X2b, frozen at 887c4da).

## 1. What is known, and from where

- [registered, #36] A1 beats C on BIC in every multilayer cell and fails the held-out rule in every
  one. Verdict: no room for RL; X2c not registered (DECISIONS #25).
- [post hoc, stated in #36] In 8 of the 10 losing (truth, cell) pairs A1 returned 4 layers on every
  start. C's fits stop far below their allowance; C_oracle recovers 0–2 % of multilayer truths.
- [code, `protocol.py` at 77bbef3] Step 7 refits each candidate (two `fit(100)`) but compares it
  with the current state unrefit. The current state at step 7 and the returned model come from
  `revert`, which goes to the lowest BIC visited anywhere in the episode, rejected candidate forks
  included (`env.py` `_evaluate` updates one shared best). A candidate rejected with
  −10 < ΔBIC < 0 can therefore become the state step 7 continues from, and the state step 9 returns.

The post hoc facts motivated this study. They are not evidence for any hypothesis below. No X2b
record is analysed further before the X2d run (physics 5.1.2).

## 2. Hypotheses

- **H1, unmatched refit.** A step-7 ΔBIC ≤ −10 can come from the candidate's extra fitting rather
  than its extra layer.
- **H2, a wrong basin compensated by layers.** From a wrong basin, extra layers lower the training
  deviance by reshaping the profile without describing the sample.
- **H3, the split.** The contiguous held-out blocks leave q ranges unconstrained, so even the right
  structure can lose on held-out points.
- **H4, the best-BIC return bypasses the margin.** Through the shared best state, rejected candidates
  (−10 < ΔBIC < 0) are continued from and returned, so the −10 margin bounds the kept path only.

Each is graded on its own (section 6). A hypothesis is called the cause only if it is supported and
the others are refuted.

## 3. Data, seeds, freeze

Carried over unchanged from DESIGN sections 3–6: classes (1, 2, 3 layers), M = 16 truths per class,
noise levels S = 10⁸ and 10⁶, grid, held-out blocks 2 and 6, bounds, budget E = 3000, L_max = 4,
every threshold of 6.2–6.5 (rule 3 stays at ≤ 1 of 16), and C as frozen. C's weakness (early stops)
is a stated limitation; a stronger C would be a separate registration.

- **Seeds.** The root 20261025 replaces 20261024 in every data stream: truths `(20261025, n, j)`,
  fitted noise `(20261025, n, j, s, 0)`, noise-scale realisations `(20261025, n, j, s, r)`, starts
  `(20261025, n, j, 2)`, C's extra layers `(20261025, n, j, 1)`, B's actions
  `(20261025, n, j, s, k, 7)`. The step-7 Monte Carlo seed `(20261024, 99, 0)` is part of the
  protocol and stays.
- **Pilots.** Debugging and timing only on unregistered seeds (class 9 or truth `1000 + n`), reading
  time, evaluation count and memory only.
- **Freeze.** At the commit of the first registered episode; every record carries it. The report
  refuses to run on a mixed commit, a missing episode, or a budget-limited cell without C(2E).
- **No optional stopping.** No truths are added after the run; an inconclusive result stays so.

## 4. Arms

All arms run on the same truths, noise and starts. "Kept path" is the sequence of states created by
the actions of the env that a protocol run ends with, together with the forks it adopted (a candidate
or merge fork that became the protocol's env). States created on forks that were not adopted are
off the kept path, even if a later `revert` reaches them.

- **A1**, unchanged from X2b: the replication control.
- **A1m, matched refit (H1).** At every step-7 decision the current state is refit on a fork exactly
  as a candidate is (`loss(log)`, `fit(100)`, open all, `loss(poisson)`, `fit(100)`); the best
  candidate is accepted if its BIC is at least 10 below that refit state; otherwise the protocol
  continues from the refit state to step 9. Step 8 compares a merged fork with a refit of the state
  it came from in the same way, keeping the merge if its BIC does not exceed the refit's. Every
  evaluation is charged to E. Returned model and `revert` as A1 (shared best).
- **A1k, kept-path return (H4).** A1, except that `revert` and the returned model use the
  lowest-BIC state on the kept path, not the episode-wide best.
- **A1mk, primary.** A1m and A1k together: matched refits, kept-path `revert` and return. Named
  primary because H1 and H4 are both read off the code (section 1), not off X2b's numbers, and a
  protocol meant for use has to remove both. A1m and A1k are its single-change arms for the
  hypothesis tests.
- **A1v, inner validation.** A1mk, with an inner block held out of every fit in steps 1–8: training
  block 4 of the 10 (q 0.1964–0.2348 Å⁻¹, 28 points), fixed as the training block midway between the
  outer held-out blocks 2 and 6, not chosen from data. A step-7 candidate is accepted only if (a) its
  BIC on the fit points (training minus inner, n = 208) is at least 10 below the refit current state's
  and (b) its Poisson deviance on the inner block is strictly lower. A step-8 merge is kept if neither
  the fit-point BIC nor the inner deviance rises. Step 9 refits on all training points (inner block
  rejoined). The returned model is the lowest-BIC state, on all training points, of the kept path,
  whose acceptances all passed the inner check.
- **Cv.** C with L chosen by the inner block: each of L = 1..4 is fit as in C (750 evaluations) on
  training minus inner; the L with the lowest inner deviance is returned as fit.
- **C**, as frozen, and **C(2E)** where the budget guard requires it.
- **B**, random actions as X2b, for the random-floor guard of the primary.
- **A1o (diagnostic).** A1 started at the truth (true L, layers and instrument), with steps 2–3
  skipped; one episode per truth and noise level.
- **TF (diagnostic).** The truth's L and values as the start, every parameter free, Poisson loss,
  one fit with `max_nfev` = E, the fit returned.
- **Instrumentation** (behaviour unchanged): at every step-7 decision of A1m, record L, ΔBIC of the
  best candidate against the unrefit state (A1's comparison) and against the refit state. In every
  A1-family episode (A1, A1m, A1k, A1mk, A1v, A1o), record whether the returned state lies on the
  kept path, and the training deviance of the last kept-path state with L equal to the true L
  (absent if never reached).

A1o, TF and the instrumentation feed only section 6. They never enter a protocol verdict.

## 5. Unit

As DESIGN 6.2: truth level, median over the four starts, per multilayer cell. E_X is the number of
truths in a cell whose median returned L exceeds the true L (median of four integers; a median of
x.5 counts as exceeding when x.5 > true L).

## 6. Predictions and refutations (physics 5.3)

- **Replication (precondition).** Prediction: A1 fails rule 3 against C in ≥ 3 of 4 multilayer cells,
  with E_A1 ≥ 4 in each of them. If A1 fails rule 3 in ≤ 1 cell, the X2b pattern did not replicate:
  H1–H4 are reported as untested and X2d proposes no protocol change. Two failing cells: replication
  partial, hypotheses graded but reported as such.
- **H1.** Prediction: (i) of the A1m decisions whose unmatched comparison would accept (≤ −10), the
  matched comparison rejects ≥ 50 %; (ii) E_A1m < E_A1 by a one-sided paired sign test over truths
  (per truth: A1m's median returned L below A1's counts for, above counts against), p < 0.01, in ≥ 2
  of 4 cells. Refuted if (i) is < 10 % or (ii) holds in no cell.
- **H2.** Prediction: (i) for the truths where A1 overshoots L, A1's kept-path state at the true L
  (median over starts of its training deviance) is more than 26.4 above A1o's state at the true L on
  the same data in ≥ 75 % of those truths; (ii) E_A1o ≤ 1 in every multilayer cell. Refuted if
  E_A1o ≥ 4 in ≥ 2 cells, or (i) is ≤ 25 %.
- **H3.** Prediction: TF loses on held-out against C (by 2 s_ho, as rule 3) on ≥ 2 of 16 truths in
  ≥ 2 cells. Refuted if TF has ≤ 1 such loss in every cell. TF's D_ho is also reported against the
  noise-scale mean + 2 s_ho. [inference, physics] A fitted true structure has D_ho ≈ n_ho ± s_ho, so
  H3 is expected to be refuted.
- **H4.** Prediction: of A1's episodes with returned L > true L, ≥ 50 % return a state off the kept
  path, and E_A1k < E_A1 as in H1 (ii). Refuted if ≤ 10 % return off the kept path.
- Between the bounds a hypothesis is unresolved. The fractions in H1 (i), H2 (i) and H4 are fractions
  of decisions or episodes, not tests.

## 7. Protocol verdict (physics 5.4)

- The primary protocol A1mk is tested against C under DESIGN 6.3 unchanged, with the 6.5 guards
  (budget, reward alignment, random floor against B).
- A1v is tested against C and against Cv, and must beat both. A1m and A1k are secondary. A pass of a
  secondary arm with a primary failure is "inconclusive, confirm with new seeds", never "procedure
  matters".
- A pass does not register X2c. A pass of the primary leads only to a confirmation run on a third
  seed root (20261026) with the same arms; no learning-room variants are defined here.

## 8. Cost and order of work

Episodes: A1, A1m, A1k, A1mk, A1v, B, C, Cv at 384 each (3072), A1o and TF at 96 each; C(2E) only if
the guard requires it. [unverified] The matched refits add about a third to A1's evaluations, so the
run is about 1.2–1.5 times X2b's (3.75 h with 4–6 processes). Order: implement (arms, kept-path
tracking, inner block, instrumentation, `report_x2d.py` applying sections 5–7 mechanically), tests,
timing pilot on unregistered seeds, then the registered run in a [HEAVY] slot.

## Amendments

1. 2026-10-24, before any registered episode and before the implementation was complete; nothing
   had run on registered seeds. From the physics check of 157f2f2 (#39, comment 6374226323), items
   (a)–(e), and from writing the code, items (f)–(h). No X2b record was used.
   - (a) **A1v is secondary.** If A1v beats C and Cv while A1mk fails, the result is
     "inconclusive, confirm with new seeds" (section 7).
   - (b) **A1v and Cv end the same way.** Cv refits its chosen L on all training points with A1v's
     step-9 refit (open all, `loss(poisson)`, `fit(100)`), charged to E. Each of its four L fits gets
     (E − 100) / 4 = 725 evaluations instead of 750. C itself is unchanged.
   - (c) **Replication, the unclassified case.** A1 fails rule 3 in ≥ 3 cells but E_A1 < 4 in at
     least one of them: "partial", graded as the two-cell case.
   - (d) **H1 (ii) and H4 compare overshoot, not returned L.** Per truth, the overshoot of an arm is
     max(median returned L − true L, 0). The sign test counts truths where the arm's overshoot is
     below A1's as for and above as against; ties, including both 0, drop out. One-sided, p < 0.01,
     in ≥ 2 of 4 multilayer cells.
   - (e) **A1m step 8.** When a merge is rejected, the protocol continues from the refit state, as in
     step 7. This applies to A1mk and A1v too.
   - (f) **H2 (i) where A1 never reached the true L on its kept path.** For each truth where A1
     overshoots, the per-truth value is the median over the starts that have a kept-path state at the
     true L. Truths with no such start are left out of the fraction and counted in the report.
   - (g) **Grading order.** A hypothesis whose refutation condition holds is refuted. Otherwise it is
     supported if all its predictions hold, and unresolved otherwise.
   - (h) **Inner block everywhere in A1v and Cv.** The inner block is left out of every quantity
     computed in steps 1–8: the fits, BIC, step-7 trigger, residual features, and the critical-edge
     and Kiessig estimates. Cv's start estimates (section 4, as C's) also use the training points
     minus the inner block.
