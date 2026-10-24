# Why refine stops short of the best fit (R2b, ReflectoLearn#25)

R1c found that `refine.refine` with the default `RefineConfig` ends far from the maximum-likelihood
fit from all 48 starts (three cases × 16), while a Poisson least-squares fit from the same starts
reaches it from 43 of 45. Issue #25 lists three candidate causes without separating them. This
experiment switches them on and off one at a time.

## Pre-registration (written before running)

Registered 2026-10-23, before any of the computations below were run. Changes go in a new commit
that says why; they are not edited in here.

**Data.** The three R1c cases (`mid_oxide`, `mid_no_oxide`, `t2_low`; truth, q grid, 10⁶ counts and
seed 0 as in `nonunique.py`), with one difference: the expected counts are computed with refine's own
forward model (`refine.make_forward`, all effects on, oxide off for `mid_no_oxide`), not with
`fim.py`'s. R1c showed that the two differ by up to 3.6 % at the first grid point (edge padding before
smearing). With refine's model generating the data, the model refine fits is exactly the true one, so
whatever is left is the search.

**Score.** For an endpoint, D is the Poisson deviance of the noisy counts under refine's forward model
(float64 deviance from the float32 model). D* is the lowest D found on that curve by any
configuration below, including a reference run (all factors on, budget ×30, started at the truth).
An endpoint has *reached* the best fit if D − D* ≤ 4. The *reach rate* of a configuration is the
number of reached endpoints out of 48 (16 per case: the 15 R1c starts plus the truth).

**Factors** (each one a change to `RefineConfig` or a stage list; nothing in `src/` changes):

| | factor | off (default) | on |
|---|---|---|---|
| A | step budget | 120 / 200 / 300 Adam steps | ×10 in every stage |
| B | stage path | stage 2 (correlation loss on d, ρ, offset) | stage 2 removed |
| C | objective | log₁₀ MSE over nonzero counts in stage 3 | Poisson deviance per point (counts scale s known, zero counts included) in stage 3 |
| D | Occam prior | `nuisance_reg` 0.2 on oxide thickness and dq | 0 |
| E | trust region | `trust_margin` 0.4 | 1.0 |
| F | step size | stage 3 at lr 0.01 to the end | stage 3 followed by two stages over the same keys at lr 10⁻³ and 10⁻⁴, 300 steps each (×10 with A) |

A, B and C are the three causes listed in #25. D, E and F are added here, before running, because
reading the code shows each can stop refine short of the likelihood optimum on its own: D because
under the log-MSE objective the penalty (about 0.006 at the truth's 10 Å oxide) is of the same order
as the objective at the optimum, so it moves the minimum; E because the roughness trust region of the
`t2_low` start (1.9–4.5 Å) excludes the ML roughness of 0 found in R1c; F because Adam at a fixed
learning rate in the sigmoid latent space jitters by roughly lr × (bound width) / 4 ≈ 0.5 Å in d for
the mid film, larger than its profile half-width (0.1 Å).

**Configurations.** Baseline (all off); each factor alone (6); all on; all on minus one factor (6).
Convergence check: all on with budget ×20 instead of ×10; if its reach rate differs from all on by
more than 2 of 48, the ×10 budget is not converged and that is reported before any verdict.

**Decision rules.**
- A factor is *sufficient* if switching it on alone gives a reach rate ≥ 43/48 (the rate of the
  Poisson least-squares fit in R1c, 43/45 ≈ 96 %, rounded down to allow for the three extra truth
  starts).
- A factor is *necessary* if all-on reaches ≥ 43/48 and all-on minus that factor reaches < 43/48.
- If all-on reaches < 43/48, refine's search is not fixable by these settings alone, and the report
  says so instead of naming a cause.
- The verdict for #25 names the sufficient factors if any, else the necessary ones. A fix is proposed
  only for factors that are sufficient or necessary.

**What would falsify the issue's account.** If the baseline on these refine-model curves already
reaches ≥ 43/48, the R1c failures came from the difference between `fim.py`'s model and refine's,
not from the search, and #25 is reclassified as a model mismatch.

**Cost.** Baseline is about 10 s per case; configurations with A run about 10 times longer. Timed on
one configuration first; the sweep is queued as [HEAVY] if it exceeds 2 minutes.

### Amendments (made after the first runs, with reasons)

1. *Reference for D\*.* After the baseline, the single factors B–F and A on `mid_oxide` had run, the
   lowest deviance any configuration had found (457, 566, 536) was above the deviance of the truth
   itself (300, 288, 323), so "lowest D found by any configuration" was not an optimum and would
   have counted fits worse than the truth as reached. D\* now also includes a reference fit that does
   not use refine's optimizer: bounded least squares on deviance residuals under refine's own forward
   model (central-difference Jacobian, the model is float32), started at the truth, within the
   global bounds. It gives 289.9, 279.0, 313.0, below the truth by 10 in each case, as expected for
   about a dozen fitted parameters. The reference run "ref" (refine, all on, ×30, from the truth) is
   kept as registered. Nothing else changes; no configuration with A had been run on more than one
   case when this was decided.

## Results

Reproduce (about 2.5 hours on CPU with other jobs running; [HEAVY]):

```bash
uv run python experiments/001_identifiability/refine_causes.py reference
uv run python experiments/001_identifiability/refine_causes.py run
uv run python experiments/001_identifiability/refine_causes.py run --configs ABCDEF@20 ref
uv run python experiments/001_identifiability/refine_causes.py report
```

Table: [`refine_causes_tables.md`](refine_causes_tables.md); data: `refine_causes.json`.

| configuration | factors on | reached (of 16 per case) | total | median ΔD |
|---|---|---|---|---|
| base | none | 0 / 0 / 0 | 0/48 | 8.3 × 10⁵ |
| each factor alone | A, B, C, D, E or F | 0 / 0 / 0 for every one | 0/48 | 1.4 × 10⁴ (C) to 9.7 × 10⁵ |
| all on | ABCDEF ×10 | 16 / 16 / 16 | **48/48** | 0.32 |
| all on, ×20 (convergence check) | ABCDEF ×20 | 16 / 16 / 16 | 48/48 | 0.21 |
| all but A (budget ×1) | BCDEF | 0 / 6 / 1 | **7/48** | 34 |
| all but B (stage 2 kept) | ACDEF | 11 / 14 / 7 | **32/48** | 1.6 |
| all but C (log-MSE objective) | ABDEF | 0 / 0 / 0 | **0/48** | 8.7 × 10⁵ |
| all but D (Occam prior on) | ABCEF | 16 / 16 / 16 | 48/48 | 0.34 |
| all but E (trust margin 0.4) | ABCDF | 14 / 16 / 16 | 46/48 | 0.23 |
| all but F (no step-size decay) | ABCDE | 13 / 14 / 10 | **37/48** | 1.4 |

Cases in the order mid_oxide / mid_no_oxide / t2_low. The reference run (all on, ×30, from the truth)
reaches D\* in all three cases.

**Verdict by the registered rules.**
- The baseline reaches 0/48 on curves from refine's own model, so #25 is not a model mismatch between
  `fim.py` and refine (the falsification condition does not apply).
- No factor is sufficient: each one alone reaches 0/48.
- All on reaches 48/48, and the ×20 check also gives 48/48, so the ×10 budget is converged.
- **Necessary: A (step budget), B (removing stage 2), C (Poisson objective) and F (step-size decay).**
  Leaving any one of them out drops the reach rate below 43/48: to 7, 32, 0 and 37 of 48.
- Not necessary: D (Occam prior, 48/48 without it) and E (wider trust region, 46/48 without it).
- The three causes listed in #25 are each necessary and none is sufficient. The fourth necessary
  factor, F, was added at registration: Adam at a fixed learning rate does not settle to within
  Δχ² ≤ 4 of the optimum.

What the leave-one-out rows show beyond the rule (observations):
- *Objective (C).* With every search factor on, the log-MSE fits never reach the ML point. For
  t2_low, all 16 starts end together at ΔD ≈ 3.0 × 10³ (median 3.4 × 10³ against a minimum of 3.3 ×
  10³), so this is where the log-MSE optimum lies, not a failed search. For the mid film the
  log-MSE fits mostly also fail to converge (median ΔD 1.1 × 10⁶). The curves come from refine's own
  model, so this is an error of the objective (log₁₀ MSE over the nonzero counts), not of the model.
  It is the clean form of R1c's 5/48 endpoints that refine's objective preferred to the ML.
- *Stage 2 (B).* With stage 2 kept and the other factors on, the correlation loss moves d to fringe
  aliases from some starts: 520–570 Å for the mid film (from d × 1.1 and × 1.2 with ρ × 1.15), and
  19–330 Å for t2_low. The correlation loss is blind to the overall level and fits the fringe pattern
  only, so it accepts another fringe order. With the default trust margin 0.4 these aliases
  would be outside the bounds for most starts (inference, not run).
- *Budget (A).* Without the ×10 budget, the other five factors get 7/48, with median ΔD 34: close
  but not converged.

### Added after the results: the four necessary factors alone (ABCF)

Not registered. The leave-one-out rows never switch D and E off together, so the combination of the
four necessary factors with the Occam prior and the trust margin at their defaults was run on its own.
That combination is the opt-in preset proposed for #25. It reaches **46/48** (16/16 for
mid_no_oxide and t2_low, 14/16 for mid_oxide), which meets the ≥ 43/48 criterion. The two misses
are the same two starts that all-but-E misses: d × 1.2 with ρ × 0.85 and × 1.0. They end at d
241–245 Å, ΔD ≈ 335 from D\*. Their cause was not separated; the wider trust margin (E) recovers them.

## R2f: out-of-sample confirmation of the preset (registered 2026-10-23, before running)

The physics review of RL#31 points out that 46/48 for ABCF is an in-sample number: the factors were
chosen on the same three curves and the same 48 fits. This check uses new noise and a new film,
with everything below fixed before running.

- **Configuration.** ABCF ×10 as in RL#31 (`RefineConfig.likelihood`), run through this script's
  `build_config` (identical stages) on refine's own forward model, as in R2b.
- **Curves (48 fits).** One Poisson draw each, 10⁶ counts, 2θ 0.2–6°, other nuisances as in R2b:
  - `thin_oxide`: the benchmark's `fixed_thin` film (90 Å / 3 Å / SLD 28), which R2b did not use,
    under the same 10 Å top oxide, fitted with the oxide on (refine's default). Seed 21.
  - `mid_oxide`, new seed 22.
  - `t2_low`, new seed 23.
  Seeds 0–7 have been used before (R1c, R2b, R2c, the reviewer's seed 7) and are excluded.
- **Starts.** 16 per curve, as in R2b: the 15 grid starts plus the truth.
- **Score.** Reached if D − D\* ≤ 4. D\* is the lowest of three things on that curve: every endpoint,
  the least-squares fit under refine's model from the truth, and the same least-squares fit restarted
  from the lowest endpoint. The restart closes the gap R2c found, where a fit from the truth stopped
  in a nuisance basin.
- **Rule.** Confirmed if at least 43 of the 48 reach. Otherwise RL#31 stays opt-in and DECISIONS #17
  goes back to recommending (a) only. Per-curve counts are reported, but the rule uses the total.

### R2f results

`uv run python experiments/001_identifiability/refine_causes.py confirm` (about 6 minutes, 2
threads, with R2a running); data in `refine_confirm.json`.

| curve | seed | reached | misses (start: ΔD, end d) |
|---|---|---|---|
| thin_oxide (new film) | 21 | 15/16 | d × 1.2 ρ × 1: 88.7, 96.6 Å (truth 90) |
| mid_oxide | 22 | 14/16 | d × 1.2 ρ × 0.85 and × 1: 336, 244.6 Å |
| t2_low | 23 | 13/16 | d × 1.2 with ρ × 0.85, × 1, × 1.15: 4.1 each, 201.0 Å |
| total | | **42/48** | |

**Verdict by the registered rule: not confirmed** (42 < 43). RL#31 stays opt-in, and DECISIONS #17
recommends (a). Three of the six misses (t2_low) are 0.1 above the ΔD ≤ 4 threshold. The rule is
applied as registered, without adjusting the threshold.

Observations, not part of the rule:
- Every miss starts from d × 1.2, the start farthest above the truth.
- The two mid_oxide misses repeat the two R2b misses on a new noise draw (same starts, same end at
  d ≈ 244.6 Å, ΔD ≈ 336). So these are a property of the preset from those starts, not of one
  curve. In R2b, a trust margin of 1.0 removed them.
- D\* came from the least-squares fit from the truth in every case. The restart from the lowest
  endpoint found nothing lower.
