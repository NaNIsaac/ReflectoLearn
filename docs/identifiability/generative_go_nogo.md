# Generative inverse model for XRR: go / no-go (R1d)

Written 2026-10-23 after Experiment 001 (R1b, R1c). Nothing was trained; no diffusion or flow model
was started.

## Verdict

**No-go for now.** In every case we checked, the posterior of the film parameters has one mode. The
wrong fits came mostly from the optimizer and partly from refine's log-MSE objective, not from the
curve. Of refine's 48 wrong endpoints, 43 are worse than the maximum-likelihood fit even under that
objective (optimizer failures). The other 5 are preferred to it by the objective (objective or model
error; README, ReflectoLearn#25). A generative posterior model would be
solving a problem these cases do not have. The question that would justify one is open: under which
conditions does a single-film or bilayer XRR curve admit several statistically indistinguishable
structures? That is tomorrow's pre-registered experiment (below). The decision is revisited on its
result.

*Revisited 2026-10-23 after R2a (condition scan, below):* the no-go stands. No analysed condition
has a multimodal label. 13 of the 18 bilayer conditions are undecided, because no start reached the
truth's basin, among them 5 of the 6 bilayer conditions in the decision region (refine bounds,
2θ ≥ 3°). They are to be re-tested in R2h.

## What the four approaches did on the checked cases

Cases: 240 Å / 6 Å / 45 film with and without a 10 Å top oxide, 200 Å / 4 Å / 12 film with the
oxide; 2θ 0.2–6°, 10⁶ counts, Kα2, footprint, offset, resolution, i0, background; all twelve refine
parameters free within refine's bounds (`README.md`, R1c).

| approach | result here | cost per curve (CPU) | status |
|---|---|---|---|
| CNN + refine | CNN not run (no checkpoint on this machine, DotFiles#30). refine alone, from 16 starts within ±20 % of the truth: 48/48 endpoints far from the best fit (ΔD > 4.8 × 10⁴), endpoint d −31 % to +45 % | ~5–10 s for 16 starts | fails as an optimizer: ReflectoLearn#25 |
| multi-start Poisson least squares | 43/45 starts reach the maximum-likelihood fit; the other 2 land on a thickness alias at ΔD ≈ 3.7 × 10⁵, clearly rejected | ~20–30 s for 15 starts | works |
| MCMC (`refine.mcmc`, emcee) | unimodal in all cases; from a failed refine endpoint it moves to the same posterior as from the ML fit. Width vs Poisson profile: agrees for 1 case, 4.7× too wide and 2–3× too narrow for the others | ~12 s, 2000 steps × 52 walkers | works for the mode; its likelihood (Gaussian in log₁₀ R) is not calibrated against counts |
| conditional normalizing flow / flow matching | not trained (out of scope tonight) | training: GPU hours; inference: ms | justified only by amortization or multimodality, neither shown here |

## Already solved by existing methods vs needs several solutions

- Solved: a single film with refine's nuisances on a lab scan to q ≈ 0.43 Å⁻¹. The curve determines
  d to ±0.04–1.4 % (profile Δχ² ≤ 1 at 10⁶ counts). A correct local likelihood fit from a handful of
  starts finds it; MCMC gives the uncertainty. What is missing is a working optimizer and objective in
  refine (#25) and a likelihood that matches the counts, not a new model class.
- Possibly needs several solutions (not shown): a thin top layer with a wide free SLD range and a
  short q range (unbounded Fisher analysis in R1b: d–oxide SLD correlation −0.998, γ up to 1300 at
  q_max 0.21 Å⁻¹); bilayers with similar SLDs (layer-order swaps); thickness aliases (a second
  local minimum near 1.9 d was found, but at ΔD ≈ 3.7 × 10⁵ it is not a competing solution).
- Amortization is a separate argument: if thousands of curves must be inverted (in-situ growth,
  mapping), a trained posterior is fast. PANPE (Starostin et al. 2025) already does this for XRR and
  NR with a normalizing flow, q-offset and normalization nuisances, and importance-sampling/MCMC
  correction. A flow-matching or diffusion version of the same thing is an engineering variant, not
  a research question, unless it addresses something PANPE cannot (see `literature.md`).

## Candidate research questions before graduation

Both are hypotheses; neither has been checked against a full literature search.

1. **Where does XRR become multimodal?** A map, over q range, film/substrate contrast, number of
   layers and how freely the nuisance layer is allowed to vary, of where the Poisson likelihood of a
   curve has several indistinguishable structures (multimodal), one stretched solution (ridge), or a
   well-determined one. Criteria pre-registered, profile likelihood plus global multi-start, checked
   against MCMC. Prior work to separate from: Durant et al. 2021 (neutron, local Fisher only), Gil &
   Windover 2012 (X-ray, one contamination case, MCMC), PANPE (multimodal posteriors shown on
   examples, not mapped). If the map shows a realistic region with multimodality, that region is the
   test bed that justifies a generative posterior, and the comparison with PANPE has a defined target.
2. **How much information do common XRR fitting objectives throw away?** Log-MSE in log₁₀ R (refine,
   GenX's logR-type FOMs) against the Poisson likelihood: efficiency against the Cramér–Rao bound,
   bias at low counts, calibration of the resulting MCMC intervals. Tonight's data already show a
   2–5× miscalibration of `refine.mcmc` widths and endpoints where log-MSE and the likelihood
   disagree on which fit is better. Small, concrete, and directly useful to anyone fitting XRR; the
   risk is that it is known folklore without a paper, which the literature check must settle.

## Pre-registered experiment for 2026-10-23: condition scan for multimodality

Registered before running. Changes go in a new commit with the reason.

**Conditions (36).**
- q_max: 2θ_max ∈ {2.2°, 3°, 6°} (q_max ≈ 0.16, 0.21, 0.43 Å⁻¹), 2θ_min 0.2°, uniform q, 0.02° step
  equivalent.
- Contrast: film SLD ∈ {12, 24, 45} (×10⁻⁶ Å⁻², Si 20.07), d 240 Å, σ 6 Å.
- Nuisance freedom: (i) refine's bounds; (ii) wide: oxide SLD 5–40, oxide thickness 0–60 Å, Kα2
  ratio 0–1.
- Stack: (i) single film with top oxide; (ii) bilayer: the film plus a second film of 120 Å, SLD
  ±25 % of the first, under it, with top oxide.
- Fixed: 10⁶ counts at R = 1, background 10⁻⁷, offset 0, dq 0.002 Å⁻¹, Kα2 0.5, footprint 500,
  oxide 10 Å / 18.9 / 3 Å.

**Per condition, on noise-free data (expected counts):**
1. Global multi-start: 64 starts from a Latin hypercube over the full prior box (film parameters
   over `config.yaml` ranges, nuisances over the condition's bounds); Poisson least squares (as in
   `nonunique.py fit`) from each.
2. Modes: endpoints with Δχ² ≤ 4 (deviance against the noise-free data; the truth has 0) are
   grouped; two endpoints are in different modes if any film parameter differs by more than 3 times
   the larger of their two profile Δχ² ≤ 1 half-widths for that parameter (computed at each endpoint
   by the `nonunique.py` profile on that parameter).
3. Labels as in R1c, applied per film parameter (d of each layer, ρ of each layer): *multimodal* if
   more than one mode with Δχ² ≤ 1 exists; *ridge* if one mode whose Δχ² ≤ 1 profile half-width is
   ≥ 2 % of the value; *identifiable* otherwise.
4. Check: for every condition labelled multimodal, draw one Poisson curve and run emcee with the
   Poisson likelihood from each mode; it counts as confirmed if both modes keep posterior mass
   (each ≥ 5 % of samples after burn-in) in at least one of the runs.

**Decision rule.**
- Go for a generative posterior study (R2) if at least one condition with refine's bounds (nuisance
  freedom (i)) and q_max ≥ 0.21 Å⁻¹ is confirmed multimodal.
- Otherwise no-go stands, and the condition map itself is the result (question 1 above, with
  multimodality only in the wide or short-range corners, if anywhere).

**Cost.** 36 × 64 = 2304 least-squares fits at about 1–5 s each (12–15 parameters), 1–3 hours on one
CPU, under 1 GB. Queued as [HEAVY] in two halves (single film, bilayer). Before that, one condition
is timed and the bilayer forward model is checked against `benchmark.parratt` as in R1b.

**Step 0 (reading, 30 minutes).** PANPE's multimodal examples (which stacks, q ranges, priors), so
that the scan does not re-discover a published case without saying so.

## R2a step 0: PANPE's multimodal cases (read 2026-10-23)

Read: arXiv:2407.18648 v1 HTML (the only arXiv version); the Science Advances version and the
supplementary figures were not read.

- Where PANPE shows several modes: a simulated two-layer XRR curve with 10 free parameters gives 7
  modes under a wide prior and one under a narrow prior (Fig. 2); a simulated four-layer structure is
  "highly ambiguous" (SM Fig. 5, not read); the experimental in situ DIP-on-Si curves (one organic
  layer over a fixed oxide) show two branches in the film SLD, of which the upper one disappears
  later in the growth (Fig. 4); one neutron polymer curve in H₂O has two solutions that a D₂O
  contrast removes (Fig. 6).
- Regime: q_max 0.15 Å⁻¹, 20–64 points, Gaussian noise of 5–30 % relative per point (their
  approximation to counting noise), priors over thickness 0–500 Å, roughness 0–50 Å, SLD 0–60;
  nuisances q offset ±0.002 Å⁻¹ and normalization ±5 %. The numeric prior widths of the
  multimodal figures are not given.
- Stated cause: the phase problem, in general terms. No mechanism (layer swap, thickness exchange,
  zero contrast) is given for any specific case. Zero-contrast bilayers, which make the two
  thicknesses exactly collinear, were removed from their test set (under 5 %), not analysed.
- Not in the paper: any map of where multimodality occurs over q range, contrast or prior width;
  any comparison with a profile likelihood or Fisher information.

Consequence for the scan (no change to the registered rules): the registered scan sits at much
better statistics than PANPE's examples (10⁶ counts at R = 1, 291 points, against 5–30 % per point
over 20–64 points) and up to q_max 0.43 Å⁻¹. A single film being unimodal there does not contradict
PANPE's single-film branches. Because the noise-free deviance scales with the counting level s, the
scan can report for every secondary mode it finds the counting level s* = 10⁶ / ΔD below which that
mode is within Δχ² ≤ 1; this is reported as an addition, not used in the registered decision.

### Amendments to the condition scan (before any condition was run with its 64 starts)

1. *Second film of the bilayer.* The registration gives its thickness (120 Å) and SLD "±25 %" but not
   the sign or its roughness. Fixed here: SLD 1.25 × the first film's (15, 30, 56.25), roughness
   6 Å like the first. One sign keeps the count at 36 conditions; the other sign is not scanned.
2. *Resolution quadrature.* `scan.py` integrates the Gaussian resolution with 15 Gauss–Hermite nodes
   instead of `fim.py`'s 41-point ±5σ rule. Against `fim.expected_counts` the counts differ by at most
   2.2 × 10⁻⁵ relative and by a deviance of 0.0004 on the noise-free curve, far below the Δχ²
   thresholds of 1 and 4. The bilayer stack agrees with `benchmark.parratt` to 3.1 × 10⁻¹⁵ (no
   absorption, 2θ 0.2–6°).
3. *Cost estimate correction (2026-10-23).* The registered estimate (1–5 s per fit) was wrong. Measured with two other runs on the
   machine and 2 threads: about 14 s per fit for a single film at 2θ ≤ 6° and about 60 s for the
   bilayer before the quadrature change, with many fits from Latin-hypercube starts stopping at the
   200-evaluation cap far from any minimum. 2304 fits are therefore several hours of CPU, not 1–3.
   The method and the decision rules are unchanged; the scan runs in the two registered halves
   (single film first), each split over three processes.
4. *Profiles restart the inner fit (2026-10-23, physics review of PR #24).* The per-mode profiles of
   rule 2 compute each fixed-parameter point as the lowest of three local fits (continued from the
   previous point, nuisances at refine's initial values, nuisances at their bound midpoints), so that
   a half-width is not set by one tracked nuisance branch. Modes themselves come from the 64 global
   starts, not from the profiles. A restart that finds a point below the mode's own deviance is
   recorded (`lowest_dchi2` < 0).
5. *Named case added: t2_low Kα2 basin (2026-10-23, before the scan ran; decision rules unchanged).*
   Condition 36 is R1c's t2_low film (200 Å / 4 Å / SLD 12, with the 10 Å top oxide, single film,
   refine's bounds, 2θ 0.2–6°), run like the 36 registered conditions. On its noisy R1c curve two ML
   solutions differ only in the Kα2 ratio (0.51 vs 0.00) within ΔD = 1, and in R2c (amendment 3)
   neither MCMC chain converged, with τ growing with chain length. The registered rules label film
   parameters (d, ρ) only, so a basin that differs only in a nuisance is one mode under them. For this
   condition the report also lists the endpoints within Δχ² ≤ 4 whose nuisances differ (Kα2 ratio
   by more than 0.1, or oxide thickness or SLD by more than 10 %), as an observation. It does not enter
   the go/no-go decision, which keeps its registered wording.

### Decision flow (written 2026-10-23, before any scan condition was analysed)

This restates the registered rules and amendments 1–5 as the order the code applies them in
(`scan.py run`, then `scan.py analyze`). It changes no rule.

1. **Global multi-start.** For each condition, 64 Latin-hypercube starts span the whole prior box,
   and a Poisson least-squares fit runs from each (at most 200 evaluations) on the noise-free
   expected counts. The truth has D = 0, so an endpoint's D is its Δχ².
2. **Candidates.** Endpoints with Δχ² ≤ 4.
3. **Modes.** Candidates are taken in order of D. Each one joins the first existing mode it is not
   distinct from, or starts a new mode. Two endpoints are distinct if some film parameter (d or ρ of
   a layer) differs by more than 3 × the larger of their two profile half-widths for that parameter.
4. **Per-mode profiles.** Each film parameter at each mode's best endpoint gets a Δχ² = 1
   half-width. The parameter is stepped out by doubling until Δχ² rises above 1 or the bound is
   reached, then bisected 6 times. Every point is the lowest of three local fits: continued,
   nuisances at refine's initial values, nuisances at bound midpoints (amendment 4). A restart that
   lands below the mode is recorded as `lowest_dchi2` < 0.
5. **Labels, per film parameter.** *Multimodal* if two or more modes have Δχ² ≤ 1 and differ in
   that parameter by the rule in step 3. *Ridge* if not, and the best mode's half-width is ≥ 2 % of
   the value. *Identifiable* otherwise.
6. **Check of multimodal labels.** Draw one Poisson curve, run emcee with the Poisson likelihood from
   each mode, and confirm if both modes keep ≥ 5 % of the samples after burn-in in at least one run.
   This is not implemented yet, and runs only if step 5 gives a multimodal label.
7. **Decision.** Go if at least one condition with refine's bounds and q_max ≥ 0.21 Å⁻¹ (2θ ≥ 3°)
   is confirmed multimodal in step 6; otherwise no-go stands and the map is the result.

Reported alongside but outside the decision:
- the nearest distinct competitor above Δχ² 4, with s* = 10⁶ / ΔD, the counting level at which it
  would come within Δχ² 1;
- for every condition, endpoints within Δχ² ≤ 4 whose nuisances differ (amendment 5).

**The named case, t2_low Kα2 basin (condition 36).** On one noisy curve, R1c found two ML solutions
that differ only in the Kα2 ratio (0.51 vs 0.00) within ΔD = 1. R2c (amendment 3) did not converge
on that curve, and τ grew with chain length. The scan fits the noise-free curve instead, where the
truth (Kα2 0.5) has D = 0. Expected readings, stated before the result:
- If the second basin shows up as endpoints with Kα2 near 0 inside Δχ² ≤ 4, it lies in the
  likelihood itself.
- If it shows up only above Δχ² 4, it is a feature of that noise draw, within reach at the counts
  given by s*.

Either way the film labels follow steps 3–5, which ignore nuisances.

### Interim (2026-10-23 03:20): fits only, no modes, no labels, no verdict

| # | condition | points | starts within Δχ² ≤ 4 | lowest Δχ² above 4 | fit time (one process, 3 threads) |
|---|---|---|---|---|---|
| 0 | single, refine bounds, 2θ ≤ 2.2°, ρ 12 | 101 | 2/64 | 4.6 × 10⁴ | 558 s |
| 1 | single, refine bounds, 2θ ≤ 2.2°, ρ 24 | 101 | 15/64 | 8.0 × 10³ | 575 s |
| 2 | single, refine bounds, 2θ ≤ 2.2°, ρ 45 | 101 | 13/64 | 1.3 × 10⁴ | 471 s |

These are raw fit counts. Whether the endpoints within Δχ² ≤ 4 form one mode or several is decided
in step 3, after all of a process's conditions are fitted.

### Amendment 6: step 6 alternative (registered 2026-10-23, before any condition was analysed; not run)

Step 6 runs as follows if step 5 gives any multimodal label; the confirmation rule itself is unchanged.
- *Single film:* one Poisson draw (seed 0, 10⁶ counts) from refine's `make_forward` at the truth.
  The chains use the Poisson likelihood under that same float32, walker-batched model. The bounds
  and flat priors are the condition's.
- *Bilayer* (refine's model has one film): the draw and the likelihood use `scan.expected_counts`,
  batched over walkers with `torch.vmap`.
- *Chains:* one run from each mode's best endpoint, in R2c amendment 3's form: coordinates whitened by
  the Poisson Fisher information at that endpoint, 0.8 `DEMove` + 0.2 `DESnookerMove`, 64 walkers,
  5000 steps, then 15000 if the kept half is shorter than 50 τ (5000 only for the bilayer, at about
  0.4 s per step).
- *Confirmed:* both modes keep ≥ 5 % of the kept samples in at least one run. Samples are assigned
  to the nearer mode in the film parameters, scaled by each mode's profile half-widths.
- An unconverged run still counts if it confirms: mass in both modes after burn-in is evidence for
  both, even from a short chain. It does not count against confirmation.

### Interim (2026-10-23, single-film fits finished): fits only, no modes, no labels, no verdict

Film 240 Å / 6 Å unless noted, oxide 10 Å / 18.9 / 3 Å on top. Condition 36 is the named case.
"Lowest Δχ² above 4" is the best endpoint outside the candidate set. It may belong to the same mode
as the candidates (steps 3–4 decide), so it is not a competitor yet. Fit times are with two
processes of 4 threads, except 0–5 (three processes of 3 threads).

| # | nuisance bounds | 2θ max | d / σ / ρ | points | starts within Δχ² ≤ 4 | lowest Δχ² above 4 | fit time |
|---|---|---|---|---|---|---|---|
| 0 | refine | 2.2° | 240/6/12 | 101 | 2/64 | 4.6 × 10⁴ | 558 s |
| 1 | refine | 2.2° | 240/6/24 | 101 | 15/64 | 8.0 × 10³ | 575 s |
| 2 | refine | 2.2° | 240/6/45 | 101 | 13/64 | 1.3 × 10⁴ | 471 s |
| 3 | refine | 3° | 240/6/12 | 141 | 3/64 | 4.7 × 10⁴ | 451 s |
| 4 | refine | 3° | 240/6/24 | 141 | 10/64 | **4.3** | 430 s |
| 5 | refine | 3° | 240/6/45 | 141 | 14/64 | 1.4 × 10⁴ | 438 s |
| 6 | refine | 6° | 240/6/12 | 291 | 1/64 | 4.7 × 10⁴ | 444 s |
| 7 | refine | 6° | 240/6/24 | 291 | 13/64 | **17** | 372 s |
| 8 | refine | 6° | 240/6/45 | 291 | 13/64 | 1.4 × 10⁴ | 323 s |
| 9 | wide | 2.2° | 240/6/12 | 101 | 3/64 | 4.4 × 10⁴ | 278 s |
| 10 | wide | 2.2° | 240/6/24 | 101 | 10/64 | 8.0 × 10³ | 235 s |
| 11 | wide | 2.2° | 240/6/45 | 101 | 11/64 | 1.8 × 10³ | 205 s |
| 12 | wide | 3° | 240/6/12 | 141 | 2/64 | 4.9 × 10³ | 231 s |
| 13 | wide | 3° | 240/6/24 | 141 | 12/64 | **4.3** | 230 s |
| 14 | wide | 3° | 240/6/45 | 141 | 9/64 | 2.1 × 10³ | 199 s |
| 15 | wide | 6° | 240/6/12 | 291 | 3/64 | 4.8 × 10⁴ | 308 s |
| 16 | wide | 6° | 240/6/24 | 291 | 16/64 | **12** | 306 s |
| 17 | wide | 6° | 240/6/45 | 291 | 10/64 | 2.5 × 10³ | 299 s |
| 36 | refine | 6° | 200/4/12 | 291 | 2/64 | 2.5 × 10⁴ | 326 s |

Two things to keep in view for the analysis, without reading anything into them yet:
- Endpoints just above the Δχ² 4 line (4.3, 12, 17) appear only for ρ = 24, the film closest to the
  Si substrate (20.07).
- The film with ρ = 12 (below Si) is reached from only 1–3 of 64 starts in every condition. So for
  that film the search coverage is thin, and a mode that only a few starts would find could be
  missed.

Cost correction (rule 24): the 19 single-film conditions took about an hour of wall time, with 2–3
processes and 3.5–10 minutes per condition. Amendment 3 had expected several hours. With two
processes, the 2θ ≤ 6° conditions took 5–7.5 minutes and the 2.2° conditions 3.5–5 minutes. The
number of evaluations per fit was not compared.

### Single-film half: steps 3–7 (analysed 2026-10-23)

Reproduce: `uv run python experiments/001_identifiability/scan.py analyze --conditions 0 1 ... 17 36`,
then `scan.py report`. Table: [`scan_tables.md`](scan_tables.md); data: `scan/modes_*.json`.

**Step 3, modes.** Every one of the 19 conditions has exactly one mode. Its best endpoint is at
Δχ² 7 × 10⁻¹¹ to 5 × 10⁻¹⁰ (condition 10: 1 × 10⁻⁶). All candidates of a condition are members of
that mode. The endpoints just above Δχ² 4 noted in the interim (4.3, 12, 17, all ρ = 24) are not
distinct from it either: the nearest distinct endpoint of every condition is at ΔD ≥ 1.8 × 10³.

**Step 4, profiles.** No half-width reached a bound. In one condition a restart landed below its
mode: condition 10, d, by 8.7 × 10⁻⁷, with the mode itself at Δχ² 9.6 × 10⁻⁷. That mode is therefore
marked as not fully converged. The amount is far below any threshold and cannot change its labels.

| ρ film | 2θ max | d half-width, refine bounds | d half-width, wide | ρ half-width (both) |
|---|---|---|---|---|
| 12 | 2.2° / 3° / 6° | 2.6 / 2.4 / 1.5 % | 3.5 / 3.2 / 1.9 % | 0.30–0.38 % |
| 24 | 2.2° / 3° / 6° | 3.2 / 2.7 / 2.4 % | 6.7 / 3.8 / 2.8 % | 0.12–0.13 % |
| 45 | 2.2° / 3° / 6° | 0.6 / 0.5 / 0.5 % | 2.5 / 2.2 / 0.8 % | 0.21–0.28 % |

Condition 36 (200/4/12, refine bounds, 6°): d 1.4 %, ρ 0.21 %.

**Step 5, labels.** No condition has a multimodal label. ρ is identifiable in all 19 conditions. d is
a ridge in 12 (half-width ≥ 2 %): ρ 24 everywhere; ρ 12 at 2θ ≤ 3°; ρ 45 with wide nuisances at
2θ ≤ 3°. d is identifiable in 7 (2, 5, 6, 8, 15, 17, 36). Two labels sit close to the 2 % line:
condition 6 (1.48 %, identifiable) and 14 (2.23 %, ridge). Eleven of the twelve ridge half-widths
are 2.2–3.8 %, not far above the line (condition 10: 6.7 %). The ridge/identifiable split of d
therefore depends on where the registered 2 % line is drawn: at 4 % only condition 10 would be a
ridge, at 1.5 % condition 6 would join the ridges. The line is the registered one and is not
moved here.

**Step 6, check of multimodal labels: not applicable.** Step 5 gave no multimodal label, so no chain
was run (amendment 6 stays unused for this half).

**Step 7, decision: no-go stands; the final decision comes after the bilayer half.** The single-film
half has no multimodal label, so it gives no "go", and the no-go of R1d stands. The registered
decision covers all 36 conditions (single film and bilayer). Conditions 18–35 are not run yet, so
this half does not settle it either way.

Physics pre-check (2026-10-23, before this verdict was written): an independent re-application of
the step 5 rules to `scan/modes_*.json` gave the same labels in 19 of 19 conditions. Its points on
scope, search power, label sensitivity and condition 10 are taken into this section.

**Limits of this half.**
- **Weak negatives at low contrast.** The ρ = 12 film is reached from only 1–3 of 64 starts. In
  conditions 0, 6, 12 and 36, it is reached from 1–2 (condition 36: 2/64, with 44 of 64 fits stopping
  at the 200-evaluation cap). Take another basin whose share of the start box equals the truth
  basin's observed share k/64. All 64 starts miss it with probability (1 − k/64)⁶⁴ ≈ e⁻ᵏ: 0.36 for
  k = 1, 0.13 for k = 2, 0.05 for k = 3. For these four conditions, "one mode" is a negative with
  weak search power, not evidence of a single mode.
- **Fits stopped at the evaluation limit.** In each condition, 26–51 of the 64 global fits stopped at
  the 200-evaluation limit (column "at cap" of `scan_tables.md`; the bilayer half gets the same
  column). Such an endpoint is a start that did not reach a minimum. It is neither a mode nor
  evidence against one. Together with the reach column it gives the search power of each condition.
- Noise-free data at 10⁶ counts. A noisy curve moves the minimum but not, to first order, the
  curvature that sets the half-widths.

**Outside the decision: nuisances (amendment 5).** Endpoints within Δχ² ≤ 4 with differing nuisances
appear in conditions 1 and 10 (2θ ≤ 2.2°, ρ 24): Kα2 ratio 0 and oxide thickness 0 at Δχ² 1.4, with
the film the same mode. Condition 36 has no such endpoint, so neither registered reading for the
named case applies: no endpoint with Kα2 near 0 appears at any Δχ² near the film mode.

Unregistered check (added after the labels, does not enter any rule): the profile in the Kα2 ratio
on the noise-free curve, each point the lowest of the three fits of amendment 4.

| condition | D at Kα2 = 0 | D at Kα2 = 0.25 | d at Kα2 = 0 |
|---|---|---|---|
| 36 (200/4/12, 6°) | 0.098 | 0.011 | 200.85 Å |
| 1 (240/6/24, 2.2°) | 0.065 | 0.008 | 233.75 Å |
| 6 (240/6/12, 6°) | 0.091 | 0.011 | 240.99 Å |

At 10⁶ counts the Kα2 ratio is not determined: 0 to 0.5 costs Δχ² < 0.1, with the oxide and d
taking up the difference. D falls monotonically from 0 through 0.25 to the truth (0.5) at these three
points, so this is one flat valley, not two basins. The two t2_low solutions of R1c (Kα2 0.51 and
0.00) are then the two ends of that valley, and the growing τ of R2c fits a chain wandering along it
to the bound. Three points do not exclude a barrier between them.

Note on the run itself: the process watcher of the runner script did not work during the
single-film fits after the restart and during this analysis. A parameter and a local variable had
the same name, case-insensitively, in PowerShell. The watcher never registered the child processes,
so it did not exit and recorded no peak memory. The memory guard summed and stopped the python
processes directly, so it was in force, but this was not observed in a kill. All 19 conditions
finished and wrote their output. Fixed in the runner before the bilayer half.

### Amendment 7: conditions without any endpoint within Δχ² ≤ 4 (2026-10-23)

Written **after the bilayer fits of 14 of the 18 conditions had been seen**, and before any bilayer
condition was analysed. In 10 of those 14 (18, 20, 21, 23, 24, 25, 26, 27, 29, 30), none of the 64
starts ended within Δχ² ≤ 4. The lowest endpoints are at ΔD 1.8 × 10³ to 2.2 × 10⁵. This is not a
rule change. It brings the implementation in line with steps 2–3, which define modes only from
endpoints within Δχ² ≤ 4:
- **Fixed.** `scan.py analyze` labelled every film parameter "identifiable" when there was no mode,
  because the label rules fell through to their last branch. Now a condition with no endpoint within
  Δχ² ≤ 4 has no mode and is labelled *not reached*. When there are modes but none within Δχ² ≤ 1,
  the ridge/identifiable test uses the best mode ("one mode" in rule 3) instead of falling through.
- **Decision.** *Not reached* is evidence neither for "go" nor for "no-go": the search did not reach
  the truth's basin, so the condition is undecided. The step 7 statement gives the number of
  undecided conditions and does not say that the bilayer half has no multimodality.
- **Observation (post hoc, outside the decision).** For a condition without a mode, a local fit from
  the truth and its profiles give the width labels of the truth's basin, and the competitor is taken
  relative to that fit. By construction this sees only the truth's basin, so it says nothing about
  multimodality.
- **Single-film half.** All 19 conditions have a reach ≥ 1/64 and a mode at Δχ² ≤ 10⁻⁶, so their
  labels are unchanged (`scan_tables.md` regenerated, same rows).
- **Follow-up.** With most bilayer conditions not reached, the bilayer half is effectively untested by
  this scan. More starts, a larger evaluation cap, or starts mixed around the truth and across the box
  belong to a separately registered experiment (R2h, registered by the coordinator), not to this one.

Physics (2026-10-23, before this amendment was committed) agreed with the fix and the observation,
and asked for the two qualifications above: "not reached" is not evidence either way, and the
truth-start fit is post hoc and blind to multimodality.

### Bilayer half: fits, and what gets analysed (2026-10-23, before any bilayer analysis)

The 18 bilayer conditions were fitted with 64 starts each (about 2 h of CPU, two processes, peak
1.5 GB). Only five reached the truth's basin: 19 (1/64), 22 (1/64), 28 (4/64), 31 (4/64) and
34 (5/64). The other 13 are *not reached* and therefore undecided (amendment 7). In each condition
41–54 of the 64 fits stopped at the evaluation limit. Steps 3–6 run for the five reached
conditions. The coordinator decided not to run the amendment 7 observation (truth-start profiles)
for the 13: it does not enter the decision, and R2h will test the same conditions again with a
stronger search. The table lists them as "not reached, not analysed".

### Bilayer half: steps 3–7 for the reached conditions (analysed 2026-10-23)

Analysed: the five reached conditions (19, 22, 28, 31, 34), all with the first film at ρ = 24. The
other 13 are *not reached* and undecided (amendment 7). Their truth-start observation was not run
(coordinator's decision above). R2h will test them again.

| # | nuisance | 2θ max | reached | d₁ half-width | ρ₁ | d₂ | ρ₂ | labels d₁ / ρ₁ / d₂ / ρ₂ | nearest distinct ΔD |
|---|---|---|---|---|---|---|---|---|---|
| 19 | refine | 2.2° | 1/64 | 2.83 % | 0.31 % | 0.13 % | 0.21 % | ridge / identifiable / identifiable / identifiable | 1.5 × 10³ |
| 22 | refine | 3° | 1/64 | 2.54 % | 0.28 % | 0.13 % | 0.20 % | ridge / identifiable / identifiable / identifiable | 2.0 × 10⁴ |
| 28 | wide | 2.2° | 4/64 | 3.86 % | 0.32 % | 0.14 % | 0.22 % | ridge / identifiable / identifiable / identifiable | 3.1 × 10⁴ |
| 31 | wide | 3° | 4/64 | 3.23 % | 0.29 % | 0.14 % | 0.21 % | ridge / identifiable / identifiable / identifiable | 3.3 × 10⁴ |
| 34 | wide | 6° | 5/64 | 2.37 % | 0.28 % | 0.14 % | 0.20 % | ridge / identifiable / identifiable / identifiable | 3.3 × 10⁴ |

**Step 3, modes.** Each reached condition has exactly one mode. All its candidates are members.

**Step 4, profiles.** No half-width reached a bound. In conditions 19 and 28, a restart landed below
the mode by 2.8 × 10⁻⁷ and 5.8 × 10⁻⁶, with the modes at Δχ² 2.8 × 10⁻⁷ and 6.2 × 10⁻⁶. Both are
marked as not fully converged, as condition 10 was. The amounts cannot change a label.

**Step 5, labels.** No multimodal label. The top film's thickness d₁ is a ridge in all five
(2.4–3.9 %, close to the 2 % line as in the single-film half). Its ρ₁ and both parameters of the
buried 120 Å film are identifiable (0.13–0.32 %).

**Step 6: not applicable** (no multimodal label).

**Step 7, decision: the no-go of R1d stands.** No analysed condition of either half has a multimodal
label. This does not establish that XRR has no multimodality in this map:
- **13 of the 18 bilayer conditions are undecided**, because no search reached the truth's basin
  (18, 20, 21, 23, 24, 25, 26, 27, 29, 30, 32, 33, 35). Every bilayer condition with ρ₁ = 12 or 45 is
  among them. Five of the six bilayer conditions in the decision region (refine bounds, 2θ ≥ 3°:
  21–26) are undecided; only 22 was analysed. The bilayer half is therefore largely untested, and
  R2h re-tests it with a stronger search.
- **Weak negatives.** Conditions 19 and 22 were reached from 1 of 64 starts. As in the single-film
  half, a second basin of the same size is missed with probability about 0.36. The same holds for
  single-film conditions 0, 6, 12 and 36 (0.13–0.36).
- The labels are for noise-free data at 10⁶ counts and the registered 2 % ridge line. Labels close to
  the line are listed above.

Run: 2 processes × 4 threads, 11–28 minutes per condition, peak 1.4 GB. The planning estimate of
about 17 minutes per condition was short.
