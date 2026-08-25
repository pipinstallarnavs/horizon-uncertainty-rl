# Random-Horizon Trading — Detailed Design Document (v2)

**Companion to README.md.** Where the README identifies problems and states the chosen mitigation, this document goes one level deeper: for every problem and threat, it specifies a concrete, implementable solution — algorithms, prompt templates, formulas, decision rules, and acceptance criteria.

---

## Part I — Core design problems and their full solutions

### P1. Memoryless exit collapses the problem into ordinary discounting

**Problem.** Constant hazard `h(t) = p` makes the objective identical to discounted reward with `γ = 1 − p`; the optimal policy is stationary and nothing temporal is tested.

**Solution (specification level).**
Use exit distributions with non-constant hazard, and *prove* non-collapse before running anything:

1. Implement a hazard module where any exit distribution is defined by its pmf `P(T = t)`, from which the code derives:
   - survival `S(t) = P(T > t)`
   - hazard `h(t) = P(T = t)/S(t−1)`
   - conditional mean remaining time `m(t) = E[T − t | T > t]`
2. **Automated non-collapse check** (run in CI before experiments): solve the backward induction (see P6) and assert that the optimal action table `π*(t, holding, age)` is *not constant in `t`* for fixed `(holding, age)` on at least 20% of reachable states. If the assertion fails, the environment is degenerate — fix parameters before spending API budget.
3. Ship three distributions:
   - `UniformExit(N=100)`: `h(t) = 1/(N−t+1)` — rising hazard, gradual updating.
   - `BimodalExit(early=[1,10], late=[91,100], w=0.5)`: hazard spike → zero → spike; creates the day-11 information event.
   - `GeometricExit(p=0.02, truncated at N)`: control condition; the same assertion above should *nearly fail* here (policy nearly stationary except truncation effects) — this validates that the checker itself works.

**Acceptance criterion.** Non-collapse check passes for uniform and bimodal; approximately fails for geometric.

---

### P2. Attribution — separating temporal-reasoning failure from every other failure

**Problem.** In a rich environment, poor LLM performance could come from market prediction, arithmetic, parsing, or context handling rather than temporal reasoning.

**Solution: a layered isolation stack.** Each layer removes one confound; together they make temporal reasoning the only remaining hard thing.

| Layer | Confound removed | Concrete implementation |
|---|---|---|
| Disclosed dynamics | Market prediction | The prompt contains the literal return table (see prompt template, Part III). Nothing about assets is hidden. |
| Deterministic base market | Luck/noise | Base runs have zero return noise; noise is a separate robustness arm. |
| Discrete 4-way action | Arithmetic and sizing errors | Action grammar is a single token choice; no numbers computed by the model affect execution. |
| Strict output grammar + repair loop | Parsing failures | Response must match regex `^(HOLD|SWITCH (A|B|C|CASH))$` after stripping. On mismatch: one automated re-ask with the error quoted. Second failure → logged as `format_error`, action defaults to HOLD, episode flagged. Format-error rate reported separately; episodes with >5% format errors excluded from reasoning metrics (but counted in a data-quality table). |
| Compact state summary | Context length | See T4 solution below. |
| Belief probe | Belief vs. action confusion | See P7. |

**Additional diagnostic: "oracle-time" ablation.** Run a variant where the prompt *tells the agent the exact realized exit day* ("you will be liquidated on day 63"). Under full information, temporal reasoning reduces to deterministic planning. If the LLM also fails *here*, the deficit is planning, not horizon-uncertainty handling — a crucial attribution fork. Expected-regret ordering that cleanly attributes:
`regret(oracle) ≤ regret(known-dist) ≤ regret(unknown-dist)`; the *gaps* between adjacent levels measure (a) uncertainty-handling cost and (b) ambiguity cost respectively.

---

### P3. Horizon-dependence of the market — guaranteeing that time matters

**Problem.** If one asset dominates at all horizons, the exit distribution is irrelevant.

**Solution: calibration by search, with formal targets.** Do not hand-tune asset numbers; select them by an automated search over parameters.

Parameterization (per README §4): Sprinter `(r_hi, r_lo, k_A)` = (boost rate, decay rate, boost length); Compounder `(r_B)`; Late bloomer `(r_early, r_late, k_C)`.

Calibration targets (all checked against the exact solver):

1. **Interior crossover:** under uniform exit, the optimal one-shot holding flips between assets at some `t*` with `15 ≤ t* ≤ 85`.
2. **Value separation:** `V*` beats the best fixed-asset policy by ≥ 5% expected terminal wealth, and beats cash-only by ≥ 15%.
3. **No dominant asset:** each of A, B, C is the optimal *initial* holding under at least one of {uniform, bimodal-early-mode-alive, bimodal-late-mode-resolved}.
4. **Cost-relevance:** with `c = 0.1%`, the optimal policy makes between 1 and 4 switches per episode in expectation (if 0, costs are too high to reward adaptation; if >6, churn is free and timing signal is diluted).
5. **Bright-line strength (bimodal):** the optimal action *changes* at day 11 for the majority of reachable states, and executing the pre-day-11 policy after day 11 loses ≥ 3% terminal wealth vs. `π*`.

Search procedure: coarse grid over the 6 parameters (~10^4 combos), evaluate targets with the solver (each solve <1s), keep the feasible set, pick the combo maximizing target-2 margin. Log the full feasible set for the appendix (shows robustness of conclusions to parameter choice).

---

### P4. Transaction costs — making trade *timing* informative

**Problem.** Free rebalancing lets churn mask whether trades track information.

**Solution.**
- Cost `c = 0.1%` of wealth per switch, applied at execution; HOLD is free. Chosen via calibration target 4 above; report a sensitivity sweep `c ∈ {0, 0.05%, 0.1%, 0.3%}` in the appendix.
- **Trade-timing alignment metric** (formalizing "timed to events").

  Information intensity `I(t) = |m(t) − m(t−1)|`, normalized as

  `align = Σ_t 1[trade at t] · I(t) / (n_trades · mean(I))`

  with `align ≈ 1` for information-blind trading and `align >> 1` for event-timed
  trading. Report with bootstrap CIs over episodes.

  **Validated 2026-08-27, and it constrains the exit distribution.** This metric
  only works if the early mode is *concentrated*. Measured on the originally
  specified bimodal distribution (mass spread over days 1–10), `I(t)` peaked at
  day 10 with a value of 6.77 and sat at 1.00 on day 11 — **0.83× its own mean**,
  so a perfectly-timed day-11 trade would have scored as *below-average*
  alignment, the exact opposite of the intent. The cause was that spread mass
  resolves the posterior gradually, leaving nothing to learn at day 11. A
  KL-divergence variant was also checked and peaked at day 99.

  With the early mass concentrated on a single day (see §3.1), `I(t)` peaks at
  day 10 at **41.75 against a mean of 1.36 — a 30× spike** — and the metric
  behaves exactly as specified. The concentrated form is therefore not a
  cosmetic choice; the alignment metric depends on it.

  **Complementary measure: decision-relevance intensity.** Because `I(t)`
  measures belief change rather than decision change, report alongside it

  `D(t) = V*(t, x_t) − Q*(t, x_t, a_{t−1})`

  the wealth lost by carrying yesterday's action into day `t`, available free
  from the solver's per-state value losses (P6). `D(t)` is zero while the optimal
  action is unchanged and spikes where the policy turns over, so it separates
  "traded when beliefs moved" from "traded when it mattered".
- **Churn guard:** episodes report switches/episode; agents exceeding 2× the optimal switch count are analyzed separately (over-trading is itself a finding, likely reflecting per-step recency bias).

---

### P5. Risk vs. ambiguity — cleanly separating the two unknowns

**Problem.** "Doesn't know T" conflates unknown realization (risk) with unknown distribution (ambiguity).

**Solution: a 3-condition information ladder with matched prompts** (only the exit-information paragraph differs; all else byte-identical):

- **K (known dist):** prompt includes the pmf as an explicit table *and* a one-line plain-language gloss ("equally likely to be any day from 1 to 100"). Both formats included to avoid testing table-reading rather than reasoning; a prompt-variant arm (Part III) separates them.
- **U (unknown dist):** "Your position will be force-liquidated at a random time. You have no information about its distribution beyond that it occurs by day 100."
- **O (oracle):** exact exit day disclosed (see P2). Serves as the planning ceiling.

Analysis: the K−O gap = cost of handling *risk*; U−K gap = cost of handling *ambiguity*. Under U, also characterize the implied prior: fit the agent's behavior (via its belief probes and switch times) to a best-matching exit distribution — does the LLM default to something like uniform? Geometric? This is a standalone descriptive finding.

---

### P6. Ground truth — exact solver, fully specified

**Problem.** Without an optimal baseline, results are uninterpretable.

**Solution: backward induction over the exact state space.**

State `x = (t, holding ∈ {A,B,C,Cash}, age ∈ {1..t})` where `age` = days current position has been held. Value `V(x)` = expected wealth *multiplier* from state `x` to termination (wealth factors out by multiplicativity).

```
for t = N down to 1:
  for each (holding, age):
    stay  = g(holding, age+1) * Vnext(holding, age+1)      # g = 1 + daily return at that age
    best_switch = max over a' ≠ holding of (1−c) * g(a', 1) * Vnext(a', 1)
    cont  = max(stay, best_switch)
    V(t, holding, age) = h(t) * 1 + (1 − h(t)) * cont
    π*(t, holding, age) = argmax
# Vnext at t = N is 1 (forced liquidation); h(N) = 1 by construction
```

- Complexity: O(N² · |A|²) ≈ 160k operations — milliseconds.
- Deliverables from the solver: `V*` (scalar), `π*` table (CSV artifact), crossover day(s) `t*`, optimal switch-day distribution, and per-state value losses `V*(x) − Q*(x, a)` for every suboptimal action `a` — used later to weight LLM errors by how costly they were, not just count them.
- **Verification:** Monte-Carlo rollouts of `π*` (10^5 episodes) must match `V*` within standard error; a brute-force enumeration on a tiny variant (N = 8) must match exactly.

---

### P7. Belief vs. action — decomposing *where* reasoning breaks

**Problem.** An agent can fail by forming wrong beliefs about remaining time, or by holding right beliefs and not acting on them. Aggregate regret can't tell these apart.

**Solution: the elicit–act–compare protocol.**

1. Each day, before the action, the agent answers one auxiliary question: "Estimate the expected number of days remaining before liquidation (a single number)." Logged, never scored in-episode, never fed back.
2. **Belief error** at `t`: `b_err(t) = |belief_t − m(t)|` where `m(t)` is the true conditional mean. Report the trajectory of `b_err` over `t` per condition.
3. **Conditional policy consistency:** define the *belief-consistent action* `a_bc(t)` = optimal action if the true remaining-time distribution were a point mass (or matched-mean distribution) at the agent's stated belief. Then classify each day:
   - beliefs right + action = `π*` → correct
   - beliefs wrong + action = `a_bc` → **belief failure** (reasoned correctly from wrong premise)
   - beliefs right + action ≠ `π*` → **execution failure** (knew the time, didn't use it)
   - both wrong → compound failure
4. Headline decomposition: share of regret attributable to each class (weight each wrong action by its solver-computed value loss from P6).

**Guard against probe contamination:** the probe itself may *induce* temporal reasoning (asking about time reminds the model to think about time). Run a no-probe arm for each main condition; if regret differs significantly between probe and no-probe arms, report both and treat the probe arm as "prompted temporal awareness" — itself an interesting manipulation, not a bug.

---

### P8. Confirmation bias — designing for either outcome

**Problem.** The motivating hypothesis ("LLMs are temporally rigid") must not be baked into the design or reporting.

**Solution.**
- Pre-register (even informally, in the repo: `PREREGISTRATION.md`, committed before first LLM run) the metrics, conditions, exclusion rules, and the *two* interpretation templates — one for "LLMs adapt" and one for "LLMs are rigid" — so the write-up is committed to symmetric treatment.
- Include prompt variants that *help* the model (explicit hazard table, worked example of the survival update in a different domain) and variants that don't. If the model succeeds when scaffolded, the finding becomes "temporal reasoning is present but not spontaneous" — a graded conclusion, pre-planned.
- Report per-model results; do not pool across models into a single "LLMs do X" claim.

---

## Part II — Threats to validity, each with a proposed solution (upgrading README §11)

### T1. Per-holding return clocks are stylized

**Threat.** Real assets don't pay returns as a function of how long *you* have held them; reviewers may call the market artificial.

**Solutions (three, layered):**
1. **Reframe honestly:** the per-holding clock is a *device for identification*, analogous to synthetic gridworlds in RL. State this in the paper; cite the tradition of stylized environments for isolating one capability.
2. **Calendar-clock variant (robustness arm):** returns depend on *calendar* day (asset A strong in days 1–30, C strong in days 60–100) rather than holding age. This is closer to regimes/seasonality and removes the "your holding changes the asset" oddity. The solver handles it with a smaller state `(t, holding)`. If conclusions replicate, the stylization is not driving results.
3. **Historical-data bridge (Phase 2):** replay real multi-asset historical windows with the same random-exit overlay. No ground-truth optimum exists, so metrics shift to (a) comparison vs. fixed-horizon prompted baselines and (b) the clock coefficient — both computable without an oracle. Positioned explicitly as external-validity evidence, not the core identification.

### T2. Full disclosure of dynamics is unrealistic

**Threat.** Real traders don't know return processes; disclosure may make the task "too easy" or unrepresentative.

**Solutions:**
1. Retain disclosure in the identification arms (it is the *point* — see P2), and add a **hidden-dynamics arm**: same market, dynamics not stated; the agent gets only realized prices. Report the interaction: does horizon adaptation degrade more than baseline performance when dynamics are hidden? That interaction is itself informative (does market uncertainty crowd out temporal reasoning?).
2. In hidden-dynamics runs, keep the solver-based optimum as the *clairvoyant* benchmark and add a *statistical* benchmark (an agent doing Bayesian estimation of the return table with the correct model class) so the LLM is compared against an achievable standard, not just the clairvoyant one.

### T3. Prompt sensitivity

**Threat.** Conclusions may hinge on phrasing.

**Solutions:**
1. **Prompt battery, pre-specified:** for each condition, 4 canonical variants — (i) distribution as a probability table, (ii) verbal description, (iii) hazard framing ("each day you survive, the chance of exiting tomorrow becomes..."), (iv) minimal framing. All results reported as mean ± range across the battery; a conclusion must hold in ≥ 3 of 4 variants to be claimed.
2. **Paraphrase robustness:** auto-generate 3 paraphrases of the best variant (different surface wording, identical content) and confirm metric stability (max metric deviation < 1 SE).
3. Commit the full prompt texts in the repo; no post-hoc prompt fixing after seeing results (any prompt change after first full run → rerun the whole grid, documented in a changelog).

### T4. Context length over 100-day episodes

**Threat.** Long histories may degrade performance for reasons unrelated to temporal reasoning (context rot).

**Solutions:**
1. **Compact state protocol (default):** the daily prompt contains the *rules* (fixed, ~300 tokens), the *current state* (day, holding, age, wealth — 1 line), and a *compressed history*: only the list of past switches with their days (not all 100 days), e.g. `Switches so far: day 1 → B; day 42 → A`. Everything the agent needs to compute beliefs is derivable from `t` alone (plus disclosed distribution), so full history is unnecessary by construction.
2. **History ablation arm:** (a) no history, (b) compressed (default), (c) full verbose history. If (c) underperforms (b), that quantifies context-rot cost separately from temporal reasoning — report it, don't let it contaminate the main arms.
3. Token budget per episode ≈ rules once + ~40 tokens/day → well under any frontier context limit; verified by a token counter in the runner with a hard alarm at 50% of the model's window.

### T5. Stochastic LLM outputs and statistical power

**Threat.** LLM sampling noise + random exit times → noisy metrics; underpowered comparisons invite false conclusions.

**Solutions:**
1. **Common random numbers:** fix a set of 200 exit-time realizations (per distribution) and, in the noise arm, 200 market noise seeds; *every* agent and baseline is evaluated on the identical set. Differences between agents are then paired, slashing variance; report paired bootstrap CIs and paired permutation tests.
2. Temperature 0 (or provider minimum) for the main arms; a temperature-sensitivity arm at T = 0.7 with 3 samples/decision (majority vote logged separately from single-sample).
3. **Power check before scaling:** pilot 30 episodes, estimate metric variance, compute the episode count needed to detect a regret difference of 2% of `V*` at 80% power; scale N episodes accordingly (budget permitting; otherwise report the minimum detectable effect).
4. Pre-specified exclusion rules only (format-error rule from P2); all exclusions counted and reported.

### T6. Model heterogeneity and API drift

**Threat.** Provider-side model updates mid-experiment silently change behavior; results not reproducible.

**Solutions:**
1. Pin model versions/snapshot IDs; record them in every result file. Rerun a 20-episode canary weekly; if canary metrics shift > 2 SE, flag and segment results by date.
2. Include ≥ 1 open-weights model with a fixed local checkpoint as the reproducibility anchor (weights hash recorded).
3. All raw transcripts logged (prompt + completion + parsed action + belief), enabling third-party recomputation of every metric from artifacts alone.

### T7. The probe changes the phenomenon (observer effect)

**Threat.** Asking for a remaining-time estimate each day may itself teach the model to attend to time.

**Solution.** Covered in P7's guard: paired probe/no-probe arms for each main condition; the difference is reported as the *effect of prompted temporal awareness* — converted from a validity threat into a measured manipulation.

### T8. Degenerate LLM strategies that score well without temporal reasoning

**Threat.** An agent could luck into decent wealth via a simple heuristic (e.g., "always hold B") without any time reasoning, muddying interpretation of mid-range regret.

**Solutions:**
1. Calibration target 2 (P3) already caps how well any fixed policy can do (≥ 5% worse than `V*`).
2. Report the **strategy-identification analysis**: for each agent, compute distance between its empirical policy and (a) each fixed policy, (b) `π*`, (c) the myopic-greedy policy (maximize next-day return only). Classify which reference policy the agent is closest to. "The LLM approximately implements always-B" is then a stated finding, not a hidden confound.
3. The bimodal bright-line (P1/P3 target 5) is immune to fixed-policy mimicry by construction: no time-blind policy can pass the day-11 checkpoint.

### T9. Data contamination / familiarity

**Threat.** Frontier models may have seen random-horizon portfolio theory; they might "recite" lifecycle heuristics without genuine in-context reasoning.

**Solutions:**
1. This concern cuts the *right* way for once: reciting the correct heuristic and applying it correctly *is* the capability under test (in-context application, not derivation). State this framing explicitly.
2. To separate recitation from application: include a **transfer variant** with a structurally novel exit distribution (e.g., hazard follows a sawtooth) unlikely to match any textbook case. If performance holds, the capability generalizes; if it collapses to lifecycle boilerplate, that's the finding.
3. Never name the problem in prompts ("random horizon", "Merton", "lifecycle") — use only operational language.

### T10. Cost and scale management

**Threat.** The full grid (2 exits × 3 info conditions × probe/no-probe × 4 prompt variants × history ablations × models × ≥200 episodes × ~100 days) explodes combinatorially.

**Solutions:**
1. **Tiered grid:** Tier 1 (headline): uniform + bimodal × K + U × default prompt × probe arm × all models × 200 episodes. Tier 2 (ablations): remaining factors varied one-at-a-time from the Tier-1 default, 60 episodes each. No full factorial.
2. Prompt caching (rules text identical across all calls) and batch APIs where available; estimated Tier-1 cost computed and logged before launch.
3. All scripted baselines and the solver are free — run the entire analysis pipeline end-to-end on scripted agents *first* (a full "dry run" producing every plot with fake agents) so the expensive LLM runs happen exactly once against a debugged pipeline.

---

## Part III — Concrete prompt templates (Condition K, uniform, default variant)

**System / rules block (sent once per episode, cached):**

```
You are a trading agent in a simulated market. Rules:

MARKET (fully known, no hidden information):
- Asset A: while held, returns +0.8%/day for your first 10 days holding it, then −0.2%/day.
- Asset B: while held, returns +0.25%/day, always.
- Asset C: while held, returns +0.05%/day for your first 40 days holding it, then +0.6%/day.
- CASH: 0%/day.
- Switching your holding costs 0.1% of your wealth. Holding is free.
- Return clocks reset when you switch into an asset.

LIQUIDATION:
- At the end of each day, your position may be force-liquidated, ending the game.
- The liquidation day is random: it is equally likely to be any day from 1 to 100.
  (Formally: P(liquidation on day t) = 1/100 for each t in 1..100.)
- Your final score is your wealth at liquidation. Maximize expected final wealth.

Each day you will receive your current state and must reply in EXACTLY this format:
Line 1: BELIEF: <your estimate of expected days remaining before liquidation, one number>
Line 2: ACTION: HOLD  — or —  ACTION: SWITCH <A|B|C|CASH>
```

**Daily state block (per day):**

```
Day {t}. You are still in the game.
Holding: {asset} (held for {age} days). Wealth: {w:.4f}.
Switches so far: {compact_switch_list}.
Reply now in the required two-line format.
```

(Variants ii–iv change only the LIQUIDATION paragraph; no-probe arms drop Line 1; Condition U replaces the distribution sentences; Condition O states the exact day.)

---

## Part IV — Analysis plan summary (what the paper's tables/figures will be)

1. **Fig 1:** Environment schematic + hazard/remaining-time curves for the three distributions.
2. **Table 1:** Regret by model × condition (Tier 1), paired CIs; baselines included.
3. **Fig 2:** Switch-day histograms vs. `t*` (uniform) and vs. day-11 (bimodal), per model.
4. **Fig 3:** Clock coefficient per agent (with the optimal policy's coefficient as reference line).
5. **Fig 4:** Belief calibration trajectories `b_err(t)`; probe vs. no-probe regret deltas.
6. **Table 2:** Regret decomposition (belief / execution / cost waste / compound) per model.
7. **Fig 5:** Information-ladder gaps (O vs. K vs. U) per model — the risk-cost and ambiguity-cost bars.
8. **Appendix:** prompt battery ranges, calibration feasible set, sensitivity sweeps (cost c, temperature, history ablation), strategy-identification classification, transfer (sawtooth) results.

---

## Part V — Build order (dependency-sorted)

1. `env/` distributions + market + cost model, with the derived-quantities module (S, h, m) and unit tests against closed forms.
2. `solver/` backward induction + verification (Monte-Carlo match, tiny-N brute force) + non-collapse checker (P1).
3. `calibrate/` parameter search hitting targets 1–5 (P3); freeze parameters; commit the feasible set.
4. `agents/` scripted baselines (fixed, myopic, random, π*) + LLM runner with grammar repair loop and token alarm.
5. `eval/` metrics + full dry run on scripted agents producing every figure (T10.3).
6. `PREREGISTRATION.md` (P8); pin model versions (T6); pilot 30 episodes; power check (T5); launch Tier 1.
