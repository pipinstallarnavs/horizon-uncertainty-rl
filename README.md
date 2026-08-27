# Random-Horizon Trading: Testing Temporal Reasoning in LLM Agents

## 1. Motivation

LLM trading agents are typically evaluated with a fixed, known time horizon (explicitly stated, or implied by daily-rebalancing conventions). Humans do not invest this way: we hold a *distribution of beliefs* over how long a position will last, and we update those beliefs as time passes.

This project builds an environment where an LLM agent must trade under **horizon uncertainty**: it enters a market, trades freely, but is force-liquidated at a **random, unknown time T**. The optimal strategy requires reasoning about elapsed time — "given that I have survived this long, how much time do I likely have left, and what should I hold for that remaining time?"

**Research question (framed as a question, not a foregone conclusion):**
Can LLM agents perform time-inhomogeneous sequential decision-making under horizon uncertainty — and if not, what is the specific signature of their failure (temporal rigidity, missed belief updates, mistimed trades)?

Either outcome is informative: success shows LLMs can do Bayesian temporal reasoning when framed correctly; failure characterizes a concrete, measurable limitation.

## 2. Problem Definition

### 2.1 Episode structure

- Discrete time, `t = 1, 2, ..., N` with `N = 100` days.
- The agent starts with wealth `W_0 = 1.0` (all cash).
- Each day, while the episode is alive, the agent observes the state and chooses an action.
- After the action, the exit event fires with hazard probability `h(t)`. If it fires, the portfolio is force-liquidated at that day's prices and the episode ends with terminal wealth `W_T`.
- If the exit never fires before day `N`, forced liquidation happens at `t = N` (the exit distributions are constructed so `P(T <= N) = 1`).

### 2.2 Objective

Maximize expected terminal wealth `E[W_T]`, where the expectation is over both market randomness and the random exit time `T`.

(A risk-averse variant using `E[log W_T]` is a planned extension; start with expected wealth for simplicity of the optimal-policy computation.)

## 3. Design Decisions and the Problems They Mitigate

Each subsection names a failure mode identified during design, and the mitigation adopted.

### 3.1 Problem: memoryless exit collapses into ordinary discounting

If the exit time is geometric (constant hazard `h(t) = p`), the objective reduces exactly to a standard discounted objective with `γ = 1 − p`:

```
E[ Σ_{t=1..T} r_t ] = Σ_t r_t (1−p)^{t−1} = Σ_t γ^{t−1} r_t
```

The optimal policy is stationary — it never depends on elapsed time — so the experiment would test nothing temporal.

**Mitigation: use only non-memoryless exit distributions.** Two conditions:

1. **Uniform exit**: `T ~ Uniform{1, ..., 100}`.
   - Conditional remaining time after surviving day `t` is `Uniform{1, ..., N−t}`, mean `(N−t+1)/2`.
   - Hazard rises smoothly: `h(t) = 1/(N−t+1)`, from 1% to 100%.
   - Tests *gradual* belief updating.

2. **Bimodal exit**: `T = 10` w.p. 0.5, else `Uniform{91,...,100}`.
   - Surviving day 10 fully resolves the mode in a single step: expected remaining time jumps `43.75 → 85.5` days.
   - Hazard is zero everywhere except day 10 (`h(10) = 0.5`, a coin flip) and the late window.
   - Creates a **bright-line event at day 11**: a rational agent should visibly change strategy. A single interpretable behavioral checkpoint, and with these parameters the optimal action changes on **75% of reachable states** at that boundary.

   **The early mode must be concentrated (revised 2026-08-27).** This condition was originally specified with the early mass spread over `Uniform{1,...,10}`. Measured against the exact optimum, that version has **no bright line at all**: spreading the mass resolves the posterior gradually, so by day 10 the agent already knows it is in the late mode, expected remaining time creeps from 50.5 to 85.5 rather than jumping, and the optimal action changes on **0%** of reachable states at day 11. Hypothesis H3 and the event-timing metric would have had no ground truth to score against. Concentrating the mass restores the intended discontinuity.

Geometric exit is retained only as a *control condition* to verify that agents behave stationarily when they should (a sanity check, not a headline experiment).

### 3.2 Problem: attribution — if the LLM fails, was it temporal reasoning or something else?

A rich environment (hidden dynamics, fractional sizing, many assets) makes failure uninterpretable: poor performance could come from bad market prediction, arithmetic slips in position sizing, or context-length issues — not the temporal reasoning we want to isolate.

**Mitigations:**

- **Fully disclosed market dynamics.** The exact return process of every asset is stated in the agent's prompt. There is nothing to predict about the market. The *only* hidden variable is `T` (and even its distribution is disclosed in the main condition — see 3.5). Any suboptimality is therefore attributable to temporal reasoning.
- **Trivially simple market.** Deterministic or low-noise synthetic returns (see Section 4). No regime switching, no hidden state.
- **Discrete action space.** Each day the agent holds exactly one of `{Asset A, Asset B, Asset C, Cash}` — 100% allocation, no fractional sizing. This removes arithmetic as a confound and makes the policy space small enough for exact solution.
- **Exact optimal baseline.** Because state and action spaces are small and dynamics are known, the true optimal policy is computed by backward induction (Section 6). The headline metric is **regret**, not raw PnL. "Lost money" is uninterpretable; "3× the regret of the optimum, concentrated in late-episode over-holding of the long-horizon asset" is a finding.

### 3.3 Problem: horizon uncertainty is irrelevant if one asset dominates at all horizons

If a single asset is best regardless of holding length, the exit distribution doesn't matter and the experiment is vacuous.

**Mitigation: assets engineered so the best choice depends on remaining time.** See Section 4. The design guarantees a computable **crossover day `t*`** at which the optimal holding flips from the long-horizon asset to the short-horizon asset (under uniform exit). The location of the LLM's switch relative to `t*` is a sharp quantitative measure.

### 3.4 Problem: free rebalancing masks temporal reasoning

With zero transaction costs, an agent can churn daily; hyperactive trading hides whether trades are *timed to informational events* or are just noise.

**Mitigation: small transaction cost.** Switching holdings costs `c = 0.1%` of wealth per switch (holding the same position is free). This creates a real tension — adapting to the shifting horizon costs money — so the *timing* of trades becomes meaningful. Under the bimodal condition, the optimal policy trades at day 11 (mode resolution); whether the LLM concentrates its trades there is a crisp readout.

### 3.5 Problem: conflating risk with ambiguity

"The agent doesn't know T" mixes two different unknowns: not knowing the *realization* of T (risk) vs. not knowing the *distribution* of T (Knightian ambiguity).

**Mitigation: separate conditions.**

- **Condition K (known distribution)** — main condition. The exit distribution is fully stated in the prompt. Tests pure temporal *reasoning*: can the agent do the survival update and act on it?
- **Condition U (unknown distribution)** — the prompt says only "you will be liquidated at some random time." Tests behavior under ambiguity.
- **Condition L (learnable distribution)** — extension. The agent plays repeated episodes with persistent memory of past exit times and can *infer* the hazard. Connects to self-evolving-agent research; deferred to Phase 2.

### 3.6 Problem: the mandatory-entry rule entangles two clocks

If the exit clock starts at the agent's chosen entry time, entry timing changes the exit distribution the agent faces — an interesting but confounding second decision.

**Mitigation: for Phase 1, the exit clock starts at episode start (day 1), and the agent is auto-entered on day 1 holding cash.** "When to first leave cash" remains a decision, but there is a single unambiguous clock. Entry-timing as a strategic variable (clock-starts-at-entry) is deferred to Phase 2.

### 3.7 Problem: confirmation-biased framing

The hypothesis "LLMs are rigid about time" is an intuition, not an established fact.

**Mitigation: symmetric design and reporting.** The experiment is designed so that *both* outcomes are publishable: metrics measure the degree and structure of temporal adaptation, not merely pass/fail. Prompt variants (explicit hazard stated numerically vs. described verbally vs. omitted) probe *when* temporal reasoning succeeds, not just whether.

## 4. Market Specification

Three risky assets plus cash, with cumulative return profiles chosen so the ranking depends on holding length. Returns are deterministic functions of holding duration in the base version (noise added as a robustness variant).

| Asset | Profile | Daily return schedule (provisional) |
|---|---|---|
| A — "Sprinter" | Momentum then decay | +0.8%/day for the first 10 days held, then −0.2%/day |
| B — "Compounder" | Slow steady growth | +0.25%/day, flat forever |
| C — "Late bloomer" | Weak early, strong late | +0.05%/day for the first 40 days held, then +0.6%/day |
| Cash | Safe | 0%/day |

### The no-re-entry rule (added 2026-08-27)

**Once the agent leaves a risky asset, it may never hold that asset again.** Cash is
exempt and may be entered and left freely. The rule is stated verbatim in the agent's
prompt.

This rule is what makes a decaying asset usable at all. Return clocks reset on switching
(below), so an asset paying more when freshly held can be farmed by bouncing out and back.
Measured against the exact optimum, the original parameters produced exactly that:

```
d1-10:A   d11:B   d12-21:A   d22:B   d23-32:A   ...      18 switches, C never held
```

Worse, the resulting policy was **identical at every `t`** — 0% of reachable states changed
action on any day — which is precisely the degeneracy §3.1 exists to prevent. The
environment would have tested nothing temporal.

Raising the switch cost does not fix it: the cost required to deter the cycle (>2.6%, some
26× the designed 0.1%) also deters the single genuine adaptation the experiment is about.
Forbidding re-entry removes the exploit by construction while leaving one-way adaptation
free, and it restores the original three asset profiles above.

With the rule in place the optimum is a single clean switch, and the two headline
conditions diverge in an interpretable way:

| Exit | Optimal path | `V*` |
|---|---|---|
| uniform | hold A days 1–10, then **B** for the rest | 1.1964 |
| bimodal | hold A days 1–10, then **C** for the rest | 1.2648 |

Both ride the Sprinter through its boost window; they differ in what they hold afterwards.
Under uniform exit remaining time is shrinking, so the steady compounder wins; under
bimodal exit, surviving day 10 buys 80 risk-free days, so the late bloomer wins. That
divergence is the behavioural signature the study is built to detect.

Notes:

- Returns depend on **how long the asset has been held** (per-position clock), which is what makes holding-length reasoning necessary. This is the intended design, and it is disclosed to the agent verbatim. Clocks reset on entry, which is why the no-re-entry rule above is required.
- Parameters above are provisional; final values are set by requiring (i) a crossover day `t*` in the interior of `[1, N]` under uniform exit, and (ii) meaningful value differences between optimal and naive policies (target: optimal beats best fixed-asset baseline by ≥ 5% expected terminal wealth). Calibrated numerically before any LLM runs.
- **Noise variant**: add i.i.d. Gaussian noise (σ = 0.5%/day) to each asset's daily return, keeping expectations as above. Verifies findings are not artifacts of determinism.

## 5. Agent Interface

### 5.1 Observation (per day)

The daily prompt contains:

- Current day `t` and the fact that the episode is alive.
- Full market rules (asset return schedules, transaction cost) — repeated every episode, stated once per episode.
- Exit-time information per condition (K: full distribution; U: existence only).
- Current holding, current wealth, days the current position has been held.
- Trade history for this episode (compact table).

Deliberately **included**: elapsed time is always visible. We are testing whether the agent *uses* the clock, not whether it can *find* it.

### 5.2 Action

One token-parseable choice per day: `HOLD` or `SWITCH <A|B|C|CASH>`.

### 5.3 Elicited beliefs (probe, logged but not scored)

Before each action, the agent is asked: "In one line, state your current estimate of expected remaining days." This gives a direct calibration readout — comparing stated beliefs to the true conditional expectation `E[T − t | T > t]` separates *belief formation* failures from *belief-to-action* failures.

## 6. Ground Truth: Optimal Policy via Backward Induction

State: `(t, current holding, holding age)`. Wealth factors out because per-day returns are multiplicative and independent of wealth; the value function is computed in log-wealth or as a wealth multiplier.

Recursion (sketch):

```
V(t, holding, age) =
  h(t) * 1.0                                # liquidated today; multiplier 1 going forward
  + (1 − h(t)) * max over actions a of:
        growth(a, age') * V(t+1, holding(a), age')
        − switch_cost_if_applicable(a)
```

Solved backward from `t = N` (where `h(N) = 1`). State space is ~100 days × 4 holdings × 100 ages ≈ 40k states — trivial to solve exactly.

Outputs:

- `V*`: optimal expected terminal wealth.
- `π*(t, holding, age)`: optimal action table, including the crossover day `t*` and (bimodal) the day-11 trade.

## 7. Baselines

| Baseline | Purpose |
|---|---|
| Optimal policy `π*` | Upper bound; regret denominator |
| Fixed-asset (always A / always B / always C / always cash) | Do you even need to trade? |
| Fixed-horizon LLM: "assume you will be liquidated on day 5" / "day 50" / "day 95" | The killer comparison: mismatched-horizon agents vs. the uncertainty-aware agent |
| Myopic LLM: prompt with no exit information and no elapsed-time emphasis | Ablation for prompt framing |
| Random policy | Sanity floor |
| Geometric-exit control runs | Verifies agents are (correctly) stationary when the problem is stationary |

## 8. Metrics

1. **Regret**: `V* − E[W_T^agent]`, averaged over ≥ 200 episodes per condition (exit-time realizations resampled; fixed seeds for reproducibility).
2. **Clock coefficient**: regression of portfolio composition on elapsed time, controlling for market state. Optimal policy has a strong time coefficient; a temporally rigid agent has ≈ 0.
3. **Crossover alignment** (uniform condition): distribution of the agent's B→A (or C→A) switch day vs. the optimal `t*`.
4. **Event-timing score** (bimodal condition): fraction of trades concentrated at the day-11 information event; probability the agent changes strategy in the window [11, 15] vs. base rate.
5. **Belief calibration**: error between stated expected-remaining-days and the true `E[T − t | T > t]`, tracked over `t`.
6. **De-risking signature**: cash/stable-asset share as a function of hazard `h(t)` — does the agent de-risk as hazard rises?
7. **Behavioral decomposition**: regret split into (a) wrong beliefs (calibration error), (b) right beliefs / wrong actions, (c) transaction-cost waste.

## 9. Experimental Grid (Phase 1)

- Exit distributions: uniform, bimodal (+ geometric control).
- Information conditions: K (known distribution), U (unknown).
- Market: deterministic base, noise variant.
- Models: at least 2 frontier LLMs + 1 open-weights model (same prompts).
- Prompt variants (Condition K only): distribution as a table of probabilities vs. verbal description.

Estimated cost: pure API inference; no GPU required. ~100 days × ~200 episodes × grid cells; use compact prompts and cache the fixed rule text.

## 10. Phase 2 Extensions (explicitly out of scope for Phase 1)

- **Condition L**: repeated episodes with persistent memory (e.g., a memory framework such as Letta) — can the agent *learn* the hazard from experience? Bridge to self-evolving-agent research.
- **Entry-timing clocks**: exit clock starts at agent-chosen entry.
- **Risk-averse objective**: `E[log W_T]`; fractional position sizing.
- **Recursive/RLM-style agents**: does recursive decomposition of the temporal problem help?
- **Fine-tuning**: can a small open model be trained (SFT on `π*` traces, or RL) to close the regret gap, and does it transfer across exit distributions?

## 11. Threats to Validity (acknowledged upfront)

- **Per-holding return clocks are stylized.** Real assets don't reward holding duration per se. This is intentional: the design isolates temporal reasoning. A follow-up with historical data trades control for realism.
- **Disclosure is unrealistic.** Real traders don't know return processes. Again intentional — Condition U and Phase 2 walk toward realism.
- **Prompt sensitivity.** Results may hinge on phrasing; mitigated by prompt variants and reporting across them, not cherry-picking.
- **Single-episode context length.** 100 days of history may strain context; mitigated by compact state summaries, and history-truncation ablations.

## 12. Deliverables

1. `env/` — environment implementation (exit distributions, market, cost model).
2. `solver/` — backward-induction optimal policy + calibration script for asset parameters.
3. `agents/` — LLM runner (any API model), scripted baselines.
4. `eval/` — episode runner, metrics, plots (regret, clock coefficient, switch-day histograms, calibration curves).
5. Report/paper draft structured around the metrics in Section 8.
