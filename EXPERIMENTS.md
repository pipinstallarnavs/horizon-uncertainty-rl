# Random-Horizon Trading — Experimental Setup & Protocol (v1)

**Companion to README.md (what & why) and DESIGN.md (how, at design level).** This document is the operational protocol: exactly what runs, in what order, with what seeds, models, budgets, logging, and analysis. It is written so that a second researcher could execute the entire experiment from this file alone.

---

## 1. Overview of the experimental factors

| Factor | Levels | Notes |
|---|---|---|
| Exit distribution | `uniform`, `bimodal`, `geometric` (control), `sawtooth` (transfer) | P1, T9 |
| Information condition | `K` (known dist), `U` (unknown), `O` (oracle day) | P5, P2 |
| Belief probe | `probe`, `noprobe` | P7, T7 |
| Prompt variant | `table`, `verbal`, `hazard`, `minimal` | T3 |
| History format | `none`, `compressed` (default), `full` | T4 |
| Market noise | `deterministic` (default), `noisy` (σ = 0.5%/day) | P2, robustness |
| Temperature | `0` (default), `0.7×3` (sensitivity) | T5 |
| Model | see §2 | |

A full factorial is intractable (T10). We run a **tiered design**: Tier 1 is the headline factorial over the factors that matter most; Tiers 2–3 vary one factor at a time from the Tier-1 default cell.

**Tier-1 default cell:** `compressed` history, `deterministic` market, temperature 0, `table+verbal` combined prompt (the K default from DESIGN.md Part III).

---

## 2. Models

| Slot | Model | Role | Pinning |
|---|---|---|---|
| M1 | Frontier model, provider 1 (e.g., latest Claude) | Headline | Exact snapshot ID recorded in every result row |
| M2 | Frontier model, provider 2 (e.g., latest GPT) | Cross-provider generality | Same |
| M3 | Open-weights 7–8B (e.g., Qwen3-8B), local, greedy decoding | Reproducibility anchor + capability floor | Weights SHA-256 recorded |
| M4 (optional) | Mid-tier model (e.g., a "mini/haiku" class) | Capability scaling hint | Same as M1/M2 |

Rules:
- All models receive byte-identical prompts (modulo provider message-format wrappers, which are logged).
- No system-prompt extras, no tools, no chain-of-thought scaffolds in Tier 1. A `+reasoning` arm (model's extended thinking enabled where available) is Tier 3 — reasoning-mode is treated as an experimental factor, not a default, since it changes the construct being measured.
- Weekly 20-episode canary per API model (T6); canary metric = Tier-1 uniform-K regret; drift > 2 SE → segment results by date and note in the changelog.

---

## 3. Episodes, seeds, and the common-random-numbers (CRN) block

### 3.1 Seed banks (frozen before any run, committed to the repo)

- `EXIT_SEEDS[dist][i]`, i = 1..200: 200 pre-drawn exit-time realizations per distribution. Drawn once with a fixed master seed (`20260814`), stored as plain CSV so they are inspectable.
- `NOISE_SEEDS[i]`, i = 1..200: market noise seeds for the noisy arm.
- Episode `i` of any agent, in any condition, uses `EXIT_SEEDS[dist][i]` (and `NOISE_SEEDS[i]` if noisy). **Every agent therefore faces the identical sequence of worlds**, making all between-agent comparisons paired (T5).

### 3.2 Episode counts

| Tier | Cell type | Episodes per cell | Rationale |
|---|---|---|---|
| Pilot | 2 cells (uniform-K, bimodal-K) × M1 | 30 | Variance estimation for the power check |
| Tier 1 | headline factorial (§4) | 200 | Target: detect Δregret = 2% of V* at 80% power (adjusted after pilot) |
| Tier 2 | one-factor-at-a-time ablations | 60 | Directional evidence only; CIs reported |
| Tier 3 | sensitivity arms | 30–60 | Appendix |

The pilot is run first; if the power calculation demands > 200 episodes per Tier-1 cell, we either raise the count (budget permitting) or report the minimum detectable effect alongside results — the decision rule is written down *before* the pilot (see §8).

### 3.3 Stopping and exclusion rules (pre-specified, T5)

- An episode is `format_failed` if > 5% of its days hit the second-level grammar failure (DESIGN.md P2). Such episodes are excluded from reasoning metrics, fully counted in the data-quality table, and never resampled.
- No other exclusions. No early stopping of cells based on observed results.

---

## 4. Run matrix

### Tier 1 — headline (per model M1–M3)

| Cell | Exit | Info | Probe | Episodes |
|---|---|---|---|---|
| T1-1 | uniform | K | probe | 200 |
| T1-2 | uniform | K | noprobe | 200 |
| T1-3 | uniform | U | probe | 200 |
| T1-4 | bimodal | K | probe | 200 |
| T1-5 | bimodal | K | noprobe | 200 |
| T1-6 | bimodal | U | probe | 200 |
| T1-7 | uniform | O | noprobe | 200 |
| T1-8 | bimodal | O | noprobe | 200 |

(Oracle cells run noprobe — a remaining-time belief question is degenerate when the day is disclosed.)

8 cells × 3 models × 200 episodes × ~100 days ≈ 480k LLM decisions per model-set. This is the budget driver; see §7.

### Tier 2 — ablations (M1 only unless noted, 60 episodes each, from the default cell T1-1 or T1-4 as base)

| Arm | Varies | Levels run |
|---|---|---|
| A1 prompt battery | prompt variant | `table`, `verbal`, `hazard`, `minimal` (× uniform-K and bimodal-K) |
| A2 paraphrases | surface wording | 3 auto-paraphrases of the winning variant |
| A3 history | history format | `none`, `full` |
| A4 noise | market noise | `noisy` |
| A5 geometric control | exit | `geometric` (K, probe) — expect near-stationary behavior |
| A6 transfer | exit | `sawtooth` (K, probe) — recitation vs. reasoning (T9) |
| A7 cost sweep | transaction cost | c ∈ {0, 0.05%, 0.3%} (scripted π* re-solved per c) |

### Tier 3 — sensitivity (M1, 30–60 episodes)

| Arm | Varies |
|---|---|
| S1 temperature | T = 0.7, 3 samples/decision; majority-vote and single-sample both logged |
| S2 reasoning mode | provider extended-thinking ON |
| S3 model scale | M4 on T1-1 and T1-4 |

### Scripted agents (free, run on the full Tier-1 matrix + every ablation)

`π*` (optimal), `always-A/B/C/cash`, `myopic-greedy`, `random`, and the three `fixed-horizon-LLM` baselines ("assume liquidation on day 5 / 50 / 95" — these are LLM runs, M1 only, 60 episodes each on uniform-K worlds).

---

## 5. Execution pipeline

### 5.1 Architecture

```
runner.py
  ├── loads: env config, seed banks, agent spec, cell spec
  ├── per episode: loop days → build prompt → call agent → parse/repair → step env
  ├── writes: one JSONL row per day (schema §6), one summary row per episode
  └── deterministic resume: (cell, model, episode) is the unit of work;
      completed units are skipped on restart (idempotent by design)
```

- Concurrency: episodes are independent → parallelize across episodes with a rate-limit-aware worker pool; days within an episode are sequential by necessity.
- Retries: transient API errors retried with exponential backoff (max 5); a day that exhausts retries aborts and requeues the episode from its last completed day (state is reconstructible from the log).
- Prompt caching: the rules block is identical across all calls within a cell — structured to hit provider prompt-cache; cache-hit rates logged.
- Dry-run gate (T10): the full pipeline, all metrics, and all figures must run end-to-end on scripted agents (`make dryrun`) with zero errors before any API key is loaded.

### 5.2 Order of execution

1. `make dryrun` (scripted agents through everything; every figure renders).
2. Pilot (30 eps × 2 cells × M1) → power check → freeze episode counts → tag repo `pilot-frozen`.
3. Tier 1, M3 first (local, free) → sanity-review metrics → then M1, M2 (canary before each launch).
4. Tier 2, Tier 3.
5. Analysis freeze: tag `data-final`; all subsequent analysis is read-only on logged artifacts.

---

## 6. Logging schema (JSONL, one row per agent-day)

```json
{
  "run_id": "t1-1_M1_ep013",
  "cell": {"exit": "uniform", "info": "K", "probe": true, "prompt": "default",
            "history": "compressed", "noise": false, "temp": 0},
  "model": {"name": "...", "snapshot": "...", "weights_sha": null},
  "episode": 13, "exit_seed": "...", "true_T": 63,
  "day": 27, "holding": "B", "age": 27, "wealth": 1.0662,
  "prompt_tokens": 512, "completion_tokens": 22, "cache_hit": true,
  "raw_completion": "...",
  "parsed": {"belief": 38.0, "action": "HOLD", "repair_used": false, "format_error": false},
  "truth": {"m_t": 37.0, "h_t": 0.0135, "pi_star_action": "HOLD",
             "q_loss_of_action": 0.0},
  "latency_ms": 840, "timestamp": "..."
}
```

- `truth.*` fields are joined from the solver at log time so every analysis is possible from the JSONL alone (T6.3).
- Episode-summary rows add: terminal wealth, regret, switch list, per-episode format-error rate.
- Raw transcripts retained verbatim; the repo's analysis code recomputes every reported number from these files (`make figures` is fully reproducible).

---

## 7. Budget model (computed before launch; numbers below are the template, filled at launch time)

Per-decision tokens (compressed history): ~500 prompt (mostly cached) + ~25 completion.
Per-episode decisions: expectation under uniform exit ≈ E[T] ≈ 50 days; bimodal ≈ 53; oracle/geometric similar. (Days after exit are not simulated — episodes end at T, halving naive cost estimates.)

```
Tier-1 per API model ≈ 8 cells × 200 eps × ~50 days ≈ 80k decisions
  → ≈ 40M prompt tokens (≥80% cache-discounted) + 2M completion tokens
Tier-2 + Tier-3 (M1) ≈ 35k decisions
```

Fill in provider prices at launch; hard budget cap enforced in the runner (spend counter; abort at cap). M3 is local and free — it is deliberately run first and fully, so the entire analysis is debugged on complete real data before frontier spend.

---

## 8. Pre-registered analysis plan

Committed as `PREREGISTRATION.md` before the pilot; summarized here.

### 8.1 Primary hypotheses (two-sided, symmetric — P8)

- **H1 (adaptation):** Tier-1 K-condition LLM regret < best fixed-asset baseline regret (paired test per model).
- **H2 (clock use):** clock coefficient (README §8.2) differs from 0 and its sign matches π*'s.
- **H3 (event timing, bimodal):** P(strategy change in days 11–15 | survived day 10) exceeds the agent's base-rate switch probability.

Each hypothesis has two pre-written interpretation templates (supports adaptation / supports rigidity); mixed results are reported per-model without pooling.

### 8.2 Statistical procedures

- All between-agent comparisons: paired on episode index (CRN), paired bootstrap (10k resamples) for CIs, paired permutation tests for p-values.
- Multiple comparisons: Holm–Bonferroni within each hypothesis family (family = one hypothesis × models).
- Regression for clock coefficient: agent's holding-class (short/long/safe, ordinal) ~ elapsed-time + market-state controls, per episode, coefficients aggregated across episodes with cluster-robust SEs (cluster = episode).
- Belief calibration: report `b_err(t)` curves with pointwise bootstrap bands; summary = area under `b_err(t)` normalized by the U-condition's theoretical-max error.
- Regret decomposition (DESIGN.md P7): shares reported with bootstrap CIs; days classified by the elicit–act–compare rules verbatim as pre-specified.

### 8.3 Robustness gates for claims

A headline claim must simultaneously: hold in ≥ 3 of 4 prompt variants (A1), survive the paraphrase check (A2), and appear in at least 2 of 3 models (else it is reported as model-specific). The geometric control must show near-zero clock coefficient for π* and, if an LLM shows a large clock coefficient *there*, its uniform/bimodal clock coefficients are flagged as potentially spurious time-attention rather than correct hazard tracking.

---

## 9. Deliverable figures/tables mapping (links Part IV of DESIGN.md to cells)

| Output | Source cells |
|---|---|
| Table 1 (regret × model × condition) | T1-1..8 + scripted |
| Fig 2 (switch-day vs t*, day-11) | T1-1, T1-4 |
| Fig 3 (clock coefficients) | T1-1, T1-4, A5 (control) |
| Fig 4 (calibration; probe effect) | T1-1 vs T1-2, T1-4 vs T1-5 |
| Fig 5 (information ladder O/K/U) | T1-1, T1-3, T1-7; T1-4, T1-6, T1-8 |
| Table 2 (regret decomposition) | probe cells |
| Appendix | A1–A7, S1–S3, data-quality table |

---

## 10. Timeline & effort estimate (single researcher)

| Week | Work |
|---|---|
| 1 | env + solver + tests (build order 1–2); calibration search (3) |
| 2 | agents + runner + logging (4); `make dryrun` green (5) |
| 3 | PREREGISTRATION.md; M3 full Tier 1 (local); pilot on M1; power check |
| 4 | M1 + M2 Tier 1 |
| 5 | Tiers 2–3; analysis freeze |
| 6 | Figures, decomposition analysis, write-up |

Slack: one full week of buffer is realistic for API friction and calibration surprises; treat this as a 7-week plan when communicating externally.

---

## 11. Failure playbook (decided now, not under pressure)

| Symptom | Pre-decided response |
|---|---|
| Calibration search finds no feasible parameter set | Widen N to 150 or relax target 2 to ≥ 3%; document the relaxation before any LLM run |
| Pilot shows format-error rate > 10% on a model | One round of grammar-prompt hardening (logged as prompt v2), full rerun of pilot; if still >10%, model is reported as format-limited and excluded from reasoning claims |
| M3 (8B) is at random-policy level everywhere | Keep it — it anchors the capability floor; add M4 to provide a mid-point |
| API canary drift mid-Tier-1 | Segment by date; if segments disagree materially, rerun the affected cell after pinning is confirmed |
| Budget cap hit mid-run | Priority order for completion: T1-1, T1-4 (core), then T1-3/T1-6 (ambiguity), then oracle cells; ablations are sacrificed first |
| Results are uniformly null (all models ≈ optimal) | This is the "LLMs can do it" branch of P8 — proceed to write-up with the scaffolding-gradient analysis (A1 minimal-vs-rich prompts) as the main story |
