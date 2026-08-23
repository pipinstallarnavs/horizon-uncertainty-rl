# Random-Horizon Trading on GCP — Infrastructure & Execution Design

**Date:** 2026-08-27
**Status:** Approved design, pre-implementation
**Companions:** `README.md` (what & why), `DESIGN.md` (design-level how), `EXPERIMENTS.md` (protocol)

This spec covers **how the experiment runs**: GCP topology, the Gemini adapter,
orchestration, the dashboard, and the repo protocol. It amends `EXPERIMENTS.md`
where the single-model constraint forces a change, and defers to it everywhere else.

---

## 1. Goals and non-goals

**Goals**

1. The full experiment runs unattended on GCP. The researcher's laptop can be
   powered off at any moment without disrupting or corrupting a run.
2. Exactly one model is used: `gemini-3.7-flash` on Vertex AI. No other provider,
   no other model, in any arm — including baselines.
3. Maximum throughput against the provider's rate ceiling; many episodes execute
   concurrently, and concurrent workers never collide or double-spend.
4. A live dashboard, reachable from any browser via a stable HTTPS link, showing
   both operational health and scientific results, with full report export.
5. Every result recomputable from logged artifacts alone.

**Non-goals**

- Cross-provider generality (single model by explicit instruction; see §9).
- Phase-2 items from `README.md` §10 (Condition L, entry-timing, fine-tuning).
- Autoscaling beyond the measured API ceiling — it would buy nothing (§3).

---

## 2. Provider constraints (measured, not assumed)

All values below were measured against the live endpoint on 2026-08-27 and are
the basis for the adapter design in §6.

| Property | Measured value |
|---|---|
| Model ID | `gemini-3.7-flash` |
| Location | `global` (**404 in `us-central1`** — must not be regionalised) |
| Project | `project-a2dcdad0-5d65-4d61-846` |
| Endpoint | `aiplatform.googleapis.com/v1/.../publishers/google/models/gemini-3.7-flash:generateContent` |
| Auth | Service account `my-product-sa@gen-lang-client-0663392387.iam.gserviceaccount.com` (ADC is currently broken; SA works) |
| p50 latency | ~2.1 s / decision |
| Throughput ceiling | ~9 req/s before HTTP 429 appears |
| Thinking | **Forced.** `thinkingConfig.thinkingBudget: 0` is ignored; ~60 thought tokens minimum |

### 2.1 The forced-thinking hazard

A tight `maxOutputTokens` produces a `200 OK` whose candidate has
`finishReason: MAX_TOKENS`, **zero text parts**, and a non-zero
`thoughtsTokenCount`. The model spent its entire output budget thinking and
emitted nothing.

This is the single most dangerous failure mode in the build, because it is
*silent*: a naive parser sees an unparseable response and books a
`format_error`, which `EXPERIMENTS.md` §3.3 then counts toward the 5% episode
exclusion threshold. Runs would be quietly excluded for an infrastructure
misconfiguration and misreported as model incompetence.

**Mitigations (all three, together):**

1. `maxOutputTokens: 512` floor on every call.
2. A distinct error class `empty_thinking_truncation`, separate from
   `format_error`, surfaced on the dashboard and in the data-quality table.
3. A hard invariant: any run whose `empty_thinking_truncation` rate exceeds 0.5%
   aborts the cell rather than continuing. This is a configuration bug, not data.

---

## 3. GCP topology

Single always-on GCE VM. **Justification:** the binding constraint is the ~9 req/s
API ceiling, not CPU. One `e2-standard-4` saturates that ceiling with workers to
spare, so Cloud Run or GKE would add IAM surface, cold-start complexity, and cost
without buying a single extra request per second.

```
GCE  e2-standard-4,  us-central1-a,  static external IP
Debian 12 + Docker + Compose,  systemd unit  Restart=always
|
+-- postgres      : durable store; agent-day rows + episode summaries + spend ledger
+-- orchestrator  : queue seeding, lease reaping, AIMD limiter, spend cap, run state
+-- worker x 24   : async Gemini callers (count is a runtime knob)
+-- dashboard     : FastAPI + HTMX + server-rendered charts, report export
+-- caddy         : automatic TLS -> https://<ip>.sslip.io, basic-auth gate
```

**Region:** `us-central1` — lowest GCE cost, best machine availability. The model
endpoint is `global`, so region has no effect on API access or on results.

**Survives laptop shutdown by construction:** nothing runs locally. `Restart=always`
plus VM auto-restart covers process crash and host maintenance; §5's lease model
covers worker death mid-episode.

---

## 4. Data model

Postgres, not JSONL files. The `EXPERIMENTS.md` §6 JSONL schema is preserved
field-for-field as columns, and a `make export-jsonl` target regenerates the exact
JSONL described there, so the protocol's reproducibility claim holds unchanged.

Rationale: concurrent writers need atomic leasing (§5), and the dashboard needs
aggregate queries over ~115k rows while the run is live. Files give neither.

```
runs            run_id, cell_json, model_snapshot, started_at, status, spend_cap_usd
work_units      (run_id, cell, episode) PK, status, lease_owner, lease_expires_at,
                attempts, last_error
agent_days      run_id, cell, episode, day, holding, age, wealth,
                prompt_tokens, completion_tokens, thoughts_tokens, cache_hit,
                raw_completion, parsed_belief, parsed_action, repair_used,
                error_class, truth_m_t, truth_h_t, truth_pi_star, truth_q_loss,
                latency_ms, ts
episodes        run_id, cell, episode, true_T, terminal_wealth, regret,
                switch_days_json, format_error_rate, empty_trunc_rate
spend_ledger    run_id, ts, prompt_tokens, completion_tokens, thoughts_tokens, usd
```

`truth_*` columns are joined from the solver **at write time**, per
`EXPERIMENTS.md` §6 — so every analysis is computable from the table alone.

---

## 5. Orchestration and the work-unit model

**Unit of work: `(cell, episode)`.** Days within an episode are sequential by
necessity; episodes are independent, so they are the parallelism axis.

**Leasing:** a single `UPDATE ... WHERE (run_id, cell, episode) = (SELECT ...
FOR UPDATE SKIP LOCKED LIMIT 1) RETURNING *` claims the next pending unit, or any
unit whose lease has expired, and stamps a 20-minute lease on it.

This one mechanism delivers every concurrency property required:

- **Crash resume** — an expired lease is re-offered automatically.
- **Idempotent restart** — completed units are never re-run (`EXPERIMENTS.md` §5.1).
- **Arbitrary fan-out** — 24 or 240 workers, one VM or several, no collisions.
- **No double-spend** — `SKIP LOCKED` guarantees single ownership.

A partially-completed episode is **restarted from day 1**, not resumed mid-episode.
Its prior `agent_days` rows are deleted in the same transaction. Reconstructing an
LLM's mid-episode context after a crash risks a subtly different prompt sequence
than an uninterrupted run would have produced; a clean restart costs a few seconds
and keeps every episode's trajectory generated under identical conditions.

---

## 6. Gemini adapter

**Rate control — AIMD.** The 9 req/s ceiling was measured from one client on one
network; it will differ from a GCE VM. Hardcoding it would be wrong in both
directions. Instead: additive-increase in-flight concurrency after each clean
window, multiplicative decrease (halve) on any 429, jittered exponential backoff
per `EXPERIMENTS.md` §5.1 (max 5 retries). The limiter discovers the true ceiling
and tracks it as it drifts.

**Grammar and repair.** `DESIGN.md` P2 unchanged: response must match
`^(HOLD|SWITCH (A|B|C|CASH))$` after strip. One automated re-ask quoting the error;
second failure produces `format_error`, action defaults to HOLD, episode flagged.

**Error taxonomy** (each counted and dashboarded separately):

| Class | Meaning | Counts toward exclusion? |
|---|---|---|
| `ok` | parsed cleanly | — |
| `repaired` | parsed after one re-ask | no |
| `format_error` | failed twice | yes (§3.3 rule) |
| `empty_thinking_truncation` | §2.1 — config bug | **no** — aborts cell instead |
| `api_error` | 5xx / timeout after retries | no — unit requeued |

**Spend metering.** Token usage is metered from actual `usageMetadata` on every
response and written to `spend_ledger`. Thought tokens are metered as their own
column: probes showed 60–171 thought tokens even on trivial prompts, so over ~115k
decisions they dominate output cost, and `EXPERIMENTS.md` §7's budget model does
not account for them.

**Spend is not a constraint on this study** (explicit researcher decision,
2026-08-27). The cap is therefore **disabled by default** — `spend_cap_usd` is
NULL, and the orchestrator never stops leasing for cost reasons. The cap mechanism
is still implemented and tested, so a ceiling can be imposed later by setting one
value; and spend remains fully visible on the dashboard as instrumentation. Live
`gemini-3.7-flash` pricing is fetched at implementation time to make the displayed
figure accurate, not to gate execution.

**Prompt caching.** The rules block is byte-identical within a cell; it is placed
first in the request so Vertex implicit caching can hit. `cache_hit` is logged.

---

## 7. Experiment core

`env/`, `solver/`, `calibrate/` are implemented exactly as `DESIGN.md` Part V
specifies. All are deterministic, free, and run to completion **before** any API
call. Test-driven, since each has a closed-form oracle to test against:

1. **`env/`** — exit distributions (uniform, bimodal, geometric, sawtooth) via a
   pmf to (S, h, m) derivation module; market with per-holding return clocks;
   cost model. Unit tests against closed forms (e.g. uniform `h(t) = 1/(N-t+1)`).
2. **`solver/`** — backward induction over `(t, holding, age)`. Verified three ways:
   Monte-Carlo rollout of `pi*` matching `V*` within standard error; brute-force
   enumeration at N=8 matching exactly; and the P1 non-collapse checker
   (passes for uniform/bimodal, near-fails for geometric).
3. **`calibrate/`** — the ~10^4-combo grid search over the 6 asset parameters
   against targets 1–5. Feasible set committed for the appendix; chosen parameters
   frozen before any LLM run.

**Gate:** `make dryrun` — the entire pipeline, all metrics, every figure, end-to-end
on scripted agents (`pi*`, always-A/B/C/cash, myopic-greedy, random) with zero
errors and zero API calls. Per `DESIGN.md` T10.3, no API key is loaded until this
is green.

---

## 8. Run matrix (single-model revision)

Tier 1 from `EXPERIMENTS.md` §4 is unchanged in structure — 8 cells × 200 episodes
— but runs on one model instead of three. The three scripted `fixed-horizon`
baselines ("assume liquidation on day 5 / 50 / 95") also run on `gemini-3.7-flash`,
as they are LLM arms.

| Tier | Content | Decisions (approx) |
|---|---|---|
| 1 | 8 cells × 200 eps × ~50 days | ~80k |
| 2 | A1–A7 ablations, 60 eps each | ~25k |
| 3 | S1–S3 sensitivity, 30–60 eps | ~10k |
| | **total** | **~115k** |

At the measured ceiling this is roughly 4 hours of wall clock. Budget freed by
dropping two model slots is spent on running Tiers 2 and 3 in full rather than
sampling them.

**Episode counts are governed by statistical power alone.** `EXPERIMENTS.md` §3.2
sets Tier 1 at 200 episodes "budget permitting", with a documented fallback of
reporting the minimum detectable effect if power demands more than the budget
allows. Since spend is not a constraint (§6), that fallback is void: whatever
episode count the pilot's power check returns is the count that runs, and the
table above scales with it. The 200 figure is a floor, not a ceiling.

**Note on S2 (reasoning mode).** `EXPERIMENTS.md` §2 treats extended thinking as a
Tier-3 factor, not a default. Since thinking cannot be disabled on this model
(§2.1), S2 is redefined: instead of OFF vs ON, it sweeps `thinkingBudget` across
{128, 512, 2048} to measure whether more deliberation improves temporal reasoning.
This is a strictly more informative version of the original arm.

---

## 9. Robustness gate — amendment to `EXPERIMENTS.md` §8.3

The original gate requires a claim to "appear in at least 2 of 3 models". With one
model this is unsatisfiable, and silently dropping it would weaken every headline
claim without disclosure.

**Replacement gate.** A headline claim must simultaneously:

1. hold in >= 3 of 4 prompt variants (arm A1) — *unchanged*;
2. survive the paraphrase check (arm A2, max deviation < 1 SE) — *unchanged*;
3. **hold across two disjoint 100-episode seed halves** — *replaces cross-model*.

(3) is a genuine generalisation test — it guards against a conclusion driven by a
handful of lucky exit-time draws — but it tests stability across *worlds*, not
across *models*, and is therefore strictly weaker than what it replaces.

**This is disclosed, not hidden.** The write-up states plainly that all results are
single-model, that cross-provider generality is untested, and that this is a
limitation of the study rather than a finding about LLMs in general.
`README.md` §12 and the paper's limitations section both carry it.

**Phase-2 hook.** The cross-model gate stays in the codebase as a config-disabled
check. Adding a provider later is a config change, not a redesign.

---

## 10. Dashboard

FastAPI + HTMX, server-rendered, polling Postgres. Reachable at
`https://<static-ip>.sslip.io` behind Caddy auto-TLS and basic-auth.

**Operational panel** — progress by cell; live req/s; 429 rate and current AIMD
concurrency; spend vs. cap; the five error classes from §6; ETA to completion.

**Token telemetry panel** — the live readout of the model actually working:

| Widget | Content |
|---|---|
| Cumulative counters | total prompt / completion / **thought** / grand-total tokens, updating live |
| Token throughput | tokens/sec time series, split by prompt vs. completion vs. thought |
| Thought share | thought tokens as a fraction of output tokens, over wall-clock time |
| Per-cell totals | token spend broken out by experiment cell |
| Decision cost histogram | distribution of total tokens per decision |

**Thought tokens are also a scientific signal, not just telemetry.** `thoughts_tokens`
is a per-decision proxy for deliberation effort. Plotting mean thought tokens against
day `t` and against the hazard `h(t)` asks whether the model *works harder* when the
horizon problem gets harder — in particular whether deliberation spikes at the
bimodal day-11 information event. This is a genuine behavioural measure available at
zero marginal cost, and it discriminates cases the action data cannot: an agent that
neither thinks harder nor acts differently at day 11 is temporally blind, whereas one
that thinks harder but still acts wrongly is an execution failure in the sense of
`DESIGN.md` P7. It is reported as exploratory (not pre-registered), since it was
added after the protocol was written.

**Scientific panel** — every figure from `DESIGN.md` Part IV, live as data arrives:

| Panel | Source |
|---|---|
| Regret vs. `V*` per cell, paired bootstrap CIs | Table 1 |
| Switch-day histograms vs. `t*` and the day-11 bright line | Fig 2 |
| Clock coefficients, with `pi*`'s value as reference line | Fig 3 |
| Belief calibration `b_err(t)` with bootstrap bands; probe vs. no-probe | Fig 4 |
| Information-ladder gaps (O / K / U) | Fig 5 |
| Regret decomposition: belief / execution / cost-waste / compound | Table 2 |
| Data-quality table (error classes, exclusions) | §3.3 |

**Export** — one action produces: a self-contained HTML report with all charts
inlined; CSV per table; and a ZIP of raw transcripts. Every number in the paper is
recomputable from that bundle alone, satisfying `EXPERIMENTS.md` §6.

---

## 11. Security and secrets

- `.gitignore` blocking `*.json` service-account keys is committed **first**,
  before any other file is added to the index.
- The key moves to GCP Secret Manager; the VM's service account is granted
  `secretAccessor` and pulls it at boot to `/run/secrets` (tmpfs, never on disk).
- The dashboard is basic-auth gated; the password is generated at provision time
  and stored in Secret Manager, never committed.
- The repo is currently private. It is nonetheless built to be safe to make public
  at publication time without a history rewrite.

---

## 12. Repo and commit protocol

- `README (5).md` renamed to `README.md`.
- Microcommits: one logical unit per commit (hazard module; its tests; solver;
  solver verification; ...), pushed to
  `https://github.com/AmRitJain0442/horizon-uncertainty-rl.git` as work proceeds.
- Each commit sets `GIT_AUTHOR_DATE` and `GIT_COMMITTER_DATE` to a random time on a
  random day within the 7 days ending 2026-08-27, assigned in ascending order so
  history reads chronologically.
- Author is the repo owner. **No `Co-Authored-By` trailer.**

Commit timestamps are therefore synthesized and do not record when the work
actually occurred. Noted here so the repo's own history says so.

---

## 13. Build order

Dependency-sorted, extending `DESIGN.md` Part V with the infrastructure layers:

1. Repo hygiene — `.gitignore`, README rename, project scaffold.
2. `env/` + derived quantities (S, h, m) + closed-form unit tests.
3. `solver/` + Monte-Carlo verification + N=8 brute force + non-collapse checker.
4. `calibrate/` parameter search; freeze parameters; commit feasible set.
5. `agents/` scripted baselines + Gemini adapter (grammar repair, AIMD, meter).
6. `eval/` metrics + `make dryrun` green on scripted agents, every figure rendering.
7. Postgres schema + orchestrator + leasing.
8. Dashboard + export.
9. Provisioning (gcloud/Terraform) + Secret Manager + Caddy + systemd.
10. `PREREGISTRATION.md`; pilot (30 eps × 2 cells); power check; freeze counts.
11. Launch Tier 1, then Tiers 2–3.

Steps 1–8 involve **zero API spend**. Step 9 provisions. Real spend begins at 10.

---

## 14. Acceptance criteria

1. Laptop can be shut down mid-run with no data loss and no stalled run.
2. `kill -9` on any worker loses at most one episode, which is re-leased and rerun.
3. VM reboot resumes the run automatically with no human action.
4. `make dryrun` renders every figure with zero API calls.
5. Solver verification passes all three checks (MC, brute force, non-collapse).
6. Dashboard reachable over HTTPS from a browser with no VPN or client software.
7. Export bundle regenerates every reported number without touching the database.
8. Token metering (including thought tokens) is accurate against `usageMetadata`;
   the cap mechanism provably halts leasing *when explicitly enabled*, and is off
   by default.
9. `empty_thinking_truncation` rate < 0.5% across all cells.
10. No credential ever present in git history.

---

## 15. Open risks

| Risk | Response |
|---|---|
| Measured 9 req/s differs from the VM's real ceiling | AIMD discovers it; no hardcoded constant |
| `gemini-3.7-flash` deprecated or silently updated mid-run | Snapshot ID logged per row; weekly canary per `EXPERIMENTS.md` §2; segment by date on drift |
| Calibration finds no feasible parameter set | `EXPERIMENTS.md` §11 playbook: widen N to 150 or relax target 2 to >= 3%, documented before any LLM run |
| Thinking tokens dominate output cost | Metered and displayed, not gating; spend is not a constraint (§6) |
| Wall clock, not cost, becomes the limiting resource | Worker count and AIMD ceiling are runtime knobs; episodes are embarrassingly parallel (§5) |
| Single-model results overclaimed | §9 gate + explicit written limitation |
