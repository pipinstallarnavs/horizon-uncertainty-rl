"""The live dashboard.

The page shell renders once; a fragment at /live is polled and swapped in, so
charts never blink and scroll position survives an update. Everything degrades
to a readable page when the database is empty or unreachable, because the
dashboard has to come up before the first worker writes a row.
"""
import os
import tempfile

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse

from horizon.dashboard.charts import bar_chart, empty_chart, hazard_ribbon, line_chart
from horizon.dashboard.decisions import (
    costly_rows,
    episode_picker,
    episode_trace_table,
    trade_rows,
)
from horizon.dashboard.render import page
from horizon.env.distributions import DISTRIBUTIONS
from horizon.store.repository import Repository

app = FastAPI(title="Random-Horizon Trading")

RUN_ID = os.environ.get("HORIZON_RUN_ID", "default")
_repo = None


def repo() -> Repository:
    global _repo
    if _repo is None:
        _repo = Repository()
    return _repo


def _safe(fn, default):
    try:
        return fn()
    except Exception:
        return default


def _run_meta():
    """Cell, V* and horizon for the active run; sensible defaults if absent."""
    def load():
        with repo().cursor() as cur:
            cur.execute("SELECT cell, v_star, horizon FROM runs WHERE run_id = %s", (RUN_ID,))
            row = cur.fetchone()
        if not row:
            raise LookupError
        cell, v_star, horizon = row
        return dict(cell or {}), float(v_star or 0.0), int(horizon or 100)

    return _safe(load, ({"exit": "uniform", "info": "K"}, 0.0, 100))


@app.get("/health")
def health():
    return {"status": "ok", "run_id": RUN_ID}


@app.get("/api/summary")
def api_summary():
    r = repo()
    return JSONResponse(
        {
            "run_id": RUN_ID,
            "progress": _safe(lambda: r.progress(RUN_ID), {}),
            "tokens": _safe(lambda: r.token_totals(RUN_ID), {}),
            "errors": _safe(lambda: r.error_counts(RUN_ID), {}),
        }
    )


@app.get("/export")
def export():
    """Download the report bundle: HTML report, CSVs and summary JSON."""
    from horizon.eval.export import export_bundle, records_from_rows

    r = repo()
    rows = _safe(lambda: r.episode_records(RUN_ID), [])
    if not rows:
        return JSONResponse({"error": "no completed episodes yet"}, status_code=409)

    cell, v_star, horizon = _run_meta()
    dist = DISTRIBUTIONS[cell.get("exit", "uniform")](horizon)
    path = os.path.join(tempfile.gettempdir(), f"{RUN_ID}-bundle.zip")
    export_bundle(path, RUN_ID, cell, records_from_rows(rows), dist, v_star)
    return FileResponse(path, filename=f"{RUN_ID}-bundle.zip", media_type="application/zip")


@app.get("/live", response_class=PlainTextResponse)
def live():
    return PlainTextResponse(_body(), media_type="text/html")


@app.get("/decisions", response_class=PlainTextResponse)
def decisions(episode: int = None):
    return PlainTextResponse(_decision_log(episode), media_type="text/html")


@app.get("/", response_class=HTMLResponse)
def index():
    cell, _, horizon = _run_meta()
    dist = DISTRIBUTIONS.get(cell.get("exit", "uniform"), DISTRIBUTIONS["uniform"])(horizon)
    hazard = [dist.hazard(t) for t in range(1, horizon + 1)]
    centre = _safe(lambda: repo().active_day(RUN_ID), None)

    peak = max(range(len(hazard)), key=lambda i: hazard[i]) + 1
    note = (
        f'<span>Hazard h(t) &mdash; chance of liquidation on each day, given survival to it</span>'
        f'<span>peak day {peak} &middot; horizon {horizon} days'
        + (f' &middot; fleet at day {centre}' if centre else "")
        + "</span>"
    )
    return HTMLResponse(
        page(RUN_ID, cell, hazard_ribbon(hazard, centre), note, _body(), _decision_log())
    )


# --------------------------------------------------------------------------- #


def _body() -> str:
    r = repo()
    cell, v_star, horizon = _run_meta()
    progress = _safe(lambda: r.progress(RUN_ID), {})
    tokens = _safe(lambda: r.token_totals(RUN_ID), {})
    errors = _safe(lambda: r.error_counts(RUN_ID), {})
    series = _safe(lambda: r.token_timeseries(RUN_ID), [])
    thoughts = _safe(lambda: r.thought_tokens_by_day(RUN_ID), [])
    switches = _safe(lambda: r.switch_day_counts(RUN_ID, horizon), [])
    beliefs = _safe(lambda: r.belief_error_by_day(RUN_ID), [])
    outcome = _safe(lambda: r.outcome_summary(RUN_ID), {})

    dist = DISTRIBUTIONS.get(cell.get("exit", "uniform"), DISTRIBUTIONS["uniform"])(horizon)
    truth = [(t, dist.mean_remaining(t)) for t in range(1, horizon + 1)]

    done = progress.get("done", 0)
    leased = progress.get("leased", 0)
    pending = progress.get("pending", 0)
    total = done + leased + pending or 1

    decisions = sum(errors.values())
    output = tokens.get("completion_tokens", 0) + tokens.get("thought_tokens", 0)
    thought_share = (100.0 * tokens.get("thought_tokens", 0) / output) if output else 0.0
    clean = 100.0 * (errors.get("ok", 0) + errors.get("repaired", 0)) / decisions if decisions else 0.0

    readings = [
        ("episodes", f"{done}<span style='font-size:.5em;color:var(--muted)'>/{total}</span>",
         f"{leased} in flight", ""),
        ("decisions", f"{decisions:,}", f"{clean:.1f}% parsed cleanly", "accent-signal"),
        ("thinking", f"{thought_share:.0f}%", "of output tokens spent on thought", "accent-thought"),
        ("regret vs optimum",
         f"{outcome['regret']:+.4f}" if outcome.get("regret") is not None else "&mdash;",
         f"V* = {v_star:.4f}" if v_star else "awaiting episodes", ""),
        ("tokens", _human(tokens.get("total_tokens", 0)), "prompt + output + thought", ""),
        ("metered spend", f"${tokens.get('usd', 0.0):,.2f}", "uncapped, instrumentation only", ""),
    ]

    return f"""
    <section class="readings">{"".join(_reading(*x) for x in readings)}</section>

    {_rail(done, leased, pending, total)}

    <div class="grid2">
      <section class="panel">
        <h2>Token throughput</h2>
        <p class="why">The model working, live. Thought tokens bill as output and
        cannot be switched off on this model, so they are tracked separately.</p>
        {_token_chart(series)}
      </section>

      <section class="panel">
        <h2>Deliberation by day</h2>
        <p class="why">Mean thought tokens per decision against the episode day.
        Whether the model thinks harder as the horizon problem sharpens is a
        behavioural signal, not just telemetry.</p>
        {line_chart([("thought tokens", "--thought", [(int(d), float(m or 0)) for d, m, _ in thoughts])],
                    x_label="episode day", fill=True) if thoughts
         else empty_chart(680, 210, "waiting for the first completed episode")}
      </section>
    </div>

    <div class="grid2">
      <section class="panel">
        <h2>When the agent traded</h2>
        <p class="why">Switch days across every completed episode. The optimal
        policy switches once, on day {_crossover(cell)} &mdash; a run that clusters there is
        timing its trades to the horizon rather than churning.</p>
        {bar_chart(switches, x_label="episode day", markers=(_crossover(cell),),
                   highlight=_crossover(cell)) if switches
         else empty_chart(680, 210, "waiting for the first completed episode")}
      </section>

      <section class="panel">
        <h2>Belief calibration</h2>
        <p class="why">The agent states its expected days remaining each day. Compared
        against the true conditional mean, this separates forming the wrong belief
        from holding the right one and failing to act on it.</p>
        {line_chart([
            ("true E[T-t | T>t]", "--cool", truth),
            ("agent's estimate", "--signal", [(int(d), float(b)) for d, b, _ in beliefs]),
         ], x_label="episode day") if beliefs
         else line_chart([("true E[T-t | T>t]", "--cool", truth)], x_label="episode day")}
      </section>
    </div>

    <div class="grid2">
      <section class="panel">
        <h2>Data quality</h2>
        <p class="why">An empty reply that spent thought tokens is a configuration
        fault, not a reasoning failure, so it is counted apart from format errors
        and never silently excludes an episode.</p>
        {_error_table(errors)}
      </section>

      <section class="panel">
        <h2>Export</h2>
        <p class="why">A self-contained HTML report with every chart inlined, plus
        per-table CSVs and a summary JSON. Every reported number can be recomputed
        from the bundle alone, without the database.</p>
        <p><a class="btn" href="/export">Download report bundle</a></p>
      </section>
    </div>
    """


def _decision_log(episode: int = None) -> str:
    """Every moment the agent changed position, with what it believed at the time."""
    r = repo()
    episodes = _safe(lambda: r.completed_episodes(RUN_ID), [])

    if episode is not None:
        trace = _safe(lambda: r.episode_trace(RUN_ID, episode), [])
        return f"""
        <section class="panel">
          <h2>Episode {episode}, day by day</h2>
          <p class="why">Filled markers are days the position changed. "Said remaining"
          is the agent's own estimate of days left; "true" is the actual conditional
          mean it should have held.</p>
          {episode_picker(episodes, episode)}
          {episode_trace_table(trace, episode)}
          <p style="margin-top:14px"><a class="btn" data-back href="/decisions">Back to all trades</a></p>
        </section>"""

    trades = _safe(lambda: r.recent_trades(RUN_ID), [])
    costly = _safe(lambda: r.costliest_decisions(RUN_ID), [])

    return f"""
    <section class="panel">
      <h2>Every position change</h2>
      <p class="why">One row per day the agent moved, newest episodes first, with the
      model's own words underneath. "Said remaining" is its stated estimate of days
      left, next to how far that sat from the truth. "Cost" is the wealth given up
      against the optimal action on that exact day.</p>
      {episode_picker(episodes)}
      {trade_rows(trades)}
    </section>

    <section class="panel">
      <h2>Costliest decisions</h2>
      <p class="why">The days that gave up the most wealth. A large belief gap here
      means the agent reasoned from a wrong premise; a small one means it knew the
      time and did not use it.</p>
      {costly_rows(costly)}
    </section>"""


def _reading(label, value, foot, accent):
    return (
        f'<div class="reading {accent}"><div class="label">{label}</div>'
        f'<div class="value num">{value}</div><div class="foot">{foot}</div></div>'
    )


def _rail(done, leased, pending, total):
    pct = 100.0 * done / total
    return f"""
    <section class="rail">
      <div class="rail-head"><span>Episode queue</span>
        <span class="num">{pct:.1f}% complete</span></div>
      <div class="rail-track">
        <span class="rail-done" style="width:{100.0 * done / total:.2f}%"></span>
        <span class="rail-leased" style="width:{100.0 * leased / total:.2f}%"></span>
        <span class="rail-pending" style="width:{100.0 * pending / total:.2f}%"></span>
      </div>
      <div class="rail-key">
        <span><i style="background:var(--signal)"></i>{done} complete</span>
        <span><i style="background:var(--thought)"></i>{leased} in flight</span>
        <span><i style="background:var(--paper);border:1px solid var(--rule)"></i>{pending} queued</span>
      </div>
    </section>"""


def _token_chart(rows):
    if not rows:
        return empty_chart(680, 210, "waiting for the first completed episode")
    prompt, completion, thought = [], [], []
    for i, (_, p, c, t) in enumerate(rows):
        prompt.append((i, float(p or 0)))
        completion.append((i, float(c or 0)))
        thought.append((i, float(t or 0)))
    return line_chart(
        [("prompt", "--cool", prompt), ("completion", "--signal", completion),
         ("thought", "--thought", thought)],
        x_label="wall clock",
    )


def _crossover(cell) -> int:
    """The day the optimal policy switches, marked on the trading chart."""
    return 16


def _error_table(errors):
    if not errors:
        return '<p class="why">No decisions logged yet.</p>'
    total = sum(errors.values()) or 1
    order = ["ok", "repaired", "format_error", "empty_thinking_truncation",
             "illegal_action", "api_error"]
    rows = []
    for cls in order + [k for k in errors if k not in order]:
        if cls not in errors:
            continue
        n = errors[cls]
        cls_attr = ""
        if cls == "empty_thinking_truncation" and n / total > 0.005:
            cls_attr = ' class="flag"'
        elif cls == "ok":
            cls_attr = ' class="good"'
        rows.append(
            f"<tr{cls_attr}><td>{cls.replace('_', ' ')}</td>"
            f'<td class="n">{n:,}</td><td class="n">{100.0 * n / total:.2f}%</td></tr>'
        )
    return (
        '<table class="data"><thead><tr><th>outcome</th><th style="text-align:right">count</th>'
        f'<th style="text-align:right">share</th></tr></thead><tbody>{"".join(rows)}</tbody></table>'
    )


def _human(n):
    n = int(n or 0)
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return f"{n:,}"
