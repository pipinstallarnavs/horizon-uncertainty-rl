"""The live dashboard.

Server-rendered HTML with a meta-refresh, deliberately: the VM sits behind a
tunnel, the page must work with no external network access, and there is no
build step to break at 3am. Charts are inline SVG from horizon.dashboard.charts.
"""
import os

from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, JSONResponse

from horizon.dashboard.charts import bar_chart, line_chart
from horizon.dashboard.render import page
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
    """The dashboard must render even when the database is empty or unreachable."""
    try:
        return fn()
    except Exception:
        return default


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


@app.get("/", response_class=HTMLResponse)
def index():
    r = repo()
    progress = _safe(lambda: r.progress(RUN_ID), {})
    tokens = _safe(lambda: r.token_totals(RUN_ID), {})
    errors = _safe(lambda: r.error_counts(RUN_ID), {})
    series = _safe(lambda: r.token_timeseries(RUN_ID), [])
    thoughts = _safe(lambda: r.thought_tokens_by_day(RUN_ID), [])

    done = progress.get("done", 0)
    total = sum(progress.values()) or 1
    pct = 100.0 * done / total

    tiles = [
        ("episodes", f"{done}/{total}", f"{pct:.1f}% complete"),
        ("total tokens", _human(tokens.get("total_tokens", 0)), "prompt + output + thought"),
        ("thought tokens", _human(tokens.get("thought_tokens", 0)), _thought_share(tokens)),
        ("metered spend", f"${tokens.get('usd', 0.0):,.2f}", "uncapped, instrumentation only"),
    ]

    throughput = _token_series(series)
    thought_curve = [(int(day), float(mean or 0)) for day, mean, _ in thoughts]

    body = f"""
    <section class="tiles">{"".join(_tile(*t) for t in tiles)}</section>

    <section class="card">
      <h2>Token throughput</h2>
      <p class="sub">Live view of the model working. Thought tokens are billed as
      output and dominate cost on this model.</p>
      {line_chart(throughput, x_label="wall clock", y_label="tokens")}
    </section>

    <section class="card">
      <h2>Deliberation effort by day</h2>
      <p class="sub">Mean thought tokens per decision against the episode day.
      Whether the model thinks harder as the horizon problem hardens is a
      behavioural signal, not just telemetry. Marker is the day-11 bright line.</p>
      {line_chart([("mean thought tokens", "--c3", thought_curve)],
                  x_label="episode day", markers=(11,))}
    </section>

    <section class="card">
      <h2>Data quality</h2>
      {_error_table(errors)}
    </section>
    """
    return HTMLResponse(page(RUN_ID, body))


def _token_series(rows):
    if not rows:
        return []
    prompt, completion, thought = [], [], []
    for i, (_, p, c, t) in enumerate(rows):
        prompt.append((i, float(p or 0)))
        completion.append((i, float(c or 0)))
        thought.append((i, float(t or 0)))
    return [
        ("prompt", "--c1", prompt),
        ("completion", "--c2", completion),
        ("thought", "--c3", thought),
    ]


def _tile(label, value, sub):
    return (
        f'<div class="tile"><div class="tile-label">{label}</div>'
        f'<div class="tile-value">{value}</div><div class="tile-sub">{sub}</div></div>'
    )


def _thought_share(tokens):
    output = tokens.get("completion_tokens", 0) + tokens.get("thought_tokens", 0)
    if not output:
        return "no output yet"
    return f"{100.0 * tokens.get('thought_tokens', 0) / output:.0f}% of output tokens"


def _error_table(errors):
    if not errors:
        return '<p class="sub">No decisions logged yet.</p>'
    total = sum(errors.values()) or 1
    order = ["ok", "repaired", "format_error", "empty_thinking_truncation",
             "illegal_action", "api_error"]
    rows = []
    for cls in order + [k for k in errors if k not in order]:
        if cls not in errors:
            continue
        count = errors[cls]
        flag = ""
        if cls == "empty_thinking_truncation" and count / total > 0.005:
            flag = ' class="bad"'
        rows.append(
            f"<tr{flag}><td>{cls}</td><td>{count:,}</td>"
            f"<td>{100.0 * count / total:.2f}%</td></tr>"
        )
    return (
        '<table class="data"><thead><tr><th>class</th><th>count</th><th>share</th>'
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
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
