"""Report export.

Produces a bundle from which every reported number can be recomputed without
touching the database, which is what EXPERIMENTS.md section 6 requires of the
artifacts. The HTML report is self-contained: charts are inline SVG, so it
opens anywhere with no network.
"""
import csv
import io
import json
import zipfile
from datetime import datetime, timezone

import numpy as np

from horizon.dashboard.charts import bar_chart, line_chart
from horizon.dashboard.render import FONTS, STYLE
from horizon.eval.metrics import (
    EpisodeRecord,
    belief_error_curve,
    clock_coefficient,
    event_timing_score,
    regret,
    switch_counts,
    switch_day_histogram,
    trade_alignment,
)


def records_from_rows(rows):
    """Rebuild EpisodeRecords from the `episodes` table."""
    out = []
    for episode, exit_day, wealth, holdings, beliefs, switch_days in rows:
        out.append(
            EpisodeRecord(
                episode=episode,
                true_exit_day=exit_day,
                terminal_wealth=float(wealth),
                holdings=tuple(holdings or ()),
                beliefs=tuple(beliefs or ()),
                switch_days=tuple(switch_days or ()),
            )
        )
    return out


def compute_tables(records, dist, v_star: float, event_day: int = 11) -> dict:
    """Every reported number, in one dict."""
    wealth = [r.terminal_wealth for r in records]
    switches = switch_counts(records)
    timing = event_timing_score(records, event_day=event_day)
    curve, counts = belief_error_curve(records, dist)
    finite = curve[~np.isnan(curve)]
    return {
        "episodes": len(records),
        "v_star": v_star,
        "mean_terminal_wealth": float(np.mean(wealth)) if wealth else float("nan"),
        "regret": regret(records, v_star),
        "regret_pct_of_v_star": (regret(records, v_star) / v_star * 100.0) if v_star else float("nan"),
        "clock_coefficient": clock_coefficient(records),
        "mean_switches": float(switches.mean()) if switches.size else 0.0,
        "trade_alignment": trade_alignment(records, dist),
        "event_timing": timing,
        "mean_belief_error": float(finite.mean()) if finite.size else float("nan"),
        "belief_error_curve": curve.tolist(),
        "belief_error_counts": counts.tolist(),
        "switch_day_histogram": switch_day_histogram(records, dist.N).tolist(),
    }


def _table(rows):
    body = "".join(
        f"<tr><td>{name}</td><td>{value}</td></tr>" for name, value in rows
    )
    return f'<table class="data"><tbody>{body}</tbody></table>'


def _fmt(value, digits=4):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if np.isnan(value):
            return "n/a"
        return f"{value:.{digits}f}"
    return str(value)


def html_report(run_id: str, cell: dict, tables: dict, dist, event_day: int = 11) -> str:
    """A standalone HTML report. Inline SVG only, no external assets."""
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    hist = tables["switch_day_histogram"]
    curve = tables["belief_error_curve"]
    truth = [dist.mean_remaining(t) for t in range(1, dist.N + 1)]

    headline = _table(
        [
            ("Episodes", tables["episodes"]),
            ("V* (optimal expected wealth)", _fmt(tables["v_star"], 5)),
            ("Mean terminal wealth", _fmt(tables["mean_terminal_wealth"], 5)),
            ("<strong>Regret</strong>", f"<strong>{_fmt(tables['regret'], 5)}</strong>"),
            ("Regret as % of V*", _fmt(tables["regret_pct_of_v_star"], 2) + "%"),
            ("Clock coefficient", _fmt(tables["clock_coefficient"], 5)),
            ("Mean switches per episode", _fmt(tables["mean_switches"], 2)),
            ("Trade alignment", _fmt(tables["trade_alignment"], 3)),
            ("Event-timing lift", _fmt(tables["event_timing"]["lift"], 3)),
            ("Mean belief error (days)", _fmt(tables["mean_belief_error"], 2)),
        ]
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Random-Horizon Trading &mdash; {run_id}</title>
{FONTS}
<style>{STYLE}</style></head>
<body>
<header><h1><span class="sub">Report</span>Random-Horizon Trading</h1>
<div class="meta"><span><b>{run_id}</b></span><span>generated {generated}</span></div></header>
<main>
  <section class="panel">
    <h2>Cell</h2>
    <p class="why">{json.dumps(cell)}</p>
  </section>

  <section class="panel">
    <h2>Headline results</h2>
    <p class="why">Regret is measured against an exactly-solved optimal policy,
    verified by brute-force enumeration and Monte-Carlo rollout.</p>
    {headline}
  </section>

  <section class="panel">
    <h2>Switch-day distribution</h2>
    <p class="why">When the agent traded. The marker is the bright line at day
    {event_day}, where the optimal action changes.</p>
    {bar_chart(hist, markers=(event_day,), highlight=event_day, x_label='episode day')}
  </section>

  <section class="panel">
    <h2>Belief calibration</h2>
    <p class="why">Mean absolute error between the agent's stated expected days
    remaining and the true conditional mean. Separates belief-formation failures
    from belief-to-action failures.</p>
    {line_chart([("belief error (days)", "--thought",
                  [(i + 1, v) for i, v in enumerate(curve)])],
                x_label="episode day", markers=(event_day,))}
  </section>

  <section class="panel">
    <h2>Ground truth: expected remaining time</h2>
    <p class="why">The quantity the agent's beliefs are scored against.</p>
    {line_chart([("E[T-t | T>t]", "--cool", [(i + 1, v) for i, v in enumerate(truth)])],
                x_label="episode day", markers=(event_day,))}
  </section>
</main>
<footer>Single-model study on gemini-3.7-flash. Cross-provider generality is
untested and is a limitation of the study, not a finding about LLMs in general.</footer>
</body></html>"""


def export_bundle(path: str, run_id: str, cell: dict, records, dist, v_star: float,
                  event_day: int = 11) -> str:
    """Write a zip containing the HTML report, CSVs and the raw episode records."""
    tables = compute_tables(records, dist, v_star, event_day)

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr(f"{run_id}-report.html", html_report(run_id, cell, tables, dist, event_day))
        bundle.writestr(
            f"{run_id}-summary.json",
            json.dumps({"run_id": run_id, "cell": cell, **tables}, indent=2, default=str),
        )

        episodes_csv = io.StringIO()
        writer = csv.writer(episodes_csv)
        writer.writerow(["episode", "true_exit_day", "terminal_wealth", "regret", "switch_days"])
        for record in records:
            writer.writerow(
                [
                    record.episode,
                    record.true_exit_day,
                    f"{record.terminal_wealth:.8f}",
                    f"{v_star - record.terminal_wealth:.8f}",
                    " ".join(str(d) for d in record.switch_days),
                ]
            )
        bundle.writestr(f"{run_id}-episodes.csv", episodes_csv.getvalue())

        curve_csv = io.StringIO()
        writer = csv.writer(curve_csv)
        writer.writerow(["day", "mean_belief_error", "n", "true_mean_remaining"])
        for i, (err, n) in enumerate(
            zip(tables["belief_error_curve"], tables["belief_error_counts"]), start=1
        ):
            writer.writerow([i, "" if err != err else f"{err:.6f}", int(n),
                             f"{dist.mean_remaining(i):.6f}"])
        bundle.writestr(f"{run_id}-belief-calibration.csv", curve_csv.getvalue())

    return path
