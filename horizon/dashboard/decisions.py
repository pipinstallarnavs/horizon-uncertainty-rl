"""The decision log.

Regret says how much was lost; this says when and why. Every row is one day the
agent changed position, carrying what it believed at the time, what it was
thinking when it decided, what the optimal policy would have done, and what the
choice cost -- so a wrong trade can be read as a wrong belief or a right belief
acted on badly, which is exactly the distinction the study is built to make.

Rows arrive as tuples in Repository.TRACE_COLUMNS order. They are zipped into
dicts here rather than unpacked positionally: the column list has grown twice
already, and a silent off-by-one in a 15-way unpack would misattribute a
model's reasoning to the wrong day.
"""
import html

from horizon.env.market import HOLDINGS

TRACE_FIELDS = (
    "episode", "day", "holding", "age", "wealth", "belief", "action",
    "m_t", "h_t", "pi_star", "q_loss", "thought_tokens", "error",
    "raw", "thought_text",
)

ACTION_VERB = {
    ("CASH", "A"): "bought", ("CASH", "B"): "bought", ("CASH", "C"): "bought",
    ("A", "CASH"): "sold to cash", ("B", "CASH"): "sold to cash", ("C", "CASH"): "sold to cash",
}


def row(record) -> dict:
    """Zip a trace tuple into a dict, with came_from present only when selected."""
    fields = TRACE_FIELDS + (("came_from",) if len(record) > len(TRACE_FIELDS) else ())
    return dict(zip(fields, record))


def _name(idx):
    try:
        return HOLDINGS[int(idx)]
    except (TypeError, ValueError, IndexError):
        return "?"


def _verb(came_from, holding) -> str:
    if came_from is None:
        return "entered"
    return ACTION_VERB.get((_name(came_from), _name(holding)), "rotated")


def _belief_gap(stated, truth):
    """How far the agent's stated remaining time sat from the truth."""
    if stated is None or truth is None:
        return "&mdash;", ""
    gap = float(stated) - float(truth)
    tone = "ok" if abs(gap) <= 5 else ("warn" if abs(gap) <= 20 else "bad")
    return f"{gap:+.0f}d", tone


def _days(value):
    return f"{float(value):.0f}d" if value is not None else "&mdash;"


def _cost(value):
    return f"{float(value):.4f}" if float(value or 0.0) > 1e-9 else "&mdash;"


def reasoning(r: dict) -> str:
    """The model's deliberation and its reply, collapsed behind a disclosure.

    The reasoning trace is the only direct evidence of *why* a position changed,
    so it is shown in full rather than clipped to a preview: a summary of a
    summary is not evidence. It stays collapsed because a hundred open traces is
    not a log any more.
    """
    thought = (r.get("thought_text") or "").strip()
    reply = (r.get("raw") or "").strip()
    if not thought and not reply:
        return '<p class="no-trace">No reply recorded for this day.</p>'

    key = f'{r.get("episode")}:{r.get("day")}'
    words = len(thought.split())
    summary = (
        f"reasoning &mdash; {words:,} words, {int(r.get('thought_tokens') or 0):,} thought tokens"
        if thought
        else "reply only &mdash; the model returned no reasoning trace"
    )

    parts = []
    if thought:
        parts.append(f'<div class="trace">{html.escape(thought)}</div>')
    if reply:
        parts.append('<p class="trace-head">what it answered</p>')
        parts.append(f'<div class="trace reply">{html.escape(reply)}</div>')

    return (
        f'<details class="think" data-key="{html.escape(key)}">'
        f"<summary>{summary}</summary>{''.join(parts)}</details>"
    )


def trade_rows(rows) -> str:
    if not rows:
        return (
            '<p class="why">No position changes recorded yet. The first trades appear '
            "once an episode finishes writing.</p>"
        )

    out = []
    for record in rows:
        r = row(record)
        came_from = r.get("came_from")
        gap, tone = _belief_gap(r["belief"], r["m_t"])
        agreed = int(r["pi_star"]) == int(r["holding"]) if r["pi_star"] is not None else None
        verdict = (
            '<span class="tag tag-ok">matched optimum</span>' if agreed
            else f'<span class="tag tag-bad">optimum held {_name(r["pi_star"])}</span>'
        )
        out.append(f"""
        <tr>
          <td class="n dim">{r["episode"]}</td>
          <td class="n day">d{r["day"]}</td>
          <td class="move"><b>{_verb(came_from, r["holding"])} {_name(r["holding"])}</b>
            <span class="dim">from {_name(came_from) if came_from is not None else "cash"}</span></td>
          <td class="n">{float(r["wealth"]):.4f}</td>
          <td class="n">{_days(r["belief"])}<span class="gap {tone}">{gap}</span></td>
          <td>{verdict}</td>
          <td class="n cost">{_cost(r["q_loss"])}</td>
          <td class="n dim">{int(r["thought_tokens"] or 0)}</td>
        </tr>
        <tr class="said"><td></td><td colspan="7">{reasoning(r)}</td></tr>""")

    return f"""
    <div class="log-wrap">
      <table class="data log">
        <thead><tr>
          <th>ep</th><th>day</th><th>decision</th><th style="text-align:right">wealth</th>
          <th style="text-align:right">said remaining</th><th>vs optimum</th>
          <th style="text-align:right">cost</th><th style="text-align:right">thought</th>
        </tr></thead>
        <tbody>{"".join(out)}</tbody>
      </table>
    </div>"""


def costly_rows(rows) -> str:
    if not rows:
        return '<p class="why">No suboptimal decisions recorded yet.</p>'
    out = []
    for record in rows:
        r = row(record)
        gap, tone = _belief_gap(r["belief"], r["m_t"])
        out.append(f"""
        <tr>
          <td class="n dim">{r["episode"]}</td>
          <td class="n day">d{r["day"]}</td>
          <td class="move">held <b>{_name(r["holding"])}</b>
            <span class="dim">optimum: {_name(r["pi_star"])}</span></td>
          <td class="n">{_days(r["belief"])}<span class="gap {tone}">{gap}</span></td>
          <td class="n">{float(r["m_t"] or 0):.0f}d</td>
          <td class="n cost bad">{float(r["q_loss"]):.4f}</td>
        </tr>
        <tr class="said"><td></td><td colspan="5">{reasoning(r)}</td></tr>""")
    return f"""
    <div class="log-wrap">
      <table class="data log">
        <thead><tr>
          <th>ep</th><th>day</th><th>decision</th>
          <th style="text-align:right">said remaining</th>
          <th style="text-align:right">actually remaining</th>
          <th style="text-align:right">cost</th>
        </tr></thead>
        <tbody>{"".join(out)}</tbody>
      </table>
    </div>"""


def episode_picker(episodes, selected=None) -> str:
    if not episodes:
        return ""
    chips = []
    for episode, exit_day, wealth, regret, switches in episodes[:24]:
        active = " active" if selected is not None and int(episode) == int(selected) else ""
        sign = "over" if (regret or 0) < 0 else "under"
        chips.append(
            f'<a class="chip{active}" href="/decisions?episode={episode}">'
            f'<b>ep {episode}</b><span class="dim">exit d{exit_day} &middot; '
            f'{switches} trades &middot; {sign} by {abs(float(regret or 0)):.3f}</span></a>'
        )
    return f'<div class="chips">{"".join(chips)}</div>'


def episode_trace_table(rows, episode) -> str:
    if not rows:
        return f'<p class="why">No rows recorded for episode {episode}.</p>'
    out = []
    for record in rows:
        r = row(record)
        came_from = r.get("came_from")
        changed = came_from is not None and int(came_from) != int(r["holding"])
        gap, tone = _belief_gap(r["belief"], r["m_t"])
        out.append(f"""
        <tr{' class="switch"' if changed else ""}>
          <td class="n day">{"&#9679;" if changed else ""} d{r["day"]}</td>
          <td class="move">{_name(r["holding"])}<span class="dim">&middot; age {r["age"]}</span></td>
          <td class="n">{float(r["wealth"]):.4f}</td>
          <td class="n">{_days(r["belief"])}<span class="gap {tone}">{gap}</span></td>
          <td class="n dim">{float(r["m_t"] or 0):.0f}d</td>
          <td class="n dim">{_name(r["pi_star"])}</td>
          <td class="n">{_cost(r["q_loss"])}</td>
        </tr>
        <tr class="said"><td></td><td colspan="6">{reasoning(r)}</td></tr>""")
    return f"""
    <div class="log-wrap tall">
      <table class="data log">
        <thead><tr><th>day</th><th>holding</th><th style="text-align:right">wealth</th>
        <th style="text-align:right">said remaining</th><th style="text-align:right">true</th>
        <th style="text-align:right">optimum</th><th style="text-align:right">cost</th></tr></thead>
        <tbody>{"".join(out)}</tbody>
      </table>
    </div>"""
