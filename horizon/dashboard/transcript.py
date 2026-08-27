"""The full record of one episode: every day, exactly as the model saw it.

The decision log answers "when did it trade?". This answers "what was it
thinking, on every single day, including the days it did nothing" -- which is
where most of the behaviour actually lives. An agent that holds through the
crossover because it never reconsidered looks identical in a trade log to one
that reconsidered every day and kept choosing wrong; only the transcript
separates them.

Position events are reconstructed here rather than read from the episode
summary, because the summary records what happened and not what was *attempted*.
A refused re-entry never appears in switch_days -- the rules cancelled it -- yet
it is the clearest evidence available that the model lost track of what it had
already spent.
"""
import html

from horizon.dashboard.decisions import row
from horizon.env.market import CASH, HOLDINGS
from horizon.env.rules import burn, is_available

EVENT_LABEL = {
    "entered": "entered",
    "rotated": "rotated",
    "exited": "sold to cash",
    "refused": "re-entry refused",
    "held": "held",
}


def _name(idx):
    try:
        return HOLDINGS[int(idx)]
    except (TypeError, ValueError, IndexError):
        return "?"


def walk(rows):
    """Replay an episode, annotating each day with its event and spent set.

    Returns a list of dicts: the trace row, plus `event`, `came_from`, `spent`
    (assets permanently given up before this day) and `available`.
    """
    out = []
    previous, mask = CASH, 0
    spent_on = {}
    for record in rows:
        r = dict(row(record))
        holding = int(r["holding"])
        requested = r.get("action")
        requested = int(requested) if requested is not None else holding

        if r.get("error") == "illegal_action" and requested != holding:
            event = "refused"
        elif holding == previous:
            event = "held"
        elif holding == CASH:
            event = "exited"
        elif previous == CASH:
            event = "entered"
        else:
            event = "rotated"

        r["event"] = event
        r["came_from"] = previous
        r["requested"] = requested
        r["spent"] = tuple(
            t for t in range(len(HOLDINGS)) if t != CASH and not is_available(mask, t)
        )
        r["spent_on"] = dict(spent_on)
        r["available"] = tuple(
            t for t in range(len(HOLDINGS)) if t == CASH or is_available(mask, t)
        )
        out.append(r)

        if holding != previous:
            if previous != CASH:
                spent_on[previous] = int(r["day"])
            mask = burn(mask, previous)
        previous = holding
    return out


def events(walked):
    """Only the days something happened -- the entry/exit/re-entry ledger."""
    return [w for w in walked if w["event"] != "held"]


def refusal_markers(walked):
    """(day, requested_asset) for every re-entry the rules cancelled."""
    return [(int(w["day"]), int(w["requested"])) for w in walked if w["event"] == "refused"]


# --------------------------------------------------------------------------- #


def _spent_note(w) -> str:
    if not w["spent"]:
        return '<span class="dim">nothing spent yet</span>'
    parts = []
    for asset in w["spent"]:
        day = w["spent_on"].get(asset)
        parts.append(f"{_name(asset)}{f' since d{day}' if day else ''}")
    return f'<span class="spent">gone: {", ".join(parts)}</span>'


def event_ledger(walked) -> str:
    """Every entry, exit, rotation and refused re-entry, in order."""
    marks = events(walked)
    if not marks:
        return '<p class="why">This episode never changed position.</p>'

    out = []
    for w in marks:
        day, event = int(w["day"]), w["event"]
        if event == "refused":
            spent_day = w["spent_on"].get(w["requested"])
            detail = (
                f'tried to re-enter <b>{_name(w["requested"])}</b>, which it left'
                + (f" on day {spent_day}" if spent_day else "")
                + f'. Rules forced it to stay in {_name(w["holding"])}.'
            )
        elif event == "entered":
            detail = f'entered <b>{_name(w["holding"])}</b> out of cash'
        elif event == "exited":
            detail = f'sold <b>{_name(w["came_from"])}</b> to cash'
        else:
            detail = f'rotated <b>{_name(w["came_from"])} &#8594; {_name(w["holding"])}</b>'

        belief = w["belief"]
        out.append(f"""
        <li class="ev ev-{event}">
          <span class="ev-day num">d{day}</span>
          <span class="ev-what">{detail}</span>
          <span class="ev-meta">said {f"{float(belief):.0f}d" if belief is not None else "&mdash;"} left
            &middot; truly {float(w["m_t"] or 0):.0f}d &middot; {_spent_note(w)}</span>
        </li>""")
    return f'<ol class="ledger">{"".join(out)}</ol>'


def transcript(walked, run_id=None) -> str:
    """Every day of the episode with the model's exact words for that day.

    Days are rendered whether or not anything happened, because "it thought hard
    and chose to do nothing" and "it never engaged" are different findings and
    only the full sequence tells them apart.
    """
    if not walked:
        return '<p class="why">No days recorded for this episode.</p>'

    out = []
    for w in walked:
        day = int(w["day"])
        thought = (w.get("thought_text") or "").strip()
        reply = (w.get("raw") or "").strip()
        belief = w["belief"]
        gap = (
            float(belief) - float(w["m_t"])
            if belief is not None and w["m_t"] is not None else None
        )
        tone = "" if gap is None else ("ok" if abs(gap) <= 5 else ("warn" if abs(gap) <= 20 else "bad"))

        body = []
        if thought:
            body.append(f'<div class="trace">{html.escape(thought)}</div>')
        else:
            body.append(
                '<p class="no-trace">No reasoning trace recorded for this day &mdash; '
                "this cell ran before traces were being captured.</p>"
            )
        if reply:
            body.append('<p class="trace-head">what it answered</p>')
            body.append(f'<div class="trace reply">{html.escape(reply)}</div>')

        key = f'{w.get("episode")}:{day}'
        out.append(f"""
        <article class="entry entry-{w["event"]}" id="d{day}">
          <header class="entry-head">
            <span class="entry-day num">day {day}</span>
            <span class="entry-tag tag-{w["event"]}">{EVENT_LABEL[w["event"]]}</span>
            <span class="entry-state">holding <b>{_name(w["holding"])}</b>
              &middot; age {w["age"]} &middot; wealth
              <span class="num">{float(w["wealth"]):.4f}</span></span>
            <span class="entry-belief">believed
              <b class="num">{f"{float(belief):.0f}d" if belief is not None else "&mdash;"}</b> left
              <span class="gap {tone}">{f"{gap:+.0f}d" if gap is not None else ""}</span>
              &middot; truly <span class="num">{float(w["m_t"] or 0):.0f}d</span></span>
            <span class="entry-spent">{_spent_note(w)}</span>
          </header>
          <details class="think" data-key="{html.escape(key)}">
            <summary>reasoning &mdash; {len(thought.split()):,} words,
              {int(w["thought_tokens"] or 0):,} thought tokens</summary>
            {"".join(body)}
          </details>
        </article>""")
    return f'<div class="transcript">{"".join(out)}</div>'


def refusal_rows(rows) -> str:
    """Run-wide: every attempt to buy back something already given up."""
    if not rows:
        return (
            '<p class="why">No agent has tried to re-enter a spent asset. Either the '
            "no-re-entry rule is being tracked correctly, or no episode has yet reached "
            "a state where breaking it was tempting.</p>"
        )
    out = []
    for record in rows:
        r = row(record)
        out.append(f"""
        <tr>
          <td class="n dim">{r["episode"]}</td>
          <td class="n day">d{r["day"]}</td>
          <td class="move">asked for <b>{_name(r["action"])}</b>
            <span class="dim">held {_name(r["holding"])}</span></td>
          <td class="n">{f'{float(r["belief"]):.0f}d' if r["belief"] is not None else "&mdash;"}</td>
          <td class="n dim">{float(r["m_t"] or 0):.0f}d</td>
        </tr>
        <tr class="said"><td></td><td colspan="4">{_short(r)}</td></tr>""")
    return f"""
    <div class="log-wrap">
      <table class="data log">
        <thead><tr><th>ep</th><th>day</th><th>the move it was denied</th>
        <th style="text-align:right">said remaining</th>
        <th style="text-align:right">truly</th></tr></thead>
        <tbody>{"".join(out)}</tbody>
      </table>
    </div>"""


def _short(r) -> str:
    from horizon.dashboard.decisions import reasoning

    return reasoning(r)
