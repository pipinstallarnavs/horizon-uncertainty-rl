"""The decision log.

Regret says how much was lost; this says when and why. Every row is one day the
agent changed position, carrying what it believed at the time, what the optimal
policy would have done, and what that choice cost -- so a wrong trade can be
read as a wrong belief or a right belief acted on badly, which is exactly the
distinction the study is built to make.
"""
import html

from horizon.env.market import HOLDINGS

ACTION_VERB = {
    ("CASH", "A"): "bought", ("CASH", "B"): "bought", ("CASH", "C"): "bought",
    ("A", "CASH"): "sold to cash", ("B", "CASH"): "sold to cash", ("C", "CASH"): "sold to cash",
}


def _name(idx):
    try:
        return HOLDINGS[int(idx)]
    except (TypeError, ValueError, IndexError):
        return "?"


def _verb(came_from, holding) -> str:
    if came_from is None:
        return "entered"
    a, b = _name(came_from), _name(holding)
    return ACTION_VERB.get((a, b), "rotated")


def _belief_gap(stated, truth):
    """How far the agent's stated remaining time sat from the truth."""
    if stated is None or truth is None:
        return "&mdash;", ""
    gap = float(stated) - float(truth)
    tone = "ok" if abs(gap) <= 5 else ("warn" if abs(gap) <= 20 else "bad")
    return f"{gap:+.0f}d", tone


def _reasoning(raw: str) -> str:
    """The model's own words, trimmed to the lines that carry the decision."""
    if not raw:
        return ""
    lines = [ln.strip() for ln in raw.strip().splitlines() if ln.strip()]
    return html.escape(" · ".join(lines[:3])[:240])


def trade_rows(rows) -> str:
    if not rows:
        return (
            '<p class="why">No position changes recorded yet. The first trades appear '
            "once an episode finishes writing.</p>"
        )

    out = []
    for (episode, day, holding, age, wealth, belief, action, m_t, h_t, pi_star,
         q_loss, thought, error, raw, came_from) in rows:
        verb = _verb(came_from, holding)
        gap, tone = _belief_gap(belief, m_t)
        optimal = _name(pi_star)
        agreed = int(pi_star) == int(holding) if pi_star is not None else None
        cost = float(q_loss or 0.0)

        verdict = (
            '<span class="tag tag-ok">matched optimum</span>' if agreed
            else f'<span class="tag tag-bad">optimum held {optimal}</span>'
        )
        cost_cell = f"{cost:.4f}" if cost > 1e-9 else "&mdash;"

        out.append(f"""
        <tr>
          <td class="n dim">{episode}</td>
          <td class="n day">d{day}</td>
          <td class="move"><b>{verb} {_name(holding)}</b>
            <span class="dim">from {_name(came_from) if came_from is not None else "cash"}</span></td>
          <td class="n">{float(wealth):.4f}</td>
          <td class="n">{f"{float(belief):.0f}d" if belief is not None else "&mdash;"}
            <span class="gap {tone}">{gap}</span></td>
          <td>{verdict}</td>
          <td class="n cost">{cost_cell}</td>
          <td class="n dim">{int(thought or 0)}</td>
        </tr>
        <tr class="said"><td></td><td colspan="7"><q>{_reasoning(raw)}</q></td></tr>""")

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
    for (episode, day, holding, age, wealth, belief, action, m_t, h_t, pi_star,
         q_loss, thought, error, raw) in rows:
        gap, tone = _belief_gap(belief, m_t)
        out.append(f"""
        <tr>
          <td class="n dim">{episode}</td>
          <td class="n day">d{day}</td>
          <td class="move">held <b>{_name(holding)}</b>
            <span class="dim">optimum: {_name(pi_star)}</span></td>
          <td class="n">{f"{float(belief):.0f}d" if belief is not None else "&mdash;"}
            <span class="gap {tone}">{gap}</span></td>
          <td class="n">{float(m_t or 0):.0f}d</td>
          <td class="n cost bad">{float(q_loss):.4f}</td>
        </tr>""")
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
            f'<b>ep {episode}</b><span class="dim">exit d{exit_day} · '
            f'{switches} trades · {sign} by {abs(float(regret or 0)):.3f}</span></a>'
        )
    return f'<div class="chips">{"".join(chips)}</div>'


def episode_trace_table(rows, episode) -> str:
    if not rows:
        return f'<p class="why">No rows recorded for episode {episode}.</p>'
    out = []
    for (ep, day, holding, age, wealth, belief, action, m_t, h_t, pi_star,
         q_loss, thought, error, raw, came_from) in rows:
        changed = came_from is not None and int(came_from) != int(holding)
        gap, tone = _belief_gap(belief, m_t)
        row_class = ' class="switch"' if changed else ""
        marker = "&#9679;" if changed else ""
        out.append(f"""
        <tr{row_class}>
          <td class="n day">{marker} d{day}</td>
          <td class="move">{_name(holding)}<span class="dim"> · age {age}</span></td>
          <td class="n">{float(wealth):.4f}</td>
          <td class="n">{f"{float(belief):.0f}d" if belief is not None else "&mdash;"}
            <span class="gap {tone}">{gap}</span></td>
          <td class="n dim">{float(m_t or 0):.0f}d</td>
          <td class="n dim">{_name(pi_star)}</td>
          <td class="n">{f"{float(q_loss):.4f}" if float(q_loss or 0) > 1e-9 else "&mdash;"}</td>
        </tr>""")
    return f"""
    <div class="log-wrap tall">
      <table class="data log">
        <thead><tr><th>day</th><th>holding</th><th style="text-align:right">wealth</th>
        <th style="text-align:right">said remaining</th><th style="text-align:right">true</th>
        <th style="text-align:right">optimum</th><th style="text-align:right">cost</th></tr></thead>
        <tbody>{"".join(out)}</tbody>
      </table>
    </div>"""
