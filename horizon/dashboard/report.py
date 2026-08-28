"""The report and the glossary.

Two problems this solves. First, every number on this dashboard is named in the
vocabulary of the protocol -- V*, m(t), q-loss, condition K -- and a reader who
does not already know that vocabulary cannot check a single claim. Second, a
summary figure is only trustworthy if you can get from it to the rows it came
from, so every result here links to the evidence rather than asking to be taken
on faith.

Run ids are decoded rather than listed. "t2-5" is not self-explanatory and
nobody should have to read a compose file to find out what they are looking at.
"""
import html

EXIT_MEANING = {
    "uniform": (
        "Uniform",
        "The liquidation day is equally likely to be any day from 1 to 100. The "
        "hazard rises steadily as days pass, because each survived day removes "
        "one of the remaining chances.",
    ),
    "bimodal": (
        "Bimodal",
        "Almost all the probability sits on day 10 or in a late window. Surviving "
        "day 10 is a bright line: the expected remaining time jumps upward the "
        "moment it passes, so an agent that reasons about the horizon should "
        "visibly change behaviour there.",
    ),
    "geometric": (
        "Geometric",
        "A flat 8% chance of liquidation every day, memoryless. Surviving tells "
        "you nothing new, so the optimal policy barely changes with time. This is "
        "the control: behaviour that still shifts with the clock here is "
        "responding to elapsed time itself, not to the horizon.",
    ),
    "sawtooth": (
        "Sawtooth",
        "A repeating hazard pattern, used to test whether a policy learned on one "
        "distribution transfers to another shape.",
    ),
}

INFO_MEANING = {
    "K": (
        "Known distribution",
        "The agent is told the exact liquidation distribution in its prompt. Any "
        "failure here is a failure to <em>act</em> on the horizon, not to know it.",
    ),
    "U": (
        "Unknown distribution",
        "The agent is told only that liquidation happens by day 100, with no "
        "information about how the risk is spread. Separates not knowing the "
        "horizon from not using it.",
    ),
    "O": (
        "Oracle",
        "The agent is told the exact day it will be liquidated. This removes the "
        "inference problem entirely and upper-bounds what any prompt can achieve.",
    ),
}

GLOSSARY = [
    ("run-id", "Run id, e.g. <code>t2-5</code>", """
     Reads as generation, then cell. <b>t2</b> is the second generation of the
     Tier-1 grid; <b>t1</b> was the first, run with a 512-token output budget
     that was clipping the model's reasoning mid-sentence. The digit after the
     dash is which of the five cells it is. Both generations are kept and both
     are readable here, but only <b>t2</b> cells carry reasoning traces."""),
    ("cell", "Cell", """
     One combination of experimental settings -- an exit distribution crossed
     with an information condition -- played over 200 episodes. Cells run
     concurrently and independently, each with its own queue of episodes."""),
    ("episode", "Episode", """
     One complete game: the agent starts holding cash on day 1, trades daily, and
     is force-liquidated on a day it does not know in advance. Its wealth on that
     day is the only thing that counts."""),
    ("horizon", "Horizon", """
     The last day liquidation can happen, fixed at 100. Not the day it
     <em>will</em> happen -- that is drawn per episode."""),
    ("exit-day", "Exit day, <i>T</i>", """
     The day this particular episode was actually liquidated, drawn from the exit
     distribution before the episode begins and never shown to the agent (except
     under the oracle condition)."""),
    ("hazard", "Hazard, <i>h(t)</i>", """
     The chance of being liquidated on day <i>t</i>, given survival to day
     <i>t</i>. The red ribbon across the top of the page is exactly this. It is
     what makes the horizon problem time-inhomogeneous: the danger of holding a
     slow asset changes as the days pass, even though the asset does not."""),
    ("m-t", "Mean remaining, <i>m(t)</i>", """
     <i>E[T &minus; t | T &gt; t]</i> -- how many more days the agent should
     expect, given it has survived to day <i>t</i>. This is the ground truth the
     agent's stated belief is scored against. Under a uniform exit it falls
     steadily; under a geometric exit it barely moves."""),
    ("belief", "Said remaining / BELIEF", """
     The agent states, every single day, how many days it thinks are left. This
     is the probe. Comparing it against <i>m(t)</i> is what separates
     <em>forming the wrong belief</em> from <em>holding the right belief and
     failing to act on it</em> -- the central distinction of the study."""),
    ("v-star", "<i>V*</i>", """
     The expected terminal wealth of the exactly optimal policy, computed by
     backward induction over every reachable state and verified by brute-force
     enumeration and Monte-Carlo rollout. Not a heuristic benchmark: it is the
     best any policy could do."""),
    ("regret", "Regret", """
     <i>V*</i> minus what the agent actually ended with. Positive means it did
     worse than optimal. Because it is measured against an expectation, a single
     lucky episode can show negative regret; the mean over many is the figure
     that means something."""),
    ("q-loss", "Cost / q-loss", """
     For one specific day: the wealth given up by taking the action it took
     instead of the optimal action, holding everything else fixed. This is what
     lets mistakes be weighted by how much they cost rather than merely
     counted."""),
    ("crossover", "Crossover day", """
     The day the optimal policy rotates out of the fast-decaying asset into the
     slow-compounding one. Under these market parameters it is day 16. An agent
     clustering its trades there is timing to the horizon; one trading uniformly
     across days is not."""),
    ("no-re-entry", "No-re-entry rule", """
     Once the agent leaves A, B or C it may never hold that asset again. Cash is
     exempt. Without this rule the optimal policy degenerates into farming an
     age-clock exploit -- cycling between two assets forever -- which would make
     the whole environment measure the wrong thing."""),
    ("spent", "Spent / gone", """
     Assets the agent has permanently given up by leaving them. Shown on every
     day of the transcript, because what is still available is part of the state
     the agent has to keep track of."""),
    ("refused", "Re-entry refused", """
     The agent asked to move back into an asset it had already left. The rules
     cancelled the move and it stayed where it was. These never appear in the
     trade record -- the trade did not happen -- but they are the clearest
     evidence available that the agent lost track of what it had spent."""),
    ("age", "Age", """
     How many consecutive days the current asset has been held. Assets pay
     different rates before and after a breakpoint measured in age, so this is
     part of the state, not decoration."""),
    ("switch-cost", "Switch cost", """
     0.1% of wealth, paid on every move between holdings. Small, but it makes
     churning strictly costly and gives the optimal policy a reason to commit."""),
    ("thought-tokens", "Thought tokens", """
     Tokens the model spent reasoning before answering. They are billed as output
     and, on this model, cannot be switched off. They dominate cost here --
     roughly 98% of output spend."""),
    ("trace", "Reasoning trace", """
     The model's own account of how it reached the decision, returned by the API
     alongside the answer. It is a <b>summary</b> of the reasoning, not the raw
     internal tokens: no current Gemini model returns those. Shown verbatim and
     unedited, but that ceiling is a limit on the evidence, not a choice."""),
    ("error-classes", "Outcome classes", """
     <b>ok</b> parsed first try. <b>repaired</b> needed one re-ask to produce
     valid format. <b>format_error</b> failed twice and defaulted to holding.
     <b>illegal_action</b> asked for a move the rules forbade. <b>api_error</b>
     the request itself failed. <b>empty_thinking_truncation</b> is a
     configuration fault, not a reasoning failure -- it gets its own class so a
     misconfigured output budget can never be misreported as the model failing at
     temporal reasoning."""),
    ("crn", "Common random numbers", """
     Exit days are drawn from a frozen seed bank keyed by episode number, so
     every cell and every baseline faces the identical set of worlds. That makes
     comparisons paired, which is far more sensitive than comparing independent
     samples."""),
]


def _esc(v):
    return html.escape(str(v))


def decode(run_id: str, cell: dict) -> str:
    """One plain sentence saying what a cell actually is."""
    exit_name = cell.get("exit", "uniform")
    info = cell.get("info", "K")
    exit_title = EXIT_MEANING.get(exit_name, (exit_name, ""))[0].lower()
    info_title = INFO_MEANING.get(info, (info, ""))[0].lower()
    return f"{exit_title} exit, {info_title}"


def _num(value, fmt="{:.4f}", dash="&mdash;"):
    return dash if value is None else fmt.format(value)


def jump(target, href, text, cls="jump"):
    """A link that switches tab and loads the evidence behind a number."""
    return f'<a class="{cls}" data-goto="{target}" href="{href}">{text}</a>'


# --------------------------------------------------------------------------- #


def cell_table(rows, active) -> str:
    """Every cell in the study, with its result so far. Rows are clickable."""
    if not rows:
        return '<p class="why">No cells have been created yet.</p>'
    out = []
    for r in rows:
        run_id = r["run_id"]
        cell = r.get("cell") or {}
        summary = r.get("summary") or {}
        done, total = r.get("done", 0), r.get("total", 0)
        traces = "yes" if r.get("has_traces") else "no"
        out.append(f"""
        <tr class="{'here' if run_id == active else ''}">
          <td><a class="cell-link" href="/?run={_esc(run_id)}"><b>{_esc(run_id)}</b></a>
            <span class="dim">{_esc(decode(run_id, cell))}</span></td>
          <td class="n">{done}<span class="dim">/{total}</span></td>
          <td class="n">{_num(summary.get("wealth"))}</td>
          <td class="n">{_num(summary.get("regret"), "{:+.4f}")}</td>
          <td class="n">{_num(summary.get("first_trade"), "d{:.1f}")}</td>
          <td class="n dim">{traces}</td>
        </tr>""")
    return f"""
    <div class="log-wrap">
      <table class="data">
        <thead><tr><th>cell</th><th style="text-align:right">episodes</th>
        <th style="text-align:right">mean wealth</th>
        <th style="text-align:right">mean regret</th>
        <th style="text-align:right">first trade</th>
        <th style="text-align:right">traces</th></tr></thead>
        <tbody>{"".join(out)}</tbody>
      </table>
    </div>
    <p class="why" style="margin-top:10px">Click a cell id to switch the whole
    dashboard to it. Regret is measured against that cell's own <i>V*</i>, so the
    column is comparable across rows even though the cells differ.</p>"""


def findings(run_id, cell, summary, crossover, refused_count, errors) -> str:
    """The three or four sentences a reader actually wants, each one clickable."""
    if not summary.get("episodes"):
        return (
            '<p class="why">No episode has finished in this cell yet. Findings appear '
            "as soon as the first one lands.</p>"
        )

    first = summary.get("first_trade")
    regret = summary.get("regret")
    decisions = sum(errors.values()) or 1
    clean = 100.0 * (errors.get("ok", 0) + errors.get("repaired", 0)) / decisions

    items = []

    if first is not None:
        delta = first - crossover
        verdict = (
            "essentially on the optimal day" if abs(delta) < 1.0
            else (f"{abs(delta):.1f} days {'late' if delta > 0 else 'early'}")
        )
        items.append(f"""
        <li><b>The first trade lands on day {first:.1f}</b> on average, against an
        optimal crossover of day {crossover} &mdash; {verdict}. This is the moment the
        agent acts on the clock, so it is the single most informative number here.
        {jump("overview", f"/live?run={run_id}", "See the trade-day distribution &#8594;")}</li>""")

    if regret is not None:
        direction = "below" if regret > 0 else "above"
        items.append(f"""
        <li><b>Mean regret is {regret:+.4f}</b>, i.e. terminal wealth
        {abs(regret):.4f} {direction} the exactly optimal policy on the same worlds.
        {jump("decisions", f"/decisions?run={run_id}", "See the costliest individual decisions &#8594;")}</li>""")

    items.append(f"""
    <li><b>{clean:.1f}% of decisions parsed cleanly</b> out of {decisions:,}.
    Anything failing to parse defaults to holding and is recorded, never silently
    dropped.
    {jump("overview", f"/live?run={run_id}", "See the outcome breakdown &#8594;")}</li>""")

    if refused_count:
        items.append(f"""
        <li class="flagged"><b>{refused_count} attempted re-entries were refused</b> by
        the rules &mdash; the agent asked to buy back an asset it had already given up.
        {jump("decisions", f"/decisions?run={run_id}", "Read the reasoning behind them &#8594;")}</li>""")
    else:
        items.append("""
        <li><b>No agent has tried to re-enter a spent asset.</b> Across every decision
        in this cell the no-re-entry rule was tracked correctly, which is a real
        result about state tracking and not merely an absence of errors.</li>""")

    return f'<ul class="findings">{"".join(items)}</ul>'


def extremes_table(rows, run_id) -> str:
    """Best and worst episodes, each linking straight to its transcript."""
    if not rows:
        return '<p class="why">No finished episodes to rank yet.</p>'
    out = []
    for kind, episode, exit_day, wealth, regret, switches, first, cost in rows:
        out.append(f"""
        <tr class="rank-{kind}">
          <td><span class="tag tag-{kind}">{"played best" if kind == "best" else "played worst"}</span></td>
          <td class="n">{jump("decisions", f"/decisions?episode={episode}&amp;run={run_id}",
                              f"episode {episode}", "jump strong")}</td>
          <td class="n cost">{float(cost or 0):.4f}</td>
          <td class="n">d{exit_day}</td>
          <td class="n">{"&mdash;" if first is None else f"d{int(first)}"}</td>
          <td class="n">{switches}</td>
          <td class="n">{float(wealth):.4f}</td>
          <td class="n dim">{float(regret or 0):+.4f}</td>
        </tr>""")
    return f"""
    <div class="log-wrap">
      <table class="data">
        <thead><tr><th></th><th style="text-align:right">episode</th>
        <th style="text-align:right">decision cost</th>
        <th style="text-align:right">liquidated</th>
        <th style="text-align:right">first trade</th>
        <th style="text-align:right">trades</th>
        <th style="text-align:right">final wealth</th>
        <th style="text-align:right">regret</th></tr></thead>
        <tbody>{"".join(out)}</tbody>
      </table>
    </div>
    <p class="why" style="margin-top:10px">Ranked by <b>decision cost</b> &mdash; the
    wealth given up against the optimal action, summed over the days actually played.
    Not by regret: regret is measured against a fixed <i>V*</i>, so an episode
    liquidated on day 2 looks catastrophic however well it was played and one
    surviving to day 100 looks brilliant however badly. Ranking by regret would sort
    these by luck. Regret is still shown, greyed, because it is the protocol's headline
    metric &mdash; but it is only meaningful averaged over paired episodes.
    Click any episode to open its full transcript.</p>"""


def glossary_body(runs) -> str:
    """Definitions, plus a decode of every run id currently in the database."""
    live = []
    for r in sorted(runs, key=lambda x: x["run_id"]):
        cell = r.get("cell") or {}
        exit_name, info = cell.get("exit", "uniform"), cell.get("info", "K")
        exit_title, exit_text = EXIT_MEANING.get(exit_name, (exit_name, ""))
        info_title, info_text = INFO_MEANING.get(info, (info, ""))
        live.append(f"""
        <tr>
          <td><a class="cell-link" href="/?run={_esc(r["run_id"])}"><b>{_esc(r["run_id"])}</b></a></td>
          <td><b>{_esc(exit_title)} exit.</b> {exit_text}</td>
          <td><b>{_esc(info_title)}.</b> {info_text}</td>
        </tr>""")

    entries = "".join(
        f'<div class="gloss" id="g-{slug}"><dt>{term}</dt><dd>{body}</dd></div>'
        for slug, term, body in GLOSSARY
    )

    return f"""
    <section class="panel">
      <h2>What the cell names mean</h2>
      <p class="why">Every run id in the database, decoded. The generation prefix says
      which configuration it ran under; the digit says which cell of the grid it is.</p>
      <div class="log-wrap">
        <table class="data gloss-table">
          <thead><tr><th>cell</th><th>exit distribution</th><th>information condition</th></tr></thead>
          <tbody>{"".join(live)}</tbody>
        </table>
      </div>
      <p class="why" style="margin-top:12px"><b>t1</b> cells ran with a 512-token output
      budget that was truncating the model's reasoning mid-sentence, so they carry no
      usable traces. <b>t2</b> is the same grid re-run with that fixed. Nothing was
      deleted: both are here, and the t1 rows remain valid for everything except
      reading the reasoning.</p>
    </section>

    <section class="panel">
      <h2>Every term on this dashboard</h2>
      <p class="why">In the order you meet them, not alphabetical.</p>
      <dl class="glossary">{entries}</dl>
    </section>"""
