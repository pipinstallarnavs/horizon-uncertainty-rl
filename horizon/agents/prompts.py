"""Prompt construction.

Only the LIQUIDATION paragraph differs between information conditions; every
other byte is identical across arms (DESIGN.md P5). That is what makes the
O/K/U information ladder a clean comparison rather than a confounded one.

Prompt text is committed and never edited after a run begins. Any change after
the first full run requires rerunning the whole grid, per EXPERIMENTS.md T3.3.
"""
from horizon.agents.base import Observation
from horizon.env.exit_dist import ExitDistribution
from horizon.env.market import HOLDINGS, Market

ACTION_GRAMMAR = r"^(HOLD|SWITCH (A|B|C|CASH))$"


def market_rules(market: Market) -> str:
    """The literal return table. Nothing about the market is hidden (P2)."""
    lines = []
    for asset in market.assets:
        if asset.name == "CASH":
            continue
        if asset.early_rate == asset.late_rate:
            lines.append(
                f"- Asset {asset.name}: while held, returns {asset.early_rate:+.2%}/day, always."
            )
        else:
            lines.append(
                f"- Asset {asset.name}: while held, returns {asset.early_rate:+.2%}/day for your "
                f"first {asset.breakpoint} days holding it, then {asset.late_rate:+.2%}/day."
            )
    lines.append("- CASH: 0.00%/day.")
    lines.append(
        f"- Switching your holding costs {market.switch_cost:.2%} of your wealth. Holding is free."
    )
    lines.append("- Return clocks reset when you switch into an asset.")
    lines.append(
        "- ONCE YOU LEAVE AN ASSET (A, B or C) YOU CAN NEVER HOLD IT AGAIN. Choose the "
        "order you use them carefully. CASH is the exception: you may move in and out "
        "of CASH as often as you like."
    )
    return "\n".join(lines)


def _mass_runs(support):
    """Group consecutive days carrying equal probability into (start, end, p) runs."""
    runs = []
    for day, prob in support:
        if runs and abs(prob - runs[-1][2]) < 1e-12 and day == runs[-1][1] + 1:
            runs[-1][1] = day
        else:
            runs.append([day, day, prob])
    return runs


def _probability_table(support) -> str:
    lines = []
    for start, end, prob in _mass_runs(support):
        span = f"day {start}" if start == end else f"days {start}-{end}"
        suffix = "" if start == end else " each"
        lines.append(f"    P(liquidation on {span}) = {prob:.4f}{suffix}")
    return "\n".join(lines)


def _hazard_table(dist: ExitDistribution) -> str:
    lines = [
        f"    day {t}: {dist.hazard(t):.4f}"
        for t in range(1, dist.N + 1)
        if dist.hazard(t) > 0
    ]
    if len(lines) > 12:
        lines = lines[:6] + ["    ..."] + lines[-6:]
    return "\n".join(lines)


def _verbal_description(dist: ExitDistribution, support, horizon: int) -> str:
    if dist.name == "uniform":
        return f"it is equally likely to be any day from 1 to {horizon}."
    if dist.name == "bimodal":
        early = [d for d, _ in support if d <= horizon // 2]
        late = [d for d, _ in support if d > horizon // 2]
        early_mass = sum(p for d, p in support if d <= horizon // 2)
        early_span = f"day {early[0]}" if len(early) == 1 else f"days {early[0]}-{early[-1]}"
        return (
            f"with probability {early_mass:.0%} it happens on {early_span}; "
            f"otherwise it is equally likely to be any day from {late[0]} to {late[-1]}."
        )
    if dist.name == "geometric":
        return (
            f"each day there is a {dist.hazard(1):.0%} chance of liquidation, independent "
            f"of how long you have survived, and it happens by day {horizon} at the latest."
        )
    return f"it occurs on some day between 1 and {horizon}."


def liquidation_paragraph(
    condition: str,
    dist: ExitDistribution,
    variant: str = "table",
    true_exit_day: int = None,
) -> str:
    """The only paragraph that varies across information conditions."""
    horizon = dist.N
    header = "- At the end of each day, your position may be force-liquidated, ending the game.\n"

    if condition == "O":
        if true_exit_day is None:
            raise ValueError("condition O requires the realised exit day")
        return header + f"- You will be liquidated at the end of day {true_exit_day}. This is certain."

    if condition == "U":
        return header + (
            "- The liquidation day is random. You have no information about its "
            f"distribution beyond that it occurs by day {horizon}."
        )

    if condition != "K":
        raise ValueError(f"unknown information condition {condition!r}")

    support = [(t, float(dist.pmf[t])) for t in range(1, horizon + 1) if dist.pmf[t] > 0]

    if variant == "verbal":
        return header + "- " + _verbal_description(dist, support, horizon)
    if variant == "hazard":
        return header + (
            "- The liquidation day is random. Each day you survive, the chance of being "
            "liquidated tomorrow changes. The chance of liquidation on day t, given you "
            "have survived to day t, is:\n" + _hazard_table(dist)
        )
    if variant == "minimal":
        return header + f"- The liquidation day is random and occurs by day {horizon}."

    # Default: table plus a one-line plain-language gloss, so the arm does not
    # accidentally test table-reading rather than reasoning (DESIGN.md P5).
    return header + (
        "- The liquidation day is random, with these probabilities:\n"
        + _probability_table(support)
        + "\n- In plain language: "
        + _verbal_description(dist, support, horizon)
    )


def system_prompt(
    market: Market,
    dist: ExitDistribution,
    condition: str,
    variant: str = "table",
    probe: bool = True,
    true_exit_day: int = None,
) -> str:
    """The rules block: sent once per episode, identical within a cell so the
    provider's prompt cache can hit."""
    reply_format = (
        "Line 1: BELIEF: <your estimate of expected days remaining before liquidation, one number>\n"
        "Line 2: ACTION: HOLD  - or -  ACTION: SWITCH <A|B|C|CASH>"
        if probe
        else "ACTION: HOLD  - or -  ACTION: SWITCH <A|B|C|CASH>"
    )
    return (
        "You are a trading agent in a simulated market. Rules:\n\n"
        "MARKET (fully known, no hidden information):\n"
        f"{market_rules(market)}\n\n"
        "LIQUIDATION:\n"
        f"{liquidation_paragraph(condition, dist, variant, true_exit_day)}\n"
        "- Your final score is your wealth at liquidation. Maximize expected final wealth.\n\n"
        "Each day you will receive your current state and must reply in EXACTLY this format:\n"
        f"{reply_format}"
    )


def daily_prompt(obs: Observation, probe: bool = True, history: str = "compressed") -> str:
    """The per-day state block. Compressed history by default (DESIGN.md T4)."""
    if history == "none" or not obs.switches:
        switch_text = "none yet"
    else:
        switch_text = "; ".join(f"day {d} -> {HOLDINGS[a]}" for d, a in obs.switches)

    spent = ", ".join(HOLDINGS[t] for t in obs.spent)
    spent_line = f"\nNo longer available (already left): {spent}." if spent else ""

    reply = (
        "Reply now in the required two-line format."
        if probe
        else "Reply now with the required ACTION line."
    )
    return (
        f"Day {obs.day}. You are still in the game.\n"
        f"Holding: {obs.holding_name} (held for {obs.age} days). Wealth: {obs.wealth:.4f}.\n"
        f"Switches so far: {switch_text}.{spent_line}\n"
        f"{reply}"
    )
