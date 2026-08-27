"""Inline SVG chart primitives.

Rendered server-side rather than pulled from a charting CDN: the VM is locked
down and the page must work with no external scripts. Every colour is a CSS
custom property so the whole set re-themes at once and stays meaningful --
oxide red is hazard, brass is the model thinking, viridian is the agent acting.
"""
import html
import math


def _clean(points):
    return [(x, y) for x, y in points if y is not None and not math.isnan(float(y))]


def hazard_ribbon(hazard, marker_day=None, width=1400, height=76):
    """The signature element: the running condition's own hazard function.

    Each day is a vertical bar whose height is h(t), the chance of being
    liquidated on that day given survival to it. The marker shows where the
    fleet's decisions currently sit, so the danger the agents are actually
    trading through is visible at a glance rather than buried in a panel.
    """
    if not hazard:
        return f'<svg class="ribbon" viewBox="0 0 {width} {height}" role="img"></svg>'

    n = len(hazard)
    peak = max(hazard) or 1.0
    step = width / n
    bars = []
    for i, h in enumerate(hazard):
        if h <= 0:
            continue
        bar_h = max(1.0, (h / peak) * (height - 16))
        bars.append(
            f'<rect x="{i * step:.2f}" y="{height - bar_h:.2f}" '
            f'width="{max(step - 0.6, 0.7):.2f}" height="{bar_h:.2f}" '
            f'fill="var(--hazard)" opacity="{0.30 + 0.55 * (h / peak):.2f}"/>'
        )

    mark = ""
    if marker_day and 1 <= marker_day <= n:
        x = (marker_day - 0.5) * step
        mark = (
            f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{height}" '
            f'stroke="var(--signal)" stroke-width="2"/>'
            f'<circle cx="{x:.1f}" cy="6" r="3.5" fill="var(--signal)"/>'
        )

    return (
        f'<svg class="ribbon" viewBox="0 0 {width} {height}" preserveAspectRatio="none" '
        f'role="img" aria-label="Hazard by day for the running exit distribution">'
        f'{"".join(bars)}{mark}</svg>'
    )


def line_chart(series, width=680, height=210, pad=38, x_label="", markers=(),
               y_min=None, y_max=None, fill=False):
    """Multi-series line chart. `series` is [(name, css_var, [(x, y), ...])]."""
    points = [p for _, _, pts in series for p in _clean(pts)]
    if not points:
        return empty_chart(width, height, "waiting for data")

    xs = [p[0] for p in points]
    ys = [float(p[1]) for p in points]
    x_lo, x_hi = min(xs), max(xs)
    y_lo = min(ys) if y_min is None else y_min
    y_hi = max(ys) if y_max is None else y_max
    if y_hi <= y_lo:
        y_hi = y_lo + 1.0

    pw, ph = width - 2 * pad, height - 2 * pad
    out = [
        f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" '
        f'aria-label="{html.escape(x_label or "chart")}">'
    ]

    for i in range(5):
        y = pad + ph * i / 4
        out.append(f'<line x1="{pad}" y1="{y:.1f}" x2="{width - pad}" y2="{y:.1f}" class="gridline"/>')
        out.append(
            f'<text x="{pad - 7}" y="{y + 3.5:.1f}" class="tick tick-y">'
            f"{_fmt(y_hi - (y_hi - y_lo) * i / 4)}</text>"
        )

    def px(x):
        return pad + pw * (x - x_lo) / ((x_hi - x_lo) or 1)

    def py(y):
        return pad + ph * (1.0 - (float(y) - y_lo) / (y_hi - y_lo))

    for day in markers:
        if x_lo <= day <= x_hi:
            out.append(
                f'<line x1="{px(day):.1f}" y1="{pad - 6}" x2="{px(day):.1f}" '
                f'y2="{height - pad}" class="marker"/>'
                f'<text x="{px(day) + 5:.1f}" y="{pad - 1}" class="marker-label">d{day}</text>'
            )

    for name, colour, pts in series:
        clean = _clean(pts)
        if not clean:
            continue
        coords = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in clean)
        if fill:
            area = f"{px(clean[0][0]):.1f},{pad + ph:.1f} {coords} {px(clean[-1][0]):.1f},{pad + ph:.1f}"
            out.append(f'<polygon points="{area}" fill="var({colour})" opacity=".12"/>')
        out.append(
            f'<polyline points="{coords}" fill="none" stroke="var({colour})" '
            f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
        )

    if x_label:
        out.append(
            f'<text x="{width / 2}" y="{height - 7}" class="axis-label">'
            f"{html.escape(x_label)}</text>"
        )
    out.append("</svg>")

    legend = "".join(
        f'<span class="key"><i style="background:var({c})"></i>{html.escape(n)}</span>'
        for n, c, pts in series
        if _clean(pts)
    )
    return f'<div class="chart-wrap">{"".join(out)}</div><div class="legend">{legend}</div>'


def bar_chart(values, width=680, height=210, pad=38, colour="--cool", markers=(),
              x_label="", highlight=None):
    """Counts by index. `highlight` recolours one bar (the optimal switch day)."""
    if not values or max(values) == 0:
        return empty_chart(width, height, "waiting for data")

    pw, ph = width - 2 * pad, height - 2 * pad
    hi = max(values)
    bw = pw / max(len(values), 1)
    out = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img">']

    for i in range(5):
        y = pad + ph * i / 4
        out.append(f'<line x1="{pad}" y1="{y:.1f}" x2="{width - pad}" y2="{y:.1f}" class="gridline"/>')
        out.append(f'<text x="{pad - 7}" y="{y + 3.5:.1f}" class="tick tick-y">{_fmt(hi - hi * i / 4)}</text>')

    for i, v in enumerate(values):
        if v <= 0:
            continue
        h = ph * (v / hi)
        c = "--signal" if highlight is not None and i == highlight else colour
        out.append(
            f'<rect x="{pad + i * bw:.1f}" y="{pad + ph - h:.1f}" '
            f'width="{max(bw - 0.8, 1):.1f}" height="{h:.1f}" fill="var({c})"/>'
        )

    for day in markers:
        if 0 <= day < len(values):
            x = pad + day * bw + bw / 2
            out.append(
                f'<line x1="{x:.1f}" y1="{pad - 6}" x2="{x:.1f}" y2="{height - pad}" class="marker"/>'
                f'<text x="{x + 5:.1f}" y="{pad - 1}" class="marker-label">d{day}</text>'
            )

    stride = max(1, len(values) // 10)
    for i in range(0, len(values), stride):
        out.append(
            f'<text x="{pad + i * bw + bw / 2:.1f}" y="{height - pad + 15}" '
            f'class="tick" text-anchor="middle">{i}</text>'
        )
    if x_label:
        out.append(f'<text x="{width / 2}" y="{height - 6}" class="axis-label">{html.escape(x_label)}</text>')
    out.append("</svg>")
    return f'<div class="chart-wrap">{"".join(out)}</div>'


def empty_chart(width, height, message):
    return (
        f'<div class="chart-wrap"><svg viewBox="0 0 {width} {height}" class="chart" role="img">'
        f'<text x="{width / 2}" y="{height / 2}" class="empty">{html.escape(message)}</text>'
        f"</svg></div>"
    )


def _fmt(v: float) -> str:
    a = abs(v)
    if a >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    if a >= 1_000:
        return f"{v / 1_000:.1f}k"
    if a >= 10:
        return f"{v:.0f}"
    if a >= 1:
        return f"{v:.1f}"
    return f"{v:.2f}"


# --------------------------------------------------------------------------- #
# Equity curves
#
# The question "when did it decide to enter?" is only answerable next to the
# money: a trade on day 16 means nothing until you can see the curve bend after
# it. So entries are drawn *on* the wealth line, not in a separate panel, and
# the holding strip underneath says what was owned for every day in between.

HOLDING_COLOUR = ("--asset-a", "--asset-b", "--asset-c", "--asset-cash")
HOLDING_LABEL = ("A", "B", "C", "CASH")


def _holding_colour(idx) -> str:
    try:
        return HOLDING_COLOUR[int(idx)]
    except (TypeError, ValueError, IndexError):
        return "--muted"


def equity_chart(wealth, entries=(), reference=None, bands=(), exit_day=None,
                 refusals=(), width=680, height=268, pad=42, x_label="episode day",
                 baseline=1.0):
    """Wealth over time with the position changes marked on the curve itself.

    `wealth`/`reference` are [(day, w)]; `entries` are (day, holding_index);
    `bands` are (first_day, last_day, holding_index) for the strip beneath;
    `refusals` are (day, requested_index) for moves the rules cancelled, drawn
    as hollow crosses so an attempted trade is never mistaken for a real one.
    """
    clean = _clean(wealth)
    if not clean:
        return empty_chart(width, height, "waiting for this episode's first day")

    strip = 13
    ref = _clean(reference or [])
    xs = [x for x, _ in clean + ref]
    ys = [float(y) for _, y in clean + ref] + [baseline]
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys), max(ys)
    span = (y_hi - y_lo) or max(abs(y_hi), 1.0) * 0.1
    y_lo, y_hi = y_lo - span * 0.10, y_hi + span * 0.16

    pw = width - 2 * pad
    ph = height - 2 * pad - strip - 8

    def px(x):
        return pad + pw * (x - x_lo) / ((x_hi - x_lo) or 1)

    def py(y):
        return pad + ph * (1.0 - (float(y) - y_lo) / (y_hi - y_lo))

    out = [
        f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" '
        f'aria-label="Wealth over time with position changes marked">'
    ]

    for i in range(5):
        y = pad + ph * i / 4
        out.append(f'<line x1="{pad}" y1="{y:.1f}" x2="{width - pad}" y2="{y:.1f}" class="gridline"/>')
        out.append(
            f'<text x="{pad - 7}" y="{y + 3.5:.1f}" class="tick tick-y">'
            f"{y_hi - (y_hi - y_lo) * i / 4:.3f}</text>"
        )

    # Break-even. Below this line the agent destroyed the capital it started with.
    if y_lo < baseline < y_hi:
        out.append(
            f'<line x1="{pad}" y1="{py(baseline):.1f}" x2="{width - pad}" '
            f'y2="{py(baseline):.1f}" class="breakeven"/>'
            f'<text x="{width - pad}" y="{py(baseline) - 4:.1f}" class="tick" '
            f'text-anchor="end">break-even</text>'
        )

    if ref:
        coords = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in ref)
        out.append(f'<polyline points="{coords}" class="reference"/>')

    coords = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in clean)
    area = (
        f"{px(clean[0][0]):.1f},{py(baseline if y_lo < baseline < y_hi else y_lo):.1f} "
        f"{coords} "
        f"{px(clean[-1][0]):.1f},{py(baseline if y_lo < baseline < y_hi else y_lo):.1f}"
    )
    out.append(f'<polygon points="{area}" fill="var(--signal)" opacity=".10"/>')
    out.append(
        f'<polyline points="{coords}" fill="none" stroke="var(--signal)" '
        f'stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"/>'
    )

    if exit_day is not None and x_lo <= exit_day <= x_hi:
        out.append(
            f'<line x1="{px(exit_day):.1f}" y1="{pad - 8}" x2="{px(exit_day):.1f}" '
            f'y2="{pad + ph:.1f}" class="exit-line"/>'
            f'<text x="{px(exit_day) - 5:.1f}" y="{pad - 2}" class="exit-label" '
            f'text-anchor="end">liquidated d{exit_day}</text>'
        )

    at = dict(clean)
    for day, holding in entries:
        if not (x_lo <= day <= x_hi):
            continue
        x, y = px(day), py(at.get(day, clean[-1][1]))
        colour = _holding_colour(holding)
        label = HOLDING_LABEL[int(holding)] if 0 <= int(holding) < 4 else "?"
        out.append(
            f'<line x1="{x:.1f}" y1="{pad - 8}" x2="{x:.1f}" y2="{pad + ph:.1f}" '
            f'class="entry-line" stroke="var({colour})"/>'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.4" fill="var(--panel)" '
            f'stroke="var({colour})" stroke-width="2.4"/>'
            f'<text x="{x:.1f}" y="{pad - 11}" class="entry-label" '
            f'fill="var({colour})" text-anchor="middle">d{day}&#8594;{label}</text>'
        )

    for day, requested in refusals:
        if not (x_lo <= day <= x_hi):
            continue
        x, y = px(day), py(at.get(day, clean[-1][1]))
        label = HOLDING_LABEL[int(requested)] if 0 <= int(requested) < 4 else "?"
        out.append(
            f'<line x1="{x:.1f}" y1="{pad - 8}" x2="{x:.1f}" y2="{pad + ph:.1f}" '
            f'class="refused-line"/>'
            f'<path d="M{x - 4.5:.1f},{y - 4.5:.1f} l9,9 M{x + 4.5:.1f},{y - 4.5:.1f} l-9,9" '
            f'class="refused-mark"/>'
            f'<text x="{x:.1f}" y="{pad - 11}" class="entry-label" '
            f'fill="var(--hazard)" text-anchor="middle">d{day}&#10007;{label}</text>'
        )

    # Holding strip: what was owned, every day, with no gaps to interpret.
    top = pad + ph + 8
    for first, last, holding in bands:
        x0, x1 = px(max(first, x_lo)), px(min(last, x_hi))
        w = max(x1 - x0, 1.2)
        out.append(
            f'<rect x="{x0:.1f}" y="{top:.1f}" width="{w:.1f}" height="{strip}" '
            f'fill="var({_holding_colour(holding)})" opacity=".82"/>'
        )
        if w > 22:
            out.append(
                f'<text x="{x0 + w / 2:.1f}" y="{top + strip - 3.5:.1f}" '
                f'class="strip-label" text-anchor="middle">'
                f"{HOLDING_LABEL[int(holding)]}</text>"
            )

    stride = max(1, int((x_hi - x_lo) // 10) or 1)
    for d in range(int(x_lo), int(x_hi) + 1, stride):
        out.append(
            f'<text x="{px(d):.1f}" y="{top + strip + 13:.1f}" class="tick" '
            f'text-anchor="middle">{d}</text>'
        )
    if x_label:
        out.append(
            f'<text x="{width / 2}" y="{height - 3}" class="axis-label">'
            f"{html.escape(x_label)}</text>"
        )
    out.append("</svg>")

    keys = ['<span class="key"><i style="background:var(--signal)"></i>agent wealth</span>']
    if ref:
        keys.append('<span class="key"><i class="dash"></i>optimal policy, same world</span>')
    if refusals:
        keys.append(
            '<span class="key"><i style="background:var(--hazard)"></i>'
            "re-entry refused by the rules</span>"
        )
    return f'<div class="chart-wrap">{"".join(out)}</div><div class="legend">{"".join(keys)}</div>'


def holding_bands(rows):
    """Collapse a per-day holding sequence into contiguous (first, last, holding)."""
    bands, entries = [], []
    start = previous = None
    for day, holding in rows:
        if previous is None:
            start, previous = day, holding
            entries.append((day, holding))
            continue
        if holding != previous:
            bands.append((start, day, previous))
            entries.append((day, holding))
            start, previous = day, holding
    if previous is not None:
        bands.append((start, rows[-1][0], previous))
    return bands, entries
