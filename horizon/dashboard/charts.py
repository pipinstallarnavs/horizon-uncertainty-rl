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
