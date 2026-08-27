"""Inline SVG chart primitives.

Charts are rendered server-side as SVG rather than pulled from a CDN, because
the dashboard is served behind a tunnel from a locked-down VM and must work with
no external network access. Everything is theme-aware via CSS custom properties.
"""
import html
import math


def _scale(values, lo, hi, size, invert=False):
    span = (hi - lo) or 1.0
    out = []
    for value in values:
        frac = (value - lo) / span
        out.append(size * (1.0 - frac) if invert else size * frac)
    return out


def line_chart(series, width=680, height=220, pad=36, y_label="", x_label="",
               markers=(), y_min=None, y_max=None):
    """Multi-series line chart. `series` is [(name, colour_var, [(x, y), ...])]."""
    points = [p for _, _, pts in series for p in pts]
    if not points:
        return empty_chart(width, height, "no data yet")

    xs = [p[0] for p in points]
    ys = [p[1] for p in points if p[1] is not None and not math.isnan(p[1])]
    if not ys:
        return empty_chart(width, height, "no data yet")

    x_lo, x_hi = min(xs), max(xs)
    y_lo = min(ys) if y_min is None else y_min
    y_hi = max(ys) if y_max is None else y_max
    if y_hi == y_lo:
        y_hi = y_lo + 1.0

    plot_w, plot_h = width - 2 * pad, height - 2 * pad
    parts = [
        f'<svg viewBox="0 0 {width} {height}" class="chart" '
        f'role="img" aria-label="{html.escape(y_label or "chart")}">'
    ]

    for i in range(5):
        y = pad + plot_h * i / 4
        value = y_hi - (y_hi - y_lo) * i / 4
        parts.append(f'<line x1="{pad}" y1="{y:.1f}" x2="{width - pad}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{pad - 6}" y="{y + 4:.1f}" class="tick tick-y">{_fmt(value)}</text>')

    for day in markers:
        if x_lo <= day <= x_hi:
            x = pad + plot_w * (day - x_lo) / ((x_hi - x_lo) or 1)
            parts.append(
                f'<line x1="{x:.1f}" y1="{pad}" x2="{x:.1f}" y2="{height - pad}" class="marker"/>'
            )
            parts.append(f'<text x="{x + 4:.1f}" y="{pad + 12}" class="marker-label">day {day}</text>')

    for name, colour, pts in series:
        clean = [(x, y) for x, y in pts if y is not None and not math.isnan(y)]
        if not clean:
            continue
        coords = []
        for x, y in clean:
            px = pad + plot_w * (x - x_lo) / ((x_hi - x_lo) or 1)
            py = pad + plot_h * (1.0 - (y - y_lo) / (y_hi - y_lo))
            coords.append(f"{px:.1f},{py:.1f}")
        parts.append(
            f'<polyline points="{" ".join(coords)}" fill="none" '
            f'stroke="var({colour})" stroke-width="2" class="series"/>'
        )

    parts.append(
        f'<text x="{width / 2}" y="{height - 6}" class="axis-label">{html.escape(x_label)}</text>'
    )
    parts.append("</svg>")

    legend = "".join(
        f'<span class="key"><i style="background:var({c})"></i>{html.escape(n)}</span>'
        for n, c, _ in series
    )
    return f'<div class="chart-wrap">{"".join(parts)}<div class="legend">{legend}</div></div>'


def bar_chart(labels, values, width=680, height=220, pad=36, colour="--c1", markers=()):
    if not values:
        return empty_chart(width, height, "no data yet")
    plot_w, plot_h = width - 2 * pad, height - 2 * pad
    hi = max(values) or 1
    bar_w = plot_w / max(len(values), 1)
    parts = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img">']

    for i in range(5):
        y = pad + plot_h * i / 4
        parts.append(f'<line x1="{pad}" y1="{y:.1f}" x2="{width - pad}" y2="{y:.1f}" class="grid"/>')
        parts.append(
            f'<text x="{pad - 6}" y="{y + 4:.1f}" class="tick tick-y">'
            f"{_fmt(hi - hi * i / 4)}</text>"
        )

    for i, value in enumerate(values):
        h = plot_h * (value / hi)
        x = pad + i * bar_w
        parts.append(
            f'<rect x="{x:.1f}" y="{pad + plot_h - h:.1f}" width="{max(bar_w - 1, 1):.1f}" '
            f'height="{h:.1f}" fill="var({colour})" rx="1"/>'
        )

    for day in markers:
        if 0 <= day < len(values):
            x = pad + day * bar_w + bar_w / 2
            parts.append(
                f'<line x1="{x:.1f}" y1="{pad}" x2="{x:.1f}" y2="{height - pad}" class="marker"/>'
            )

    step = max(1, len(labels) // 10)
    for i in range(0, len(labels), step):
        x = pad + i * bar_w + bar_w / 2
        parts.append(
            f'<text x="{x:.1f}" y="{height - pad + 14}" class="tick">{html.escape(str(labels[i]))}</text>'
        )
    parts.append("</svg>")
    return f'<div class="chart-wrap">{"".join(parts)}</div>'


def empty_chart(width, height, message):
    return (
        f'<div class="chart-wrap"><svg viewBox="0 0 {width} {height}" class="chart">'
        f'<text x="{width / 2}" y="{height / 2}" class="empty">{html.escape(message)}</text>'
        f"</svg></div>"
    )


def _fmt(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}k"
    if abs(value) >= 10:
        return f"{value:.0f}"
    return f"{value:.2f}"
