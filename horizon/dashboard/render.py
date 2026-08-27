"""Page shell and styling.

Self-contained: no CDN, no external fonts, no build step. The VM is locked down
and the page is served through a tunnel, so anything it cannot fetch would
simply be missing.
"""

STYLE = """
:root {
  --bg: #fbfaf8; --panel: #ffffff; --ink: #1a1a19; --muted: #6b6a66;
  --line: #e5e2dc; --accent: #2f5d50;
  --c1: #2f5d50; --c2: #b8763e; --c3: #7a4e8c; --c4: #3d6b8f;
  --bad: #a63d3d;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #16161a; --panel: #1e1e24; --ink: #ececef; --muted: #9a99a3;
    --line: #2e2e36; --accent: #7fb8a5;
    --c1: #7fb8a5; --c2: #e0a06a; --c3: #b18cc9; --c4: #7aa8cc;
    --bad: #e08585;
  }
}
:root[data-theme="dark"] {
  --bg: #16161a; --panel: #1e1e24; --ink: #ececef; --muted: #9a99a3;
  --line: #2e2e36; --accent: #7fb8a5;
  --c1: #7fb8a5; --c2: #e0a06a; --c3: #b18cc9; --c4: #7aa8cc;
  --bad: #e08585;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font: 15px/1.55 ui-sans-serif, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
header {
  padding: 22px 28px; border-bottom: 1px solid var(--line);
  display: flex; align-items: baseline; gap: 16px; flex-wrap: wrap;
}
h1 { margin: 0; font-size: 19px; letter-spacing: -0.01em; font-weight: 620; }
.run { color: var(--muted); font-size: 13px; font-family: ui-monospace, monospace; }
main { padding: 24px 28px 64px; max-width: 1100px; margin: 0 auto; }
.tiles {
  display: grid; gap: 14px; margin-bottom: 24px;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
}
.tile { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; }
.tile-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.07em; color: var(--muted); }
.tile-value { font-size: 27px; font-weight: 640; margin: 5px 0 2px; letter-spacing: -0.02em; }
.tile-sub { font-size: 12px; color: var(--muted); }
.card {
  background: var(--panel); border: 1px solid var(--line);
  border-radius: 10px; padding: 18px 20px; margin-bottom: 20px;
}
.card h2 { margin: 0 0 3px; font-size: 15px; font-weight: 620; }
.sub { color: var(--muted); font-size: 13px; margin: 0 0 14px; max-width: 70ch; }
.chart-wrap { overflow-x: auto; }
.chart { width: 100%; height: auto; min-width: 420px; display: block; }
.grid { stroke: var(--line); stroke-width: 1; }
.tick { fill: var(--muted); font-size: 10px; }
.tick-y { text-anchor: end; }
.axis-label { fill: var(--muted); font-size: 11px; text-anchor: middle; }
.marker { stroke: var(--c2); stroke-width: 1.5; stroke-dasharray: 4 3; }
.marker-label { fill: var(--c2); font-size: 10px; }
.empty { fill: var(--muted); font-size: 13px; text-anchor: middle; }
.legend { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 8px; font-size: 12px; color: var(--muted); }
.key { display: inline-flex; align-items: center; gap: 6px; }
.key i { width: 11px; height: 3px; border-radius: 2px; display: inline-block; }
table.data { border-collapse: collapse; width: 100%; font-size: 13px; }
table.data th, table.data td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--line); }
table.data th { color: var(--muted); font-weight: 550; font-size: 11px;
                text-transform: uppercase; letter-spacing: 0.06em; }
table.data td:nth-child(2), table.data td:nth-child(3) { font-variant-numeric: tabular-nums; }
tr.bad td { color: var(--bad); font-weight: 600; }
.btn {
  display: inline-block; padding: 9px 16px; border-radius: 7px;
  background: var(--accent); color: var(--panel); text-decoration: none;
  font-size: 13px; font-weight: 600;
}
.btn:hover { filter: brightness(1.08); }
footer { color: var(--muted); font-size: 12px; padding: 0 28px 32px; max-width: 1100px; margin: 0 auto; }
"""


def page(run_id: str, body: str, refresh: int = 10) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="{refresh}">
<title>Random-Horizon Trading</title>
<style>{STYLE}</style>
</head>
<body>
<header>
  <h1>Random-Horizon Trading</h1>
  <span class="run">run {run_id} &middot; gemini-3.7-flash &middot; refreshing every {refresh}s</span>
</header>
<main>{body}</main>
<footer>
  Regret is measured against an exactly-solved optimal policy, verified by
  brute-force enumeration and Monte-Carlo rollout. Single-model study: no
  cross-provider generality is claimed.
</footer>
</body>
</html>"""
