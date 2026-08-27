"""Page shell and design system.

Direction: an actuarial ledger. The subject is survival under an unknown clock,
so the visual language is life tables and hazard curves rather than a generic
dashboard. Colour is load-bearing: oxide red means hazard and nothing else,
brass means the model thinking, viridian means the agent acting.

Self-contained apart from Google Fonts: no CDN scripts, no build step. The VM is
locked down and anything it cannot fetch would simply be missing.
"""

FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?'
    "family=Space+Grotesk:wght@400;500;700&"
    'family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">'
)

STYLE = """
:root {
  --paper:  #e9ebe4;
  --panel:  #f2f3ee;
  --ink:    #16202a;
  --muted:  #5d6a6b;
  --rule:   #ccd1c6;
  --hazard: #b4442e;
  --thought:#8a6a2f;
  --signal: #2e6b5e;
  --cool:   #3f6187;
  --grid:   rgba(22,32,42,.07);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --paper:  #12181d;
    --panel:  #19212a;
    --ink:    #e6eae4;
    --muted:  #8c9a99;
    --rule:   #2b3640;
    --hazard: #e0705a;
    --thought:#d3a75c;
    --signal: #6cc0a8;
    --cool:   #7ba3cf;
    --grid:   rgba(230,234,228,.07);
  }
}
:root[data-theme="dark"] {
  --paper:#12181d; --panel:#19212a; --ink:#e6eae4; --muted:#8c9a99;
  --rule:#2b3640; --hazard:#e0705a; --thought:#d3a75c; --signal:#6cc0a8;
  --cool:#7ba3cf; --grid:rgba(230,234,228,.07);
}

* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font-family: "Space Grotesk", ui-sans-serif, system-ui, sans-serif;
  font-size: 15px;
  line-height: 1.5;
  /* Faint ledger ruling: the state table this whole study is solved over. */
  background-image:
    linear-gradient(var(--grid) 1px, transparent 1px),
    linear-gradient(90deg, var(--grid) 1px, transparent 1px);
  background-size: 30px 30px, 30px 30px;
}
.num { font-family: "JetBrains Mono", ui-monospace, monospace; font-variant-numeric: tabular-nums; }

/* ---- hazard ribbon: the signature element -------------------------------- */
.ribbon { display: block; width: 100%; height: 76px; border-bottom: 1px solid var(--rule); }
.ribbon-note {
  display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap;
  padding: 6px 26px 0; font-size: 11px; color: var(--muted);
  letter-spacing: .09em; text-transform: uppercase;
}

/* ---- masthead ------------------------------------------------------------ */
header {
  display: flex; align-items: flex-end; justify-content: space-between;
  gap: 18px; flex-wrap: wrap; padding: 18px 26px 16px;
  border-bottom: 2px solid var(--ink);
}
h1 {
  margin: 0; font-size: clamp(21px, 3.4vw, 30px); font-weight: 700;
  letter-spacing: -.026em; line-height: 1.04;
}
h1 .sub { display: block; font-size: .46em; font-weight: 500; letter-spacing: .16em;
          text-transform: uppercase; color: var(--muted); margin-bottom: 5px; }
.meta { display: flex; align-items: center; gap: 14px; flex-wrap: wrap;
        font-size: 12px; color: var(--muted); letter-spacing: .05em; }
.meta b { color: var(--ink); font-weight: 500; }
.pulse { display: inline-flex; align-items: center; gap: 7px; color: var(--signal);
         font-weight: 500; letter-spacing: .1em; text-transform: uppercase; font-size: 11px; }
.pulse i { width: 7px; height: 7px; border-radius: 50%; background: var(--signal);
           animation: beat 2.4s ease-in-out infinite; }
@keyframes beat { 0%,100% { opacity: 1; transform: scale(1); }
                  50% { opacity: .35; transform: scale(.8); } }

main { padding: 22px 26px 72px; max-width: 1240px; margin: 0 auto; }

/* ---- readings ------------------------------------------------------------ */
.readings {
  display: grid; gap: 1px; background: var(--rule);
  border: 1px solid var(--rule); margin-bottom: 22px;
  grid-template-columns: repeat(auto-fit, minmax(178px, 1fr));
}
.reading { background: var(--panel); padding: 14px 16px 15px; }
.reading .label { font-size: 10px; letter-spacing: .13em; text-transform: uppercase;
                  color: var(--muted); }
.reading .value { font-size: 31px; font-weight: 700; letter-spacing: -.03em;
                  line-height: 1.05; margin: 7px 0 3px; }
.reading .foot { font-size: 11.5px; color: var(--muted); }
.reading.accent-hazard .value { color: var(--hazard); }
.reading.accent-thought .value { color: var(--thought); }
.reading.accent-signal .value { color: var(--signal); }

/* ---- progress rail ------------------------------------------------------- */
.rail { border: 1px solid var(--rule); background: var(--panel); padding: 13px 16px 15px;
        margin-bottom: 22px; }
.rail-head { display: flex; justify-content: space-between; align-items: baseline;
             gap: 12px; font-size: 11px; letter-spacing: .12em; text-transform: uppercase;
             color: var(--muted); margin-bottom: 9px; }
.rail-track { display: flex; height: 12px; background: var(--paper);
              border: 1px solid var(--rule); overflow: hidden; }
.rail-track span { display: block; transition: width .6s cubic-bezier(.4,0,.2,1); }
.rail-done { background: var(--signal); }
.rail-leased { background: var(--thought); }
.rail-pending { background: transparent; }
.rail-key { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 9px;
            font-size: 11.5px; color: var(--muted); }
.rail-key i { width: 9px; height: 9px; display: inline-block; margin-right: 6px; }

/* ---- panels -------------------------------------------------------------- */
.grid2 { display: grid; gap: 18px; grid-template-columns: repeat(auto-fit, minmax(390px, 1fr)); }
.panel { border: 1px solid var(--rule); background: var(--panel); padding: 16px 18px 18px;
         margin-bottom: 18px; }
.panel h2 { margin: 0; font-size: 13px; font-weight: 700; letter-spacing: .1em;
            text-transform: uppercase; }
.panel .why { margin: 5px 0 15px; font-size: 12.5px; color: var(--muted);
              max-width: 62ch; line-height: 1.5; }

/* ---- charts -------------------------------------------------------------- */
.chart-wrap { overflow-x: auto; }
.chart { width: 100%; height: auto; min-width: 340px; display: block; }
.gridline { stroke: var(--rule); stroke-width: 1; }
.tick { fill: var(--muted); font-size: 10px;
        font-family: "JetBrains Mono", ui-monospace, monospace; }
.tick-y { text-anchor: end; }
.axis-label { fill: var(--muted); font-size: 10px; text-anchor: middle;
              letter-spacing: .1em; text-transform: uppercase; }
.marker { stroke: var(--hazard); stroke-width: 1.5; stroke-dasharray: 3 3; }
.marker-label { fill: var(--hazard); font-size: 10px; letter-spacing: .06em;
                font-family: "JetBrains Mono", ui-monospace, monospace; }
.empty { fill: var(--muted); font-size: 12px; text-anchor: middle; letter-spacing: .04em; }
.legend { display: flex; gap: 15px; flex-wrap: wrap; margin-top: 9px;
          font-size: 11.5px; color: var(--muted); }
.key { display: inline-flex; align-items: center; gap: 6px; }
.key i { width: 12px; height: 3px; display: inline-block; }

/* ---- tables -------------------------------------------------------------- */
table.data { border-collapse: collapse; width: 100%; font-size: 13px; }
table.data th, table.data td { text-align: left; padding: 7px 10px;
                               border-bottom: 1px solid var(--rule); }
table.data th { color: var(--muted); font-weight: 500; font-size: 10px;
                letter-spacing: .12em; text-transform: uppercase; }
table.data td.n { font-family: "JetBrains Mono", ui-monospace, monospace;
                  font-variant-numeric: tabular-nums; text-align: right; }
tr.flag td { color: var(--hazard); font-weight: 700; }
tr.good td:first-child { color: var(--signal); }

/* ---- action -------------------------------------------------------------- */
.btn { display: inline-block; padding: 10px 18px; background: var(--ink);
       color: var(--paper); text-decoration: none; font-size: 12px; font-weight: 500;
       letter-spacing: .1em; text-transform: uppercase; border: 1px solid var(--ink); }
.btn:hover { background: transparent; color: var(--ink); }
.btn:focus-visible { outline: 2px solid var(--signal); outline-offset: 3px; }

footer { max-width: 1240px; margin: 0 auto; padding: 0 26px 40px;
         font-size: 11.5px; color: var(--muted); line-height: 1.65; max-width: 78ch; }

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation-duration: .001ms !important; transition-duration: .001ms !important; }
}
@media (max-width: 560px) {
  main, header, .ribbon-note, footer { padding-left: 16px; padding-right: 16px; }
  .reading .value { font-size: 26px; }
}
"""

# Poll a fragment and swap it in, so the page never blinks and scroll position
# survives. Meta-refresh reloaded the whole document every ten seconds, which
# made reading a chart mid-update impossible.
SCRIPT = """
(function () {
  var busy = false;
  async function tick() {
    if (busy || document.hidden) return;
    busy = true;
    try {
      var res = await fetch('/live', { cache: 'no-store' });
      if (res.ok) {
        var html = await res.text();
        var slot = document.getElementById('live');
        if (slot) slot.innerHTML = html;
        var stamp = document.getElementById('stamp');
        if (stamp) stamp.textContent = new Date().toLocaleTimeString();
      }
    } catch (e) {
      /* keep the last good render on the screen rather than blanking it */
    } finally {
      busy = false;
    }
  }
  setInterval(tick, 5000);
  document.addEventListener('visibilitychange', function () {
    if (!document.hidden) tick();
  });
})();
"""


def page(run_id: str, cell: dict, ribbon: str, ribbon_note: str, body: str) -> str:
    exit_name = cell.get("exit", "uniform")
    info = cell.get("info", "K")
    condition = {"K": "known distribution", "U": "unknown distribution",
                 "O": "oracle"}.get(info, info)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Random-Horizon Trading &mdash; {run_id}</title>
{FONTS}
<style>{STYLE}</style>
</head>
<body>
{ribbon}
<div class="ribbon-note">{ribbon_note}</div>
<header>
  <h1><span class="sub">Live experiment</span>Random-Horizon Trading</h1>
  <div class="meta">
    <span class="pulse"><i></i>live</span>
    <span><b>{run_id}</b></span>
    <span>{exit_name} exit &middot; {condition}</span>
    <span>gemini-3.7-flash</span>
    <span>updated <b id="stamp" class="num">&mdash;</b></span>
  </div>
</header>
<main><div id="live">{body}</div></main>
<footer>
  An agent trades in a market and is force-liquidated at a random, unknown day.
  Regret is measured against an exactly-solved optimal policy, verified by
  brute-force enumeration and Monte-Carlo rollout. Single-model study on
  gemini-3.7-flash: cross-provider generality is untested and is a limitation of
  the study, not a finding about language models in general.
</footer>
<script>{SCRIPT}</script>
</body>
</html>"""
