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
  /* Assets get their own ramp. Hazard red, brass and viridian already mean
     something specific; reusing them for holdings would make the page lie. */
  --asset-a:    #3f6187;
  --asset-b:    #8a6a2f;
  --asset-c:    #2e6b5e;
  --asset-cash: #97a3a2;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --paper:#12181d; --panel:#19212a; --ink:#e6eae4; --muted:#8c9a99;
    --rule:#2b3640; --hazard:#e0705a; --thought:#d3a75c; --signal:#6cc0a8;
    --cool:#7ba3cf; --grid:rgba(230,234,228,.07);
    --asset-a:#7ba3cf; --asset-b:#d3a75c; --asset-c:#6cc0a8; --asset-cash:#6d7b7a;
  }
}
:root[data-theme="dark"] {
  --paper:#12181d; --panel:#19212a; --ink:#e6eae4; --muted:#8c9a99;
  --rule:#2b3640; --hazard:#e0705a; --thought:#d3a75c; --signal:#6cc0a8;
  --cool:#7ba3cf; --grid:rgba(230,234,228,.07);
  --asset-a:#7ba3cf; --asset-b:#d3a75c; --asset-c:#6cc0a8; --asset-cash:#6d7b7a;
}

* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0; background: var(--paper); color: var(--ink);
  font-family: "Space Grotesk", ui-sans-serif, system-ui, sans-serif;
  font-size: 15px; line-height: 1.5;
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
  gap: 18px; flex-wrap: wrap; padding: 18px 26px 16px; border-bottom: 2px solid var(--ink);
}
h1 { margin: 0; font-size: clamp(21px, 3.4vw, 30px); font-weight: 700;
     letter-spacing: -.026em; line-height: 1.04; }
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

/* ---- cell selector ------------------------------------------------------- */
.cells { display: flex; gap: 8px; flex-wrap: wrap; padding: 12px 26px 0; }
.cell-chip {
  display: inline-flex; flex-direction: column; gap: 1px; padding: 6px 11px 7px;
  border: 1px solid var(--rule); background: var(--panel); text-decoration: none;
  color: var(--ink); font-size: 11.5px; line-height: 1.35;
}
.cell-chip:hover { border-color: var(--ink); }
.cell-chip.active { border-color: var(--hazard); border-width: 2px; padding: 5px 10px 6px; }
.cell-chip .dim { color: var(--muted); font-size: 10px; letter-spacing: .06em;
                  text-transform: uppercase; }
.cell-chip:focus-visible { outline: 2px solid var(--signal); outline-offset: 2px; }

/* ---- tabs ---------------------------------------------------------------- */
.tabs { display: flex; border-bottom: 1px solid var(--rule); margin-bottom: 20px; }
.tab {
  appearance: none; background: none; border: 0; border-bottom: 2px solid transparent;
  padding: 11px 18px; font: inherit; font-size: 12px; font-weight: 500;
  letter-spacing: .12em; text-transform: uppercase; color: var(--muted); cursor: pointer;
}
.tab:hover { color: var(--ink); }
.tab[aria-selected="true"] { color: var(--ink); border-bottom-color: var(--hazard); }
.tab:focus-visible { outline: 2px solid var(--signal); outline-offset: -2px; }

/* ---- readings ------------------------------------------------------------ */
.readings {
  display: grid; gap: 1px; background: var(--rule);
  border: 1px solid var(--rule); margin-bottom: 22px;
  grid-template-columns: repeat(auto-fit, minmax(178px, 1fr));
}
.reading { background: var(--panel); padding: 14px 16px 15px; }
.reading .label { font-size: 10px; letter-spacing: .13em; text-transform: uppercase; color: var(--muted); }
.reading .value { font-size: 31px; font-weight: 700; letter-spacing: -.03em;
                  line-height: 1.05; margin: 7px 0 3px; }
.reading .foot { font-size: 11.5px; color: var(--muted); }
.reading.accent-hazard .value { color: var(--hazard); }
.reading.accent-thought .value { color: var(--thought); }
.reading.accent-signal .value { color: var(--signal); }

/* ---- progress rail ------------------------------------------------------- */
.rail { border: 1px solid var(--rule); background: var(--panel); padding: 13px 16px 15px;
        margin-bottom: 22px; }
.rail-head { display: flex; justify-content: space-between; align-items: baseline; gap: 12px;
             font-size: 11px; letter-spacing: .12em; text-transform: uppercase;
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
.panel { border: 1px solid var(--rule); background: var(--panel);
         padding: 16px 18px 18px; margin-bottom: 18px; }
.panel h2 { margin: 0; font-size: 13px; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; }
.panel .why { margin: 5px 0 15px; font-size: 12.5px; color: var(--muted);
              max-width: 66ch; line-height: 1.5; }

/* ---- charts -------------------------------------------------------------- */
.chart-wrap { overflow-x: auto; }
.chart { width: 100%; height: auto; min-width: 340px; display: block; }
.gridline { stroke: var(--rule); stroke-width: 1; }
.tick { fill: var(--muted); font-size: 10px; font-family: "JetBrains Mono", ui-monospace, monospace; }
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
.key i.dash { background: repeating-linear-gradient(90deg, var(--muted) 0 4px, transparent 4px 7px); }

/* ---- equity curve -------------------------------------------------------- */
.breakeven { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 2 4; }
.reference { fill: none; stroke: var(--muted); stroke-width: 1.6; stroke-dasharray: 5 4; }
.entry-line { stroke-width: 1.2; stroke-dasharray: 2 4; opacity: .75; }
.entry-label { font-size: 10px; font-weight: 700; letter-spacing: .02em;
               font-family: "JetBrains Mono", ui-monospace, monospace; }
.exit-line { stroke: var(--hazard); stroke-width: 1.6; }
.exit-label { fill: var(--hazard); font-size: 10px; letter-spacing: .06em;
              font-family: "JetBrains Mono", ui-monospace, monospace; }
.strip-label { fill: var(--panel); font-size: 9px; font-weight: 700; letter-spacing: .08em;
               font-family: "JetBrains Mono", ui-monospace, monospace; }

/* ---- tables -------------------------------------------------------------- */
table.data { border-collapse: collapse; width: 100%; font-size: 13px; }
table.data th, table.data td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--rule); }
table.data th { color: var(--muted); font-weight: 500; font-size: 10px;
                letter-spacing: .12em; text-transform: uppercase; }
table.data td.n { font-family: "JetBrains Mono", ui-monospace, monospace;
                  font-variant-numeric: tabular-nums; text-align: right; }
tr.flag td { color: var(--hazard); font-weight: 700; }
tr.good td:first-child { color: var(--signal); }

/* ---- decision log -------------------------------------------------------- */
.log-wrap { overflow-x: auto; border-top: 1px solid var(--rule); }
.log-wrap.tall { max-height: 620px; overflow-y: auto; }
table.log { font-size: 12.5px; }
table.log th { position: sticky; top: 0; background: var(--panel); z-index: 1; }
table.log td { vertical-align: baseline; }
table.log td.day { font-weight: 700; white-space: nowrap; }
table.log td.move { min-width: 190px; }
table.log .dim { color: var(--muted); font-weight: 400; margin-left: 6px; font-size: .92em; }
table.log td.cost { color: var(--hazard); font-weight: 500; }
table.log td.cost.bad { font-weight: 700; }
table.log tr.switch td { background: rgba(46,107,94,.09); }
table.log tr.said td { border-bottom: 1px solid var(--rule); padding-top: 0; }
table.log tr.said q { color: var(--muted); font-size: 11.5px; font-style: normal;
                      font-family: "JetBrains Mono", ui-monospace, monospace; }
.gap { margin-left: 7px; font-size: .88em; }
.gap.ok { color: var(--signal); }
.gap.warn { color: var(--thought); }
.gap.bad { color: var(--hazard); }
.tag { font-size: 10px; letter-spacing: .08em; text-transform: uppercase;
       padding: 2px 7px; border: 1px solid currentColor; white-space: nowrap; }
.tag-ok { color: var(--signal); }
.tag-bad { color: var(--hazard); }

.chips { display: flex; gap: 7px; flex-wrap: wrap; margin-bottom: 14px; }
.chip { display: inline-flex; flex-direction: column; gap: 1px; padding: 7px 11px;
        border: 1px solid var(--rule); text-decoration: none; color: var(--ink);
        font-size: 11.5px; background: var(--paper); }
.chip:hover { border-color: var(--ink); }
.chip.active { border-color: var(--hazard); border-width: 2px; }
.chip .dim { color: var(--muted); font-size: 10.5px; margin: 0; }
.chip:focus-visible { outline: 2px solid var(--signal); outline-offset: 2px; }

/* ---- reasoning trace ----------------------------------------------------- */
details.think { margin: 4px 0 2px; }
details.think > summary {
  cursor: pointer; list-style: none; display: inline-flex; align-items: center; gap: 7px;
  font-size: 10.5px; letter-spacing: .1em; text-transform: uppercase; color: var(--thought);
  padding: 2px 0; user-select: none;
}
details.think > summary::-webkit-details-marker { display: none; }
details.think > summary::before {
  content: "B8"; display: inline-block; transition: transform .15s ease; font-size: 11px;
}
details.think[open] > summary::before { transform: rotate(90deg); }
details.think > summary:hover { color: var(--ink); }
details.think > summary:focus-visible { outline: 2px solid var(--signal); outline-offset: 2px; }
.trace {
  margin: 8px 0 4px; padding: 11px 13px; background: var(--paper);
  border-left: 2px solid var(--thought); max-height: 340px; overflow-y: auto;
  font-family: "JetBrains Mono", ui-monospace, monospace; font-size: 11.5px;
  line-height: 1.62; color: var(--ink); white-space: pre-wrap; word-break: break-word;
}
.trace.reply { border-left-color: var(--signal); }
.trace-head { font-size: 10px; letter-spacing: .12em; text-transform: uppercase;
              color: var(--muted); margin: 9px 0 0; }
.no-trace { font-size: 11.5px; color: var(--muted); font-style: italic; }

/* ---- event ledger -------------------------------------------------------- */
.ledger { list-style: none; margin: 0; padding: 0; border-top: 1px solid var(--rule); }
.ev { display: grid; grid-template-columns: 56px 1fr; gap: 2px 12px; align-items: baseline;
      padding: 10px 0 11px; border-bottom: 1px solid var(--rule); border-left: 3px solid transparent;
      padding-left: 11px; }
.ev-day { font-weight: 700; font-size: 13px; }
.ev-what { font-size: 13.5px; }
.ev-meta { grid-column: 2; font-size: 11.5px; color: var(--muted); }
.ev-entered { border-left-color: var(--asset-a); }
.ev-rotated { border-left-color: var(--asset-c); }
.ev-exited  { border-left-color: var(--asset-cash); }
.ev-refused { border-left-color: var(--hazard); background: rgba(180,68,46,.06); }
.spent { color: var(--hazard); }

/* ---- transcript ---------------------------------------------------------- */
.transcript { border-top: 1px solid var(--rule); }
.entry { padding: 12px 0 13px 12px; border-bottom: 1px solid var(--rule);
         border-left: 3px solid transparent; }
.entry-held { border-left-color: var(--rule); }
.entry-entered { border-left-color: var(--asset-a); }
.entry-rotated { border-left-color: var(--asset-c); }
.entry-exited  { border-left-color: var(--asset-cash); }
.entry-refused { border-left-color: var(--hazard); background: rgba(180,68,46,.06); }
.entry-head { display: flex; gap: 8px 16px; flex-wrap: wrap; align-items: baseline;
              font-size: 12px; color: var(--muted); margin-bottom: 4px; }
.entry-day { font-size: 14px; font-weight: 700; color: var(--ink); min-width: 62px; }
.entry-state b, .entry-belief b { color: var(--ink); }
.entry-tag { font-size: 9.5px; letter-spacing: .1em; text-transform: uppercase;
             padding: 2px 7px; border: 1px solid currentColor; }
.tag-held { color: var(--muted); }
.tag-entered { color: var(--asset-a); }
.tag-rotated { color: var(--asset-c); }
.tag-exited { color: var(--muted); }
.tag-refused { color: var(--hazard); font-weight: 700; }
.entry .trace { max-height: none; }
.btn.ghost { background: transparent; color: var(--ink); }
.btn.ghost:hover { background: var(--ink); color: var(--paper); }

/* ---- report and glossary ------------------------------------------------- */
.findings { list-style: none; margin: 0; padding: 0; }
.findings li { padding: 12px 0 13px 14px; border-bottom: 1px solid var(--rule);
               border-left: 3px solid var(--signal); font-size: 13.5px; line-height: 1.55; }
.findings li.flagged { border-left-color: var(--hazard); background: rgba(180,68,46,.06); }
.findings li:last-child { border-bottom: 0; }
a.jump { color: var(--cool); text-decoration: none; border-bottom: 1px solid currentColor;
         white-space: nowrap; font-size: .95em; }
a.jump:hover { color: var(--ink); }
a.jump.strong { font-weight: 700; }
a.jump:focus-visible, a.cell-link:focus-visible { outline: 2px solid var(--signal);
                                                  outline-offset: 2px; }
a.cell-link { color: var(--ink); text-decoration: none; border-bottom: 1px solid var(--rule); }
a.cell-link:hover { border-bottom-color: var(--ink); }
tr.here td { background: rgba(46,107,94,.10); }
tr.here td:first-child { box-shadow: inset 3px 0 0 var(--signal); }
.tag-best { color: var(--signal); }
.tag-worst { color: var(--hazard); }

.glossary { margin: 0; }
.gloss { padding: 13px 0 14px; border-bottom: 1px solid var(--rule); }
.gloss:target { background: rgba(46,107,94,.10); box-shadow: inset 3px 0 0 var(--signal);
                padding-left: 11px; }
.gloss dt { font-size: 13.5px; font-weight: 700; letter-spacing: -.01em; margin-bottom: 4px; }
.gloss dt code { font-family: "JetBrains Mono", ui-monospace, monospace; font-size: .92em;
                 background: var(--paper); border: 1px solid var(--rule); padding: 1px 5px; }
.gloss dd { margin: 0; font-size: 13px; line-height: 1.6; color: var(--muted); max-width: 74ch; }
.gloss dd b { color: var(--ink); }
.gloss-table td { vertical-align: top; font-size: 12.5px; line-height: 1.5; }
.gloss-table td:first-child { white-space: nowrap; }

/* ---- action -------------------------------------------------------------- */
.btn { display: inline-block; padding: 10px 18px; background: var(--ink); color: var(--paper);
       text-decoration: none; font-size: 12px; font-weight: 500; letter-spacing: .1em;
       text-transform: uppercase; border: 1px solid var(--ink); }
.btn:hover { background: transparent; color: var(--ink); }
.btn:focus-visible { outline: 2px solid var(--signal); outline-offset: 3px; }

footer { margin: 0 auto; padding: 0 26px 40px; font-size: 11.5px;
         color: var(--muted); line-height: 1.65; max-width: 78ch; }

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation-duration: .001ms !important; transition-duration: .001ms !important; }
}
@media (max-width: 560px) {
  main, header, .ribbon-note, footer { padding-left: 16px; padding-right: 16px; }
  .reading .value { font-size: 26px; }
}
"""

# Poll a fragment and swap it in, so the page never blinks and scroll position
# survives. A meta-refresh reloaded the whole document every ten seconds, which
# made reading a chart mid-update impossible.
SCRIPT = """
(function () {
  var busy = false;
  var current = 'overview';
  // The decisions tab is stateful: opening one episode has to survive the next
  // poll, so the tab remembers its own URL instead of always refetching /decisions.
  var main = document.querySelector('main');
  var q = main && main.dataset.run ? '?run=' + encodeURIComponent(main.dataset.run) : '';
  var url = {
    overview: '/live' + q,
    decisions: '/decisions' + q,
    report: '/report' + q,
    glossary: '/glossary' + q
  };

  function openKeys(slot) {
    var keys = {};
    slot.querySelectorAll('details[data-key]').forEach(function (d) {
      if (d.open) keys[d.dataset.key] = 1;
    });
    return keys;
  }

  var expandAll = false;

  function restore(slot, keys, scrolls) {
    slot.querySelectorAll('details[data-key]').forEach(function (d) {
      if (expandAll || keys[d.dataset.key]) d.open = true;
    });
    slot.querySelectorAll('.log-wrap').forEach(function (el, i) {
      if (scrolls[i]) el.scrollTop = scrolls[i];
    });
  }

  function show(name) {
    current = name;
    document.querySelectorAll('.tab').forEach(function (b) {
      b.setAttribute('aria-selected', String(b.dataset.tab === name));
    });
    document.querySelectorAll('.tabpanel').forEach(function (p) {
      p.hidden = p.dataset.tab !== name;
    });
    tick();
  }

  async function swap(name, href) {
    var slot = document.querySelector('.tabpanel[data-tab="' + name + '"]');
    if (!slot) return;
    var res = await fetch(href, { cache: 'no-store' });
    if (!res.ok) return;
    // A reader with a reasoning trace expanded must not have it snap shut every
    // five seconds, and a half-scrolled log must not jump back to the top.
    var keys = openKeys(slot);
    var scrolls = Array.prototype.map.call(
      slot.querySelectorAll('.log-wrap'), function (el) { return el.scrollTop; });
    slot.innerHTML = await res.text();
    restore(slot, keys, scrolls);
  }

  async function tick() {
    if (busy || document.hidden) return;
    busy = true;
    try {
      await swap(current, url[current]);
      var stamp = document.getElementById('stamp');
      if (stamp) stamp.textContent = new Date().toLocaleTimeString();
    } catch (e) {
      /* keep the last good render on screen rather than blanking it */
    } finally {
      busy = false;
    }
  }

  document.addEventListener('click', function (e) {
    var tab = e.target.closest('.tab');
    if (tab) { e.preventDefault(); show(tab.dataset.tab); return; }
    var all = e.target.closest('[data-expand-all]');
    if (all) {
      // Reading a whole episode means reading every day, not clicking eighty
      // disclosures. The state has to outlive the poll or it undoes itself.
      expandAll = !expandAll;
      document.querySelectorAll('.transcript details[data-key]').forEach(function (d) {
        d.open = expandAll;
      });
      all.textContent = expandAll ? 'Collapse every day' : 'Expand every day';
      return;
    }
    // A number in the report is only trustworthy if you can reach the rows
    // behind it, so report links switch tab and load their own target.
    var goto = e.target.closest('a[data-goto]');
    if (goto) {
      e.preventDefault();
      var target = goto.dataset.goto;
      expandAll = false;
      url[target] = goto.getAttribute('href');
      show(target);
      var slot = document.querySelector('.tabpanel[data-tab="' + target + '"]');
      if (slot) slot.scrollIntoView({ block: 'start' });
      return;
    }

    var link = e.target.closest('a.chip, a.btn[data-back]');
    if (link && current === 'decisions') {
      e.preventDefault();
      expandAll = false;
      url.decisions = link.getAttribute('href');
      swap('decisions', url.decisions).then(function () {
        var slot = document.querySelector('.tabpanel[data-tab="decisions"]');
        if (slot) slot.scrollIntoView({ block: 'start' });
      });
    }
  });

  setInterval(tick, 5000);
  document.addEventListener('visibilitychange', function () { if (!document.hidden) tick(); });
})();
"""


def _cell_chips(runs, active: str) -> str:
    """One chip per experiment cell in the database.

    The study is a grid, not a single run, and a reader who cannot move between
    cells cannot compare them -- which is the entire point of the grid.
    """
    if not runs or len(runs) < 2:
        return ""
    out = []
    for r in runs:
        c = r.get("cell") or {}
        done, total = r.get("done", 0), r.get("total", 0)
        pct = f"{100.0 * done / total:.0f}%" if total else "queued"
        out.append(
            f'<a class="cell-chip{" active" if r["run_id"] == active else ""}" '
            f'href="/?run={r["run_id"]}"><b>{r["run_id"]}</b>'
            f'<span class="dim">{c.get("exit", "?")} &middot; {c.get("info", "?")} '
            f'&middot; {pct}</span></a>'
        )
    return f'<div class="cells">{"".join(out)}</div>'


def page(run_id: str, cell: dict, ribbon: str, ribbon_note: str,
         body: str, decision_log: str, runs=(), report: str = "",
         glossary: str = "") -> str:
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
{_cell_chips(runs, run_id)}
<main data-run="{run_id}">
  <div class="tabs" role="tablist">
    <button class="tab" role="tab" data-tab="overview" aria-selected="true">Overview</button>
    <button class="tab" role="tab" data-tab="report" aria-selected="false">Report</button>
    <button class="tab" role="tab" data-tab="decisions" aria-selected="false">Decision log</button>
    <button class="tab" role="tab" data-tab="glossary" aria-selected="false">Glossary</button>
  </div>
  <div class="tabpanel" data-tab="overview" role="tabpanel">{body}</div>
  <div class="tabpanel" data-tab="report" role="tabpanel" hidden>{report}</div>
  <div class="tabpanel" data-tab="decisions" role="tabpanel" hidden>{decision_log}</div>
  <div class="tabpanel" data-tab="glossary" role="tabpanel" hidden>{glossary}</div>
</main>
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
