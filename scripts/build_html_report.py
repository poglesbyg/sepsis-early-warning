"""Render reports/REPORT.md as a single self-contained HTML page.

Figures are inlined as data URIs so the page can be handed to someone as one
file. The Markdown subset here is exactly what report.py emits -- headings,
paragraphs, pipe tables, images, and inline bold/italic/code -- so there is no
Markdown dependency to install.
"""

from __future__ import annotations

import base64
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

FONTS = (
    "https://fonts.googleapis.com/css2?"
    "family=Spectral:ital,wght@0,400;0,600;1,400&"
    "family=IBM+Plex+Sans:wght@400;500;600&"
    "family=IBM+Plex+Mono:wght@400;500;600&display=swap"
)


# --------------------------------------------------------------------------
# Minimal Markdown
# --------------------------------------------------------------------------
def inline(text: str) -> str:
    out = html.escape(text)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", out)
    return out


def image_data_uri(rel: str) -> str | None:
    path = REPORTS / rel
    if not path.exists():
        return None
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def render_table(rows: list[str]) -> str:
    header = [c.strip() for c in rows[0].strip().strip("|").split("|")]
    aligns = [
        "right" if c.strip().endswith(":") and not c.strip().startswith(":") else "left"
        for c in rows[1].strip().strip("|").split("|")
    ]
    body = [
        [c.strip() for c in r.strip().strip("|").split("|")] for r in rows[2:]
    ]

    def cells(values, tag):
        return "".join(
            f'<{tag} class="{a}">{inline(v)}</{tag}>' for v, a in zip(values, aligns)
        )

    head = f"<thead><tr>{cells(header, 'th')}</tr></thead>"
    rows_html = "".join(f"<tr>{cells(r, 'td')}</tr>" for r in body)
    return f'<div class="scroller"><table>{head}<tbody>{rows_html}</tbody></table></div>'


def to_html(md: str) -> tuple[str, str]:
    """Return (page title, body html)."""
    lines = md.splitlines()
    title = "Sepsis Early Warning"
    parts: list[str] = []
    buffer: list[str] = []
    pending_caption: str | None = None

    def flush() -> None:
        nonlocal buffer
        if buffer:
            parts.append(f"<p>{inline(' '.join(buffer))}</p>")
            buffer = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            flush()
            i += 1
            continue

        image = re.match(r"^!\[(.*?)\]\((.*?)\)$", stripped)
        if image:
            flush()
            uri = image_data_uri(image.group(2))
            caption = pending_caption or image.group(1)
            pending_caption = None
            if uri:
                parts.append(
                    '<figure class="plate">'
                    f'<img src="{uri}" alt="{html.escape(caption)}">'
                    f"<figcaption>{inline(caption)}</figcaption></figure>"
                )
            i += 1
            continue

        if stripped == "<!-- replay -->":
            flush()
            parts.append(replay_widget())
            i += 1
            continue

        if stripped.startswith("|"):
            flush()
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            if len(block) >= 2:
                parts.append(render_table(block))
            continue

        if stripped.startswith("#"):
            flush()
            level = len(stripped) - len(stripped.lstrip("#"))
            text = stripped[level:].strip()
            if level == 1:
                title = text
            else:
                tag = f"h{min(level, 4)}"
                parts.append(f'<{tag} id="{slug(text)}">{inline(text)}</{tag}>')
            i += 1
            continue

        # A bolded standalone line immediately before an image is its caption.
        if re.fullmatch(r"\*\*.+\*\*", stripped) and i + 2 < len(lines) and lines[i + 2].strip().startswith("!["):
            flush()
            pending_caption = stripped.strip("*")
            i += 1
            continue

        buffer.append(stripped)
        i += 1

    flush()
    return title, "\n".join(parts)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


# --------------------------------------------------------------------------
# Readout strip
# --------------------------------------------------------------------------
def readout() -> str:
    payload = json.loads((REPORTS / "metrics.json").read_text())
    results = {(r["model"], r["split"]): r for r in payload["results"]}
    lead = {r["model"]: r for r in payload.get("lead_time", [])}
    champion = max(
        (r for r in payload["results"] if r["split"] == "test"), key=lambda r: r["utility"]
    )["model"]

    test = results[(champion, "test")]
    ext = results[(champion, "external")]
    timing = lead.get(champion, {})

    stats = [
        ("40,336", "ICU admissions", "two health systems"),
        (f"{test['auroc']:.3f}", "AUROC, internal test", "hospital A, scored once"),
        (f"{ext['auroc']:.3f}", "AUROC, external", "hospital B, never seen"),
        (f"{timing.get('median_lead_time_h', float('nan')):.0f} h", "median warning", "before clinical onset"),
        (f"{timing.get('detection_rate', float('nan')):.0%}", "caught before onset", f"at {timing.get('false_alarms_per_true_detection', float('nan')):.1f} false alarms each"),
    ]
    cells = "".join(
        f'<div class="stat"><span class="value">{v}</span>'
        f'<span class="label">{html.escape(l)}</span>'
        f'<span class="note">{html.escape(n)}</span></div>'
        for v, l, n in stats
    )
    return f'<section class="readout">{cells}</section>'


CSS = """
:root {
  --paper:      #f4f6f7;
  --surface:    #ffffff;
  --ink:        #16232b;
  --ink-soft:   #4a5b64;
  --ink-faint:  #7d8e97;
  --rule:       #d3dbdf;
  --rule-soft:  #e6ebed;
  --signal:     #0f6d8c;
  --signal-dim: #e2eef2;
  --flag:       #b33a1f;
  --stable:     #2f7d5c;
  --shadow:     0 1px 2px rgba(22,35,43,.06), 0 8px 24px -16px rgba(22,35,43,.28);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --paper:      #10171b;
    --surface:    #161f25;
    --ink:        #dce5e8;
    --ink-soft:   #9fb1b9;
    --ink-faint:  #6f838c;
    --rule:       #25333a;
    --rule-soft:  #1d282e;
    --signal:     #4fb6d6;
    --signal-dim: #12333f;
    --flag:       #e0714f;
    --stable:     #5fb58c;
    --shadow:     0 1px 2px rgba(0,0,0,.4), 0 8px 24px -16px rgba(0,0,0,.8);
  }
}
:root[data-theme="dark"] {
  --paper:      #10171b;
  --surface:    #161f25;
  --ink:        #dce5e8;
  --ink-soft:   #9fb1b9;
  --ink-faint:  #6f838c;
  --rule:       #25333a;
  --rule-soft:  #1d282e;
  --signal:     #4fb6d6;
  --signal-dim: #12333f;
  --flag:       #e0714f;
  --stable:     #5fb58c;
  --shadow:     0 1px 2px rgba(0,0,0,.4), 0 8px 24px -16px rgba(0,0,0,.8);
}

* { box-sizing: border-box; }

body {
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font-family: "IBM Plex Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 16px;
  line-height: 1.65;
  -webkit-font-smoothing: antialiased;
}

.page {
  max-width: 1060px;
  margin: 0 auto;
  padding: clamp(2rem, 5vw, 4.5rem) clamp(1.1rem, 4vw, 2.5rem) 6rem;
  display: flex;
  flex-direction: column;
  gap: 2rem;
}

/* --- masthead --------------------------------------------------------- */
.masthead { display: flex; flex-direction: column; gap: 1rem; }

.eyebrow {
  font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: .7rem;
  font-weight: 500;
  letter-spacing: .14em;
  text-transform: uppercase;
  color: var(--signal);
}

h1 {
  font-family: Spectral, Georgia, "Times New Roman", serif;
  font-weight: 600;
  font-size: clamp(2.1rem, 5.5vw, 3.3rem);
  line-height: 1.08;
  letter-spacing: -.015em;
  text-wrap: balance;
  margin: 0;
}

.standfirst {
  font-family: Spectral, Georgia, serif;
  font-size: clamp(1.05rem, 2.2vw, 1.2rem);
  line-height: 1.55;
  color: var(--ink-soft);
  max-width: 62ch;
  margin: 0;
}

.trace { width: 100%; height: 64px; display: block; color: var(--signal); }

.meta {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: .74rem;
  letter-spacing: .04em;
  color: var(--ink-faint);
  display: flex;
  flex-wrap: wrap;
  gap: .5rem 1.4rem;
  padding-top: .75rem;
  border-top: 1px solid var(--rule);
}

/* --- readout ---------------------------------------------------------- */
.readout {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(158px, 1fr));
  gap: 1px;
  background: var(--rule);
  border: 1px solid var(--rule);
  border-radius: 3px;
  overflow: hidden;
}
.stat {
  background: var(--surface);
  padding: 1.05rem 1.1rem 1.15rem;
  display: flex;
  flex-direction: column;
  gap: .18rem;
}
.stat .value {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-variant-numeric: tabular-nums;
  font-size: 1.55rem;
  font-weight: 500;
  letter-spacing: -.02em;
  color: var(--signal);
}
.stat .label {
  font-size: .8rem;
  font-weight: 500;
  color: var(--ink);
}
.stat .note {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: .68rem;
  letter-spacing: .02em;
  color: var(--ink-faint);
}

/* --- prose ------------------------------------------------------------ */
.body { display: flex; flex-direction: column; gap: 1.15rem; }

.body > p,
.body > h2,
.body > h3,
.body > h4 { max-width: 68ch; }

h2 {
  font-family: Spectral, Georgia, serif;
  font-weight: 600;
  font-size: clamp(1.5rem, 3.2vw, 1.9rem);
  line-height: 1.2;
  letter-spacing: -.012em;
  text-wrap: balance;
  margin: 2.6rem 0 0;
  padding-top: 1.1rem;
  border-top: 2px solid var(--ink);
}

h3 {
  font-family: "IBM Plex Sans", sans-serif;
  font-weight: 600;
  font-size: 1.06rem;
  letter-spacing: .002em;
  text-wrap: balance;
  margin: 1.9rem 0 0;
}
h3::before {
  content: "";
  display: block;
  width: 34px;
  height: 2px;
  background: var(--signal);
  margin-bottom: .7rem;
}

p { margin: 0; }

code {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: .86em;
  background: var(--signal-dim);
  color: var(--signal);
  padding: .1em .34em;
  border-radius: 2px;
}

strong { font-weight: 600; }

/* --- tables ----------------------------------------------------------- */
.scroller {
  overflow-x: auto;
  border: 1px solid var(--rule);
  border-radius: 3px;
  background: var(--surface);
  box-shadow: var(--shadow);
}
table {
  border-collapse: collapse;
  width: 100%;
  font-size: .84rem;
  font-variant-numeric: tabular-nums;
}
thead th {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: .68rem;
  font-weight: 500;
  letter-spacing: .07em;
  text-transform: uppercase;
  color: var(--ink-faint);
  background: var(--rule-soft);
  padding: .6rem .85rem;
  border-bottom: 1px solid var(--rule);
  white-space: nowrap;
}
tbody td {
  padding: .52rem .85rem;
  border-bottom: 1px solid var(--rule-soft);
  white-space: nowrap;
}
tbody tr:last-child td { border-bottom: none; }
td.right, th.right { text-align: right; }
td.left:first-child {
  font-weight: 500;
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: .8rem;
  color: var(--ink);
}
tbody td.right { font-family: "IBM Plex Mono", ui-monospace, monospace; color: var(--ink-soft); }
tbody td strong { color: var(--signal); font-weight: 600; }

/* --- figures ---------------------------------------------------------- */
.plate {
  margin: 1.2rem 0 .4rem;
  border: 1px solid var(--rule);
  border-radius: 3px;
  background: var(--surface);
  box-shadow: var(--shadow);
  overflow: hidden;
}
.plate img {
  display: block;
  width: 100%;
  height: auto;
  /* matplotlib renders on white; on the dark ground a plain PNG glares. */
  background: #fff;
}
.plate figcaption {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: .7rem;
  line-height: 1.5;
  letter-spacing: .02em;
  color: var(--ink-faint);
  padding: .7rem .9rem .8rem;
  border-top: 1px solid var(--rule);
}
:root[data-theme="dark"] .plate img,
:root:not([data-theme="light"]) .plate img { opacity: .92; }
@media (prefers-color-scheme: light) {
  :root:not([data-theme="dark"]) .plate img { opacity: 1; }
}

/* --- colophon --------------------------------------------------------- */
.colophon {
  margin-top: 3rem;
  padding-top: 1.1rem;
  border-top: 1px solid var(--rule);
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: .72rem;
  line-height: 1.7;
  color: var(--ink-faint);
  max-width: 74ch;
}
.colophon a { color: var(--signal); }

@media (prefers-reduced-motion: reduce) {
  * { animation: none !important; transition: none !important; }
}
"""

TRACE = """
<svg class="trace" viewBox="0 0 1000 64" preserveAspectRatio="none" aria-hidden="true" focusable="false">
  <path d="M0 46 L60 46 L72 40 L84 48 L96 44 L150 44 L162 38 L174 47 L186 43 L250 43
           L262 36 L274 46 L286 41 L350 41 L362 33 L374 45 L386 39 L450 38 L466 28
           L478 44 L492 35 L560 33 L578 20 L592 42 L606 30 L680 26 L700 10 L714 40
           L730 22 L800 18 L820 4 L834 38 L850 16 L1000 14"
        fill="none" stroke="currentColor" stroke-width="1.4"
        stroke-linejoin="round" opacity=".55"/>
  <line x1="700" y1="0" x2="700" y2="64" stroke="currentColor" stroke-width="1"
        stroke-dasharray="3 4" opacity=".45"/>
</svg>
"""


# --------------------------------------------------------------------------
# Replay
# --------------------------------------------------------------------------
CASE_LABELS = {
    "median_catch": "Median catch",
    "near_miss": "Only just",
    "missed": "Missed",
    "false_alarm": "False alarm",
}


REPLAY_CSS = """
.replay {
  margin: 2.4rem 0;
  border: 1px solid var(--rule);
  border-radius: 10px;
  background: var(--surface);
  box-shadow: var(--shadow);
  overflow: hidden;
}
.replay .case-bar {
  display: flex;
  flex-wrap: wrap;
  gap: 0;
  border-bottom: 1px solid var(--rule);
  background: var(--signal-dim);
}
.replay .case {
  flex: 1 1 9rem;
  display: flex;
  flex-direction: column;
  gap: .1rem;
  padding: .6rem .8rem;
  border: 0;
  border-right: 1px solid var(--rule);
  background: transparent;
  color: var(--ink-soft);
  font: inherit;
  text-align: left;
  cursor: pointer;
}
.replay .case:last-child { border-right: 0; }
.replay .case:hover { color: var(--ink); }
.replay .case[aria-current="true"] {
  background: var(--surface);
  color: var(--ink);
  box-shadow: inset 0 -2px 0 var(--signal);
}
.replay .case-role { font-size: .82rem; font-weight: 600; letter-spacing: .01em; }
.replay .case-id {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: .7rem;
  color: var(--ink-faint);
}
.replay-head {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  gap: .4rem 1.2rem;
  padding: .85rem 1rem .2rem;
}
.replay-head p { margin: 0; }
.replay .who { font-size: .84rem; color: var(--ink-soft); }
.replay .verdict { font-size: .84rem; font-weight: 600; color: var(--ink); }
.replay .verdict.fired { color: var(--flag); }
.replay .verdict.quiet { color: var(--stable); }
.replay svg.risk { display: block; width: 100%; height: auto; padding: 0 .4rem; }
.replay .vitals {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: .2rem 1rem;
  padding: 0 1rem .4rem;
}
.replay .vital { min-width: 0; }
.replay .vital-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  font-size: .72rem;
  color: var(--ink-faint);
  letter-spacing: .04em;
  text-transform: uppercase;
}
.replay .vital-head b {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: .82rem;
  color: var(--ink);
  font-weight: 500;
  text-transform: none;
}
.replay .vital svg { display: block; width: 100%; height: auto; }
.replay .transport {
  display: flex;
  align-items: center;
  gap: .75rem;
  padding: .5rem 1rem .9rem;
}
.replay .play {
  min-width: 4.4rem;
  padding: .35rem .8rem;
  border: 1px solid var(--signal);
  border-radius: 999px;
  background: var(--signal);
  color: #fff;
  font: inherit;
  font-size: .82rem;
  font-weight: 600;
  cursor: pointer;
}
.replay .play:hover { filter: brightness(1.08); }
.replay .transport input[type="range"] { flex: 1; accent-color: var(--signal); min-width: 0; }
.replay .clock {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: .78rem;
  color: var(--ink-soft);
  white-space: nowrap;
}
.replay .replay-foot {
  margin: 0;
  padding: 0 1rem 1rem;
  font-size: .74rem;
  color: var(--ink-faint);
}
.replay text { font-family: "IBM Plex Sans", system-ui, sans-serif; }
@media (max-width: 620px) {
  .replay .case { flex-basis: 50%; }
  .replay .clock { font-size: .7rem; }
}
"""


REPLAY_JS = """
(function () {
  var node = document.getElementById('replay-data');
  if (!node) return;
  var data = JSON.parse(node.textContent);
  var cases = data.cases;
  var threshold = data.threshold;
  var NS = 'http://www.w3.org/2000/svg';

  var riskSvg = document.getElementById('replay-risk');
  var vitalsBox = document.getElementById('replay-vitals');
  var scrub = document.getElementById('replay-scrub');
  var clock = document.getElementById('replay-clock');
  var playBtn = document.getElementById('replay-play');
  var whoLine = document.getElementById('replay-who');
  var verdict = document.getElementById('replay-verdict');

  var W = 960, H = 300, L = 58, R = 18, T = 20, B = 40;
  var active = 0, cursor = 0, timer = null;

  function el(name, attrs, text) {
    var n = document.createElementNS(NS, name);
    for (var k in attrs) n.setAttribute(k, attrs[k]);
    if (text !== undefined) n.textContent = text;
    return n;
  }
  function clear(n) { while (n.firstChild) n.removeChild(n.firstChild); }

  function scales(c) {
    var hours = c.hours;
    var h0 = hours[0], h1 = hours[hours.length - 1];
    var top = Math.max(Math.max.apply(null, c.risk), threshold * 1.6) * 1.12;
    return {
      x: function (h) { return L + (h1 === h0 ? 0 : (h - h0) / (h1 - h0)) * (W - L - R); },
      y: function (v) { return T + (1 - v / top) * (H - T - B); },
      top: top, h0: h0, h1: h1
    };
  }

  function path(pts) {
    return pts.map(function (p, i) { return (i ? 'L' : 'M') + p[0].toFixed(1) + ' ' + p[1].toFixed(1); }).join(' ');
  }

  function drawRisk(c) {
    clear(riskSvg);
    var s = scales(c);
    var g = el('g', {});

    // Axes and gridlines.
    [0, threshold, s.top].forEach(function (v) {
      var y = s.y(v);
      g.appendChild(el('line', {
        x1: L, x2: W - R, y1: y, y2: y,
        stroke: 'currentColor', 'stroke-width': v === threshold ? 1.2 : 1,
        'stroke-dasharray': v === threshold ? '5 4' : '',
        opacity: v === threshold ? .55 : .16
      }));
      g.appendChild(el('text', {
        x: L - 8, y: y + 3.5, 'text-anchor': 'end',
        'font-size': 11, fill: 'currentColor', opacity: .55
      }, v.toFixed(3)));
    });
    g.appendChild(el('text', {
      x: W - R, y: s.y(threshold) - 7, 'text-anchor': 'end',
      'font-size': 11, fill: 'currentColor', opacity: .6
    }, 'alert threshold'));

    // Hour axis.
    var span = s.h1 - s.h0;
    var step = span > 96 ? 24 : span > 40 ? 12 : span > 16 ? 6 : 2;
    for (var h = Math.ceil(s.h0 / step) * step; h <= s.h1; h += step) {
      g.appendChild(el('text', {
        x: s.x(h), y: H - B + 18, 'text-anchor': 'middle',
        'font-size': 11, fill: 'currentColor', opacity: .5
      }, h));
    }
    g.appendChild(el('text', {
      x: L, y: H - 6, 'font-size': 11, fill: 'currentColor', opacity: .5
    }, 'hours since ICU admission'));

    // Onset: when the care team was already acting.
    if (c.onset_hour !== null && c.onset_hour <= s.h1) {
      var ox = s.x(c.onset_hour);
      g.appendChild(el('line', {
        x1: ox, x2: ox, y1: T - 4, y2: H - B,
        stroke: 'currentColor', 'stroke-width': 1.4, 'stroke-dasharray': '2 3', opacity: .65
      }));
      g.appendChild(el('text', {
        x: ox - 6, y: T + 8, 'text-anchor': 'end', 'font-size': 11,
        fill: 'currentColor', opacity: .75
      }, 'clinical onset'));
    }

    // Lab draws, on the same timeline as the risk they produced.
    c.lab_draws.forEach(function (h) {
      g.appendChild(el('line', {
        x1: s.x(h), x2: s.x(h), y1: H - B + 2, y2: H - B + 8,
        stroke: 'currentColor', 'stroke-width': 1.6, opacity: .35, class: 'lab'
      }));
    });

    var pts = c.hours.map(function (h, i) { return [s.x(h), s.y(c.risk[i])]; });
    g.appendChild(el('path', {
      d: path(pts), fill: 'none', stroke: 'currentColor',
      'stroke-width': 1.2, opacity: .18
    }));
    g.appendChild(el('path', {
      id: 'replay-trace', d: '', fill: 'none', stroke: 'var(--signal)',
      'stroke-width': 2.2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round'
    }));
    g.appendChild(el('line', {
      id: 'replay-now', x1: 0, x2: 0, y1: T - 4, y2: H - B,
      stroke: 'var(--ink-faint)', 'stroke-width': 1, opacity: .5
    }));
    g.appendChild(el('circle', {
      id: 'replay-dot', r: 4.5, cx: -20, cy: -20, fill: 'var(--signal)'
    }));

    // Where it first crossed, drawn only once the cursor reaches it.
    if (c.first_alert_hour !== null) {
      g.appendChild(el('circle', {
        id: 'replay-alert', cx: s.x(c.first_alert_hour), cy: -20, r: 5.5,
        fill: 'none', stroke: 'var(--flag)', 'stroke-width': 2, opacity: 0
      }));
      g.appendChild(el('text', {
        id: 'replay-alert-label', x: s.x(c.first_alert_hour), y: -20,
        'text-anchor': 'middle', 'font-size': 11, 'font-weight': 600,
        fill: 'var(--flag)', opacity: 0
      }, 'first alert'));
    }
    riskSvg.appendChild(g);
    return s;
  }

  function drawVitals(c) {
    clear(vitalsBox);
    var w = 240, h = 62, pad = 6;
    data.vitals.forEach(function (name) {
      var series = c.vitals[name] || [];
      var seen = series.filter(function (v) { return v !== null; });
      var lo = seen.length ? Math.min.apply(null, seen) : 0;
      var hi = seen.length ? Math.max.apply(null, seen) : 1;
      if (hi - lo < 1e-9) { hi = lo + 1; }

      var wrap = document.createElement('div');
      wrap.className = 'vital';
      var head = document.createElement('div');
      head.className = 'vital-head';
      head.innerHTML = '<span>' + name + '</span><b data-vital="' + name + '">--</b>';
      wrap.appendChild(head);

      var svg = el('svg', { viewBox: '0 0 ' + w + ' ' + h, 'data-series': name });
      var x = function (i) {
        return pad + (series.length < 2 ? 0 : i / (series.length - 1)) * (w - 2 * pad);
      };
      var y = function (v) { return pad + (1 - (v - lo) / (hi - lo)) * (h - 2 * pad); };

      var pts = [];
      series.forEach(function (v, i) { if (v !== null) pts.push([x(i), y(v)]); });
      svg.appendChild(el('path', {
        d: path(pts), fill: 'none', stroke: 'currentColor', 'stroke-width': 1, opacity: .16
      }));
      svg.appendChild(el('path', {
        class: 'vital-trace', d: '', fill: 'none', stroke: 'currentColor',
        'stroke-width': 1.6, opacity: .75
      }));
      wrap.appendChild(svg);
      vitalsBox.appendChild(wrap);

      svg._pts = series.map(function (v, i) { return v === null ? null : [x(i), y(v)]; });
    });
  }

  function render(c, s) {
    var i = cursor;
    var pts = [];
    for (var k = 0; k <= i; k++) pts.push([s.x(c.hours[k]), s.y(c.risk[k])]);
    document.getElementById('replay-trace').setAttribute('d', path(pts));

    var cx = s.x(c.hours[i]), cy = s.y(c.risk[i]);
    var now = document.getElementById('replay-now');
    now.setAttribute('x1', cx); now.setAttribute('x2', cx);
    var dot = document.getElementById('replay-dot');
    dot.setAttribute('cx', cx); dot.setAttribute('cy', cy);
    var firing = c.risk[i] >= threshold;
    dot.setAttribute('fill', firing ? 'var(--flag)' : 'var(--signal)');

    var alert = document.getElementById('replay-alert');
    if (alert) {
      var reached = c.first_alert_hour !== null && c.hours[i] >= c.first_alert_hour;
      var ai = c.hours.indexOf(c.first_alert_hour);
      alert.setAttribute('opacity', reached ? 1 : 0);
      alert.setAttribute('cy', s.y(c.risk[ai < 0 ? 0 : ai]));
      var label = document.getElementById('replay-alert-label');
      label.setAttribute('opacity', reached ? 1 : 0);
      label.setAttribute('y', s.y(c.risk[ai < 0 ? 0 : ai]) - 12);
    }

    // Vitals, revealed to the same hour.
    vitalsBox.querySelectorAll('svg[data-series]').forEach(function (svg) {
      var name = svg.getAttribute('data-series');
      var pts2 = [];
      var latest = null;
      for (var k = 0; k <= i; k++) {
        var p = svg._pts[k];
        if (p) { pts2.push(p); latest = c.vitals[name][k]; }
      }
      svg.querySelector('.vital-trace').setAttribute('d', path(pts2));
      var out = vitalsBox.querySelector('b[data-vital="' + name + '"]');
      if (out) out.textContent = latest === null ? '--' : latest;
    });

    var hour = c.hours[i];
    var parts = ['hour ' + hour, 'risk ' + c.risk[i].toFixed(4)];
    if (firing) {
      parts.push('ALERTING');
      if (c.onset_hour !== null) {
        var gap = c.onset_hour - hour;
        parts.push(gap > 0 ? gap.toFixed(0) + ' h before onset'
                           : Math.abs(gap).toFixed(0) + ' h after onset');
      }
    } else {
      parts.push('below threshold');
    }
    clock.textContent = parts.join('  |  ');
  }

  function verdictText(c) {
    if (!c.septic) {
      return ['quiet', 'Never septic. Alerted on ' + c.n_alert_hours + ' of '
              + c.stay_hours + ' hours.'];
    }
    if (c.lead_time_hours === null) {
      return ['quiet', 'Septic. The model never crossed the threshold.'];
    }
    if (c.lead_time_hours > 0) {
      return ['fired', 'Caught ' + c.lead_time_hours.toFixed(0)
              + ' h before the care team acted.'];
    }
    return ['quiet', 'Alerted ' + Math.abs(c.lead_time_hours).toFixed(0)
            + ' h after the care team acted. Not an early warning.'];
  }

  var scale = null;
  function load(n) {
    stop();
    active = n;
    var c = cases[n];
    cursor = 0;
    scrub.max = c.hours.length - 1;
    scrub.value = 0;
    whoLine.textContent = c.patient_id + '  |  ' + (c.age === null ? 'age not recorded' : c.age + ' years')
      + '  |  ' + c.unit + '  |  ' + c.stay_hours + ' h stay';
    var v = verdictText(c);
    verdict.className = 'verdict ' + v[0];
    verdict.textContent = v[1];
    scale = drawRisk(c);
    drawVitals(c);
    render(c, scale);
    Array.prototype.forEach.call(document.querySelectorAll('.replay .case'), function (b, k) {
      if (k === n) { b.setAttribute('aria-current', 'true'); }
      else { b.removeAttribute('aria-current'); }
    });
  }

  function step() {
    var c = cases[active];
    if (cursor >= c.hours.length - 1) { stop(); return; }
    cursor += 1;
    scrub.value = cursor;
    render(c, scale);
  }

  function stop() {
    if (timer) { clearInterval(timer); timer = null; }
    playBtn.textContent = 'Play';
  }

  function play() {
    var c = cases[active];
    if (cursor >= c.hours.length - 1) { cursor = 0; scrub.value = 0; render(c, scale); }
    // About nine ICU hours a second: a whole stay in roughly ten seconds.
    timer = setInterval(step, 110);
    playBtn.textContent = 'Pause';
  }

  playBtn.addEventListener('click', function () { timer ? stop() : play(); });
  scrub.addEventListener('input', function () {
    stop();
    cursor = parseInt(scrub.value, 10) || 0;
    render(cases[active], scale);
  });
  Array.prototype.forEach.call(document.querySelectorAll('.replay .case'), function (b) {
    b.addEventListener('click', function () { load(parseInt(b.dataset.case, 10)); });
  });

  load(0);
})();
"""


def replay_widget() -> str:
    """The interactive replay, or nothing if the payload has not been built.

    Everything the widget needs is inlined: the page has to survive being emailed
    as a single file, so there is no fetch, no library and no server. The data is
    pre-computed by `sepsis replay` -- the browser only draws it.
    """
    path = REPORTS / "replay.json"
    if not path.exists():
        return ""

    payload = json.loads(path.read_text())
    tabs = "".join(
        f'<button type="button" class="case" data-case="{n}"'
        f'{" aria-current=\'true\'" if n == 0 else ""}>'
        f'<span class="case-role">{html.escape(CASE_LABELS.get(c["role"], c["role"]))}</span>'
        f'<span class="case-id">{html.escape(c["patient_id"])}</span></button>'
        for n, c in enumerate(payload["cases"])
    )
    return f"""<section class="replay" id="replay" aria-label="Admission replay">
  <script type="application/json" id="replay-data">{json.dumps(payload)}</script>
  <div class="case-bar" role="tablist">{tabs}</div>
  <div class="replay-head">
    <p class="who" id="replay-who"></p>
    <p class="verdict" id="replay-verdict"></p>
  </div>
  <svg class="risk" id="replay-risk" viewBox="0 0 960 300" role="img"
       aria-label="Model risk against ICU hour"></svg>
  <div class="vitals" id="replay-vitals"></div>
  <div class="transport">
    <button type="button" id="replay-play" class="play">Play</button>
    <input type="range" id="replay-scrub" min="0" max="1" value="0" step="1"
           aria-label="ICU hour">
    <output class="clock" id="replay-clock"></output>
  </div>
  <p class="replay-foot">Validation split, {html.escape(payload["model"])} at the frozen
     threshold of {payload["threshold"]:.4f}. Ticks under the axis mark hours when a
     sparse lab was drawn.</p>
</section>"""


def build() -> Path:
    md = (REPORTS / "REPORT.md").read_text()
    title, body = to_html(md)

    page = f"""<title>Sepsis Early Warning</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="{FONTS}">
<style>{CSS}{REPLAY_CSS}</style>

<main class="page">
  <header class="masthead">
    <span class="eyebrow">PhysioNet / CinC Challenge 2019 · results</span>
    <h1>Predicting sepsis six hours before the clinician does</h1>
    <p class="standfirst">Hourly risk scoring across 40,336 ICU admissions from two
      independent health systems, scored on the challenge's own time-dependent
      clinical utility rather than on accuracy.</p>
    {TRACE}
    <div class="meta">
      <span>1,552,210 ICU hours</span>
      <span>345 causal features</span>
      <span>1.8% positive hours</span>
      <span>~90% of lab cells missing</span>
      <span>external site held back until final scoring</span>
    </div>
  </header>

  {readout()}

  <div class="body">
    {body}
  </div>

  <footer class="colophon">
    Generated by <code>sepsis evaluate</code>. Splits are at the level of an
    admission, never an ICU hour; hyperparameters, calibration maps, blend weights
    and decision thresholds were fixed on validation; hospital A's test split was
    scored once and hospital B was untouched until the final table. Confidence
    intervals come from a patient-level cluster bootstrap. Data: Reyna et al.,
    <em>Critical Care Medicine</em> 48(2), 2020, distributed by PhysioNet under
    ODbL v1.0 and downloaded at run time.
  </footer>
</main>
<script>{REPLAY_JS}</script>
"""
    # Escape every non-ASCII codepoint as a numeric entity. The published page is
    # wrapped in a head this script does not control, so it cannot rely on a
    # charset declaration -- and a mis-decoded UTF-8 em dash is a very visible bug.
    page = "".join(c if ord(c) < 128 else f"&#{ord(c)};" for c in page)

    out = REPORTS / "report.html"
    out.write_text(page, encoding="ascii")
    print(f"[html] wrote {out} ({len(page) / 1024:.0f} KB)")
    return out


if __name__ == "__main__":
    sys.exit(0 if build() else 1)
