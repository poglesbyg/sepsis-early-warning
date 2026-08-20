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


def build() -> Path:
    md = (REPORTS / "REPORT.md").read_text()
    title, body = to_html(md)

    page = f"""<title>Sepsis Early Warning</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="{FONTS}">
<style>{CSS}</style>

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
