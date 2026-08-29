"""Render docs/writing/two-boundaries.md as a self-contained page for the site.

The essay argues about what confidence intervals do and do not establish, so its
figures are intervals: a point estimate, its bounds, and a marked zero. **They are
computed from the committed experiment artifacts, never hand-drawn.** A figure whose
geometry was typed in by hand is a claim nobody checked, which is the failure the
essay is about.

The Markdown source carries `<!-- figure:name -->` markers. They are invisible
wherever the Markdown is read directly and become figures here, so the prose stays
one file rather than drifting between a source and a rendering.

Standard library only, like the report builder: the workflow that publishes this
runs on a checkout with no dependencies installed.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_html_report import FONTS, to_html  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
SOURCE = ROOT / "docs" / "writing" / "two-boundaries.md"

# Geometry of an interval strip, in viewBox units.
LEFT, RIGHT, WIDTH = 62, 60, 520


def _unit_rows() -> dict[str, dict[str, float]]:
    with (REPORTS / "experiment_unit_transfer.csv").open() as fh:
        return {
            row["cohort"].split(" (")[0]: {k: _maybe_float(v) for k, v in row.items()}
            for row in csv.DictReader(fh)
        }


def _maybe_float(value: str):
    try:
        return float(value)
    except ValueError:
        return value


def _load(name: str) -> dict:
    return json.loads((REPORTS / f"experiment_{name}.json").read_text())


# --------------------------------------------------------------------------
# The one visual device: an estimate, its interval, and where zero sits
# --------------------------------------------------------------------------
def interval_figure(caption: str, rows: list[dict], note: str) -> str:
    """One axis, one row per estimate, zero always drawn.

    A row whose interval contains zero is drawn in the alert colour, because for
    every quantity in this essay crossing zero is what decides the reading:
    utility at zero is never alerting, and a difference at zero is no effect.
    """
    lows = [r["lo"] for r in rows] + [0.0]
    highs = [r["hi"] for r in rows] + [0.0]
    pad = (max(highs) - min(lows)) * 0.12 or 0.01
    lo, hi = min(lows) - pad, max(highs) + pad
    span = hi - lo

    def x(value: float) -> float:
        return LEFT + (value - lo) / span * (WIDTH - LEFT - RIGHT)

    height = 40 + 34 * len(rows)
    axis_y = height - 22
    parts = [
        f'<svg viewBox="0 0 {WIDTH} {height + 18}" role="img" aria-label="{caption}">',
        f'<line class="axis" x1="{LEFT}" y1="{axis_y}" x2="{WIDTH - RIGHT}" y2="{axis_y}" stroke-width="1"/>',
        f'<line class="zero" x1="{x(0):.1f}" y1="14" x2="{x(0):.1f}" y2="{axis_y}" '
        f'stroke-width="1" stroke-dasharray="3 3"/>',
        f'<text class="zerolab" x="{x(0):.1f}" y="{axis_y + 16}" font-size="10.5" '
        f'text-anchor="middle">0</text>',
    ]

    for n, row in enumerate(rows):
        y = 30 + 34 * n
        crosses = row["lo"] <= 0 <= row["hi"]
        tone = "crosses" if crosses else "clear"
        delay = f' style="animation-delay:{n * .12:.2f}s"' if n else ""
        parts += [
            f'<text class="lab" x="0" y="{y + 4}" font-size="11.5">{row["label"]}</text>',
            f'<line class="bar-{tone} grow" x1="{x(row["lo"]):.1f}" y1="{y}" '
            f'x2="{x(row["hi"]):.1f}" y2="{y}" stroke-width="2.5" stroke-linecap="round"{delay}/>',
            f'<circle class="dot-{tone} fade" cx="{x(row["point"]):.1f}" cy="{y}" r="4.5"{delay}/>',
            f'<text x="{WIDTH - RIGHT + 8}" y="{y + 4}" font-size="11.5">{row["point"]:+.3f}</text>',
        ]
    parts.append("</svg>")

    return (
        '<figure class="interval">'
        f'<figcaption>{caption}</figcaption>'
        f'<div class="interval-scroller">{"".join(parts)}</div>'
        f'<p class="note">{note}</p>'
        "</figure>"
    )


def figures() -> dict[str, object]:
    units = _unit_rows()
    micu, sicu = units["MICU"], units["SICU"]
    mech = _load("mechanism")

    def utility() -> str:
        return interval_figure(
            "Clinical utility at the MICU threshold &middot; 95% interval",
            [
                {"label": "MICU", "point": micu["utility_frozen"],
                 "lo": micu["utility_frozen_lo"], "hi": micu["utility_frozen_hi"]},
                {"label": "SICU", "point": sicu["utility_frozen"],
                 "lo": sicu["utility_frozen_lo"], "hi": sicu["utility_frozen_hi"]},
            ],
            f"The SICU interval runs from {sicu['utility_frozen_lo']:+.3f} to "
            f"{sicu['utility_frozen_hi']:+.3f}. It contains zero, so at the transferred "
            f"threshold this deployment is not distinguishable from never alerting at all.",
        )

    def gain() -> str:
        point = sicu["utility_local"] - sicu["utility_frozen"]
        return interval_figure(
            "Gain from a locally chosen threshold &middot; 95% interval",
            [{"label": "SICU", "point": point,
              "lo": sicu["gain_lo"], "hi": sicu["gain_hi"]}],
            f"Interval {sicu['gain_lo']:+.3f} to {sicu['gain_hi']:+.3f}, clear of zero. "
            f"The identical procedure on the MICU bucket, where the threshold was "
            f"already tuned, gives {micu['utility_local'] - micu['utility_frozen']:+.3f} "
            f"with an interval spanning zero: it only finds a recovery where one exists.",
        )

    def mechanism() -> str:
        ci = mech["gap_difference_ci"]
        return interval_figure(
            "Reduction in transfer gap when ordering is withheld &middot; 95% interval",
            [{"label": "A &rarr; B", "point": mech["gap_difference"],
              "lo": ci[0], "hi": ci[1]}],
            f"Interval {ci[0]:+.4f} to {ci[1]:+.4f}. It excludes zero and sits right "
            f"against it: the direction holds, the size is not established by this.",
        )

    return {
        "<!-- figure:utility -->": utility,
        "<!-- figure:gain -->": gain,
        "<!-- figure:mechanism -->": mechanism,
    }


CSS = """
:root {
  --paper:#f2f5f6; --surface:#ffffff; --ink:#14212a; --ink-soft:#45565f;
  --ink-faint:#7b8c95; --rule:#d5dde1; --rule-soft:#e6ecee;
  --signal:#0f6d8c; --flag:#b33a1f; --stable:#2f7d5c;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --paper:#0f161a; --surface:#161f25; --ink:#dde6e9; --ink-soft:#a0b2ba;
    --ink-faint:#71858e; --rule:#26343b; --rule-soft:#1c272d;
    --signal:#56b4d3; --flag:#e2795c; --stable:#62b48c;
  }
}
:root[data-theme="dark"] {
  --paper:#0f161a; --surface:#161f25; --ink:#dde6e9; --ink-soft:#a0b2ba;
  --ink-faint:#71858e; --rule:#26343b; --rule-soft:#1c272d;
  --signal:#56b4d3; --flag:#e2795c; --stable:#62b48c;
}
* { box-sizing: border-box; }
body {
  margin:0; background:var(--paper); color:var(--ink);
  font-family:Spectral, Georgia, "Times New Roman", serif;
  font-size:18px; line-height:1.62; -webkit-font-smoothing:antialiased;
}
.page { max-width:42rem; margin:0 auto; padding:clamp(2.5rem,7vw,5.5rem) clamp(1.15rem,5vw,2rem) 5rem; }
.eyebrow {
  font-family:"IBM Plex Sans", system-ui, sans-serif; font-size:.72rem; font-weight:500;
  letter-spacing:.13em; text-transform:uppercase; color:var(--ink-faint); margin:0 0 1.4rem;
}
h1 {
  font-weight:300; font-size:clamp(2.1rem,6.4vw,3.35rem); line-height:1.08;
  letter-spacing:-.018em; text-wrap:balance; margin:0 0 1.5rem;
}
h1 .turn { color:var(--flag); font-style:italic; }
.standfirst { font-size:1.16rem; line-height:1.55; color:var(--ink-soft); margin:0 0 2.6rem; max-width:34rem; }
p { margin:0 0 1.15rem; max-width:38rem; }
strong { font-weight:600; }
h2 {
  font-family:"IBM Plex Sans", system-ui, sans-serif; font-size:1.02rem; font-weight:600;
  line-height:1.35; text-wrap:balance; margin:3.2rem 0 1.1rem; padding-top:1.1rem;
  border-top:1px solid var(--rule); max-width:38rem;
}
code {
  font-family:"IBM Plex Mono", ui-monospace, monospace;
  font-variant-numeric:tabular-nums; font-size:.88em;
}
a { color:var(--signal); text-decoration:none; border-bottom:1px solid var(--rule); padding-bottom:1px; }
a:hover { border-bottom-color:var(--signal); }
a:focus-visible { outline:2px solid var(--signal); outline-offset:3px; border-radius:2px; }
ul, ol { max-width:38rem; padding-left:1.15rem; margin:1.5rem 0; }
li { margin-bottom:.85rem; padding-left:.3rem; }
li::marker { font-family:"IBM Plex Mono", monospace; color:var(--ink-faint); font-size:.85em; }
figure.interval {
  margin:2.2rem 0; padding:1.15rem 0 1rem;
  border-top:1px solid var(--rule); border-bottom:1px solid var(--rule);
}
figure.interval figcaption {
  font-family:"IBM Plex Sans", system-ui, sans-serif; font-size:.74rem; font-weight:500;
  letter-spacing:.1em; text-transform:uppercase; color:var(--ink-faint); margin-bottom:.55rem;
}
.interval-scroller { overflow-x:auto; }
figure.interval svg { display:block; width:100%; height:auto; min-width:25rem; }
figure.interval .note { font-size:.92rem; line-height:1.5; color:var(--ink-soft); margin:.7rem 0 0; max-width:34rem; }
svg text { font-family:"IBM Plex Mono", ui-monospace, monospace; font-variant-numeric:tabular-nums; fill:var(--ink-soft); }
svg .lab { font-family:"IBM Plex Sans", system-ui, sans-serif; fill:var(--ink-soft); }
svg .axis { stroke:var(--rule); }
svg .zero { stroke:var(--flag); }
svg .zerolab { fill:var(--flag); }
.bar-crosses { stroke:var(--flag); } .dot-crosses { fill:var(--flag); }
.bar-clear { stroke:var(--stable); }  .dot-clear { fill:var(--stable); }
@media (prefers-reduced-motion: no-preference) {
  .grow { transform-origin:left center; animation:grow .55s cubic-bezier(.2,.7,.3,1) both; }
  .fade { animation:fade .5s ease-out .35s both; }
  @keyframes grow { from { transform:scaleX(0); } to { transform:scaleX(1); } }
  @keyframes fade { from { opacity:0; } to { opacity:1; } }
}
.scroller { overflow-x:auto; margin:1.9rem 0; }
table {
  border-collapse:collapse; width:100%; min-width:30rem;
  font-family:"IBM Plex Sans", system-ui, sans-serif; font-size:.88rem;
}
th, td { padding:.58rem .8rem .58rem 0; text-align:left; }
td.right, th.right {
  text-align:right; padding-right:1.4rem;
  font-family:"IBM Plex Mono", ui-monospace, monospace; font-variant-numeric:tabular-nums;
}
thead th {
  font-weight:500; font-size:.73rem; letter-spacing:.07em; text-transform:uppercase;
  color:var(--ink-faint); border-bottom:1px solid var(--rule);
}
tbody tr + tr td { border-top:1px solid var(--rule-soft); }
hr { border:0; border-top:1px solid var(--rule); margin:2.6rem 0; }
.colophon {
  margin-top:2.6rem; padding-top:1.3rem; border-top:1px solid var(--rule);
  font-family:"IBM Plex Sans", system-ui, sans-serif; font-size:.85rem;
  line-height:1.7; color:var(--ink-faint); display:flex; flex-direction:column; gap:.35rem;
}
"""


def build() -> Path:
    title, body = to_html(SOURCE.read_text(), hooks=figures())

    # The source's own H1 becomes ``title`` rather than body content, so the
    # headline below is the only one on the page.
    page = f"""<title>The Threshold Didn't Transfer</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="{FONTS.replace('0,400;0,600;1,400', '0,300;0,400;0,600;1,300;1,400')}">
<style>{CSS}</style>

<article class="page">
  <p class="eyebrow">Clinical ML &middot; transfer &amp; measurement</p>
  <h1>The model transferred.<br>The alert threshold <span class="turn">didn't</span>.</h1>
  <p class="standfirst">A sepsis early-warning model crossed from a medical ICU to a
  surgical ICU almost unchanged in behaviour, and the deployed configuration became
  statistically indistinguishable from switching it off.</p>
  {body}
  <div class="colophon">
    <span>Code, experiments and a model card &middot;
      <a href="https://github.com/poglesbyg/sepsis-early-warning">github.com/poglesbyg/sepsis-early-warning</a></span>
    <span>Full report, with an hour-by-hour replay of individual admissions &middot;
      <a href="../">the report</a></span>
  </div>
</article>
"""
    page = "".join(c if ord(c) < 128 else f"&#{ord(c)};" for c in page)
    out = REPORTS / "essay.html"
    out.write_text(page, encoding="ascii")
    print(f"[essay] wrote {out} ({len(page) / 1024:.0f} KB, title {title!r})")
    return out


if __name__ == "__main__":  # pragma: no cover
    sys.exit(0 if build() else 1)
