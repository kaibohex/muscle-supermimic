"""The README's "time to 0.83" figure (fast_training_light.svg, fast_training_dark.svg) from the project page's chart
data (static/data/charts.js of kaibohex/muscle-supermimic-page, mshuman_speed): the same bars the page draws in its MS-Human-700 section, the
paper's Fig. 3 and tab:ladder (every bar the mean of three seeds, labelled as the paper prints it).

  python docs/assets/fast_training_svg.py [charts.js]   # default: the page's repository checked out next to this one
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT.parent / "muscle-supermimic-page/static/data/charts.js"
OUT = ROOT / "docs/assets"

THEMES = {
    "light": {"text": "#1a1a1a", "sub": "#5f5e5a", "tick": "#898781", "grid": "#e1e0d9", "zero": "#c3c2b7",
              "base": "#2a78d6", "theirs": "#1baf7a", "ours": "#eb6834"},
    "dark": {"text": "#f0efec", "sub": "#c3c2b7", "tick": "#898781", "grid": "#2c2c2a", "zero": "#4a4a46",
             "base": "#3987e5", "theirs": "#199e70", "ours": "#d95926"},
}
X0, PX_PER_H, ROW, BAR = 300.0, 91.12, 44, 20
FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"


def t_label(shown):
    """The mean as the paper prints it, without its standard deviation: '3.9 ± 1.0 h' -> '3.9 h'."""
    mean, unit = shown.split(" ± ")[0].split(" ")[0], shown.split(" ")[-1]
    return f"{mean} {unit}"


def svg(sp, th):
    n = len(sp["minutes"])
    notes = {n - 2: "37× sooner than the first release", n - 1: "the same checkpoints, refined"}  # the paper's ratio
    axis_bottom = 48 + ROW * n - 8
    aria = ("Training time to a held-out normalized return of 0.83 on MS-Human-700, step by step, each the mean of three "
            "seeds: " + "; ".join(f"{l} ({s}) {t_label(v)}" for l, s, v in zip(sp["label"], sp["sub"], sp["shown"]))
            + "; 37 times sooner than the first release at 50 Hz")
    color = {"base": th["base"], "theirs": th["theirs"], "ours": th["ours"], "both": "url(#both)"}
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="860" height="{axis_bottom + 56}" '
         f'viewBox="0 0 860 {axis_bottom + 56}" font-family="{FONT}" role="img" aria-label="{aria}">',
         '<defs><pattern id="both" patternUnits="userSpaceOnUse" width="8" height="8" patternTransform="rotate(45)">'
         f'<rect width="4" height="8" fill="{th["theirs"]}"/><rect x="4" width="4" height="8" fill="{th["ours"]}"/>'
         '</pattern></defs>',
         f'<text x="288" y="16" text-anchor="end" font-size="13" font-weight="600" fill="{th["text"]}">Time to 0.83 held-out</text>',
         f'<text x="300" y="16" font-size="12" fill="{th["sub"]}">MS-Human-700, 108 held-out clips, one RTX 5090; '
         'each bar the mean of three seeds</text>']
    for h in range(5):
        x = X0 + PX_PER_H * h
        if h:
            o.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="34" y2="{axis_bottom}" stroke="{th["grid"]}" stroke-width="1"/>')
        o.append(f'<text x="{x:.1f}" y="{axis_bottom + 16}" text-anchor="middle" font-size="11" fill="{th["tick"]}">'
                 f'{h if h == 0 else f"{h} h"}</text>')
    o.append(f'<line x1="{X0:.1f}" x2="{X0:.1f}" y1="34" y2="{axis_bottom}" stroke="{th["zero"]}" stroke-width="1"/>')
    for i, m in enumerate(sp["minutes"]):
        y, x1 = 48 + ROW * i, X0 + PX_PER_H * m / 60
        o.append(f'<path d="M{X0:.1f},{y:.1f}H{x1 - 4:.1f}A4,4 0 0 1 {x1:.1f},{y + 4:.1f}V{y + BAR - 4:.1f}'
                 f'A4,4 0 0 1 {x1 - 4:.1f},{y + BAR:.1f}H{X0:.1f}Z" fill="{color[sp["source"][i]]}"/>')
        o.append(f'<text x="288" y="{y + 8:.1f}" text-anchor="end" font-size="13" font-weight="500" fill="{th["text"]}">'
                 f'{sp["label"][i]}</text>')
        o.append(f'<text x="288" y="{y + 22:.1f}" text-anchor="end" font-size="11" fill="{th["sub"]}">{sp["sub"][i]}</text>')
        step = f"  {notes[i]}" if i in notes else ""
        tspan = f'<tspan font-weight="400" fill="{th["sub"]}">{step}</tspan>' if step else ""
        o.append(f'<text x="{x1 + 8:.1f}" y="{y + 14:.1f}" font-size="12" font-weight="600" fill="{th["text"]}">'
                 f'{t_label(sp["shown"][i])}{tspan}</text>')
    ly = axis_bottom + 31
    for x, name, fill in ((300, "first release", th["base"]), (421, "from InstantTrack / BlinkMimic", th["theirs"]),
                          (647, "ours", th["ours"]), (712, "both", "url(#both)")):
        o.append(f'<rect x="{x}" y="{ly}" width="10" height="10" rx="2" fill="{fill}"/>')
        o.append(f'<text x="{x + 16}" y="{ly + 9}" font-size="12" fill="{th["sub"]}">{name}</text>')
    o.append("</svg>")
    return "\n".join(o) + "\n"


def main():
    s = Path(sys.argv[1] if len(sys.argv) > 1 else DATA).read_text().strip().rstrip(";")
    sp = json.loads(s[s.index("{"):])["mshuman_speed"]
    for name, th in THEMES.items():
        (OUT / f"fast_training_{name}.svg").write_text(svg(sp, th))
        print(OUT / f"fast_training_{name}.svg")


if __name__ == "__main__":
    main()
