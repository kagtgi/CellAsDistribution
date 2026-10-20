#!/usr/bin/env python3
"""Figure 3 (rev3): how much of the likelihood gain survives rewriting real held-out cells in a representation and decoding it
back to gene frequencies, for two decoders of the same rank sentence.

python figures/fig3_representation.py --out paper/figs
Values: headroom_kept_matched in paper/auto_rev3/{norman,pbmc}_test_ceiling_v3.txt (src/repr_ceiling.py, test split, rev3).
'C2S log-rank map' = the published reconstruction; 'learned map' = E[frequency | gene, rank-position bin] fit on training cells
(rev3, added after the test results; post hoc). Drawn at the text width (165 mm) so that no label is scaled below print size.
"""
import argparse
import sys
from pathlib import Path

from matplotlib.patches import Rectangle

SKILL = Path.home() / '.claude' / 'skills' / 'paper-schematic-figures' / 'scripts'
sys.path.insert(0, str(SKILL))
from figkit import kit  # noqa: E402
from figkit.kit import BLUE, BLUE_FILL, TEAL, TEAL_FILL, GREY, INK  # noqa: E402

NAME = 'fig3_representation'
W, H = 165.0, 52.0
LAB, TXT = 9.0, 8.0          # pt at print size (the figure is included at 100%)

# (representation, decoder, Norman %, PBMC %), matched estimator, test split
SHARE = (
    ('Rank sentence', 'C2S log-rank map', 38, 57),
    ('Rank sentence', 'learned map', 88, 58),
    ('Value bins (51)', 'bin means', 73, 58),
    ('Mass levels (top 48)', 'level centres', 64, 56),
    ('Frequency vector', 'none needed', 100, 100),
)
RANK_R2 = (0.68, 0.74)       # C2S criterion for the log-rank map: R^2 of log expression on log rank (src/rank_r2.py)


def compose():
    c, d, font = kit.new_figure(H, width_mm=W)
    kit.single_panel(c)
    x0, bw = 50.0, 82.0                       # bar origin and length of 100%
    d.label(x0, 4.0, '0', TXT, color=GREY)
    d.label(x0 + bw / 2, 4.0, '50%', TXT, color=GREY)
    d.label(x0 + bw, 4.0, '100%', TXT, color=GREY)
    for f in (0.0, 0.5, 1.0):
        d.ax.plot([x0 + bw * f] * 2, [d.Y(6.5), d.Y(44.0)], color=GREY, lw=.4 if f else .6, ls='-' if f == 0 else ':', zorder=3)
    y0, dy, bh = 8.5, 7.0, 2.7
    for k, (rep, dec, nor, pb) in enumerate(SHARE):
        y = y0 + k * dy
        ours = k == len(SHARE) - 1
        d.label(x0 - 2.0, y + 1.3, rep, LAB, color=TEAL if ours else INK, weight='bold' if ours else 'normal', ha='right')
        d.label(x0 - 2.0, y + 4.6, dec, TXT, color=GREY, ha='right')
        for j, (v, col, fill) in enumerate(((nor, BLUE, BLUE_FILL), (pb, TEAL, TEAL_FILL))):
            yy = y + j * (bh + .5)
            d.ax.add_patch(Rectangle((x0, d.Y(yy + bh)), bw * v / 100.0, bh, facecolor=fill, edgecolor=col, lw=.6, zorder=4))
            t = f'{v}%' + (f'   ($R^2$ = {RANK_R2[j]:.2f})' if k == 0 else '')
            d.label(x0 + bw * v / 100.0 + 1.2, yy + bh / 2, t, TXT, color=col, ha='left')
    ly = 48.5
    d.ax.add_patch(Rectangle((x0, d.Y(ly + 1.2)), 3.2, 2.4, facecolor=BLUE_FILL, edgecolor=BLUE, lw=.6, zorder=4))
    d.label(x0 + 4.5, ly, 'Norman K562 perturbations', TXT, ha='left')
    d.ax.add_patch(Rectangle((x0 + 48.0, d.Y(ly + 1.2)), 3.2, 2.4, facecolor=TEAL_FILL, edgecolor=TEAL, lw=.6, zorder=4))
    d.label(x0 + 52.5, ly, 'PBMC cell types', TXT, ha='left')
    return c, font


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='paper/figs')
    a = ap.parse_args()
    c, font = compose()
    caption = ('**The likelihood lost by a rank sentence depends on how it is decoded.** Real held-out test cells rewritten in each '
               'representation and decoded back to gene frequencies; bars give the share of the likelihood gain of the cells\' own '
               'frequency vectors over the reference that survives. The C2S log-rank map keeps 38% on Norman although it reconstructs log '
               'expression with R^2 0.68; a map learned on training cells keeps 88%.')
    kit.finish(c, a.out, NAME, caption, {'font_file': font, 'data_source': 'paper/auto_rev3/{norman,pbmc}_test_ceiling_v3.txt '
                                         '(headroom_kept_matched); R^2: paper/auto_rev2/rank_r2_{norman,pbmc}_test.json',
                                         'model_trained': False})


if __name__ == '__main__':
    main()
