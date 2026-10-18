#!/usr/bin/env python3
"""Figure 3: why a distribution on a fixed gene panel. (a) what each way of writing a cell keeps; (b) measured share of the likelihood
gain that survives rewriting real held-out cells in each representation (validation split).

python figures/fig3_representation.py --out paper/figs
Panel b values are copied from ~/cad_out/auto_cell/{norman,pbmc}_test_repr_ceiling.json (src/repr_ceiling.py, 15 Oct 2026); panel a is
the content of Proposition 1 in the paper, not a measurement. rank-2048 equals rank-200 (only non-zero genes enter a sentence) and is not drawn.
"""
import argparse
import sys
from pathlib import Path

from matplotlib.patches import Circle, Wedge, Rectangle

SKILL = Path.home() / '.claude' / 'skills' / 'paper-schematic-figures' / 'scripts'
sys.path.insert(0, str(SKILL))
from figkit import kit  # noqa: E402
from figkit.kit import (BLUE, BLUE_FILL, TEAL, TEAL_FILL, PURPLE, PURPLE_FILL, ORANGE,  # noqa: E402
                        GREY, GREY_FILL, INK, MODULE, NOTE, SMALL)
import matplotlib  # noqa: E402

NAME = 'fig3_representation'
H = 76.0

# (label, example text, magnitude kept, likelihood on counts, invertible with library size)  2 = yes, 1 = partly, 0 = no
ROWS = (
    ('Rank sentence', 'GENE-A GENE-B GENE-C …', 0, 0, 0),
    ('Value bins', 'GENE-A:48  GENE-B:47 …', 1, 0, 0),
    ('Mass levels (top 48)', 'GENE-A:9  GENE-B:8 …', 1, 0, 0),
    ('Frequency + library size', r'$p=x/N$,  $N$', 2, 2, 2),
)
# measured, TEST split, share of the likelihood gain of the frequency vector kept, matched estimator
# (src/repr_ceiling.py, headroom_kept_matched in {norman,pbmc}_test_repr_ceiling_v2.json)
SHARE = (  # (representation, Norman, PBMC), same row order as panel a
    ('Rank sentence', 38, 57),
    ('Value bins', 73, 58),
    ('Mass levels', 64, 57),
    ('Frequency', 100, 100),
)
# C2S criterion for the same rank reconstruction: R^2 of log1p(CP10k) on log rank, held-out cells (src/rank_r2.py)
RANK_R2 = (0.68, 0.74)


def mark(d, x, y, level, colour):
    r = 1.6
    if level == 2:
        d.ax.add_patch(Circle((x, d.Y(y)), r, facecolor=colour, edgecolor=colour, lw=.6, zorder=6))
    elif level == 1:
        d.ax.add_patch(Circle((x, d.Y(y)), r, facecolor='#FFFFFF', edgecolor=colour, lw=.6, zorder=5))
        d.ax.add_patch(Wedge((x, d.Y(y)), r, 90, 270, facecolor=colour, edgecolor=colour, lw=.6, zorder=6))
    else:
        d.ax.add_patch(Circle((x, d.Y(y)), r, facecolor='#FFFFFF', edgecolor=GREY, lw=.6, zorder=5))


def compose():
    c, d, font = kit.new_figure(H)
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    kit.panel(c, d, 1.0, 1.0, 100.0, 72.0, 'a', 'What each representation keeps')
    kit.panel(c, d, 106.0, 1.0, 73.0, 72.0, 'b', 'Likelihood gain kept after rewriting real cells')

    # -- panel a: a table of marks ----------------------------------------------
    x_name, x_ex = 5.0, 5.0
    cols = (62.0, 77.0, 92.0)
    heads = ('magnitude', 'likelihood', 'invertible')
    subs = ('kept', 'on counts', 'with $N$')
    for cx, h1, h2 in zip(cols, heads, subs):
        d.label(cx, 14.0, h1, NOTE, color=INK, weight='bold')
        d.label(cx, 17.4, h2, SMALL, color=GREY)
    d.ax.plot([5.0, 99.0], [d.Y(20.2)] * 2, color=GREY, lw=.4, zorder=3)
    y0, dy = 27.0, 10.5
    for k, (name, ex, m, lk, inv) in enumerate(ROWS):
        y = y0 + k * dy
        ours = k == len(ROWS) - 1
        edge, fill = (TEAL, TEAL_FILL) if ours else (GREY, GREY_FILL)
        d.box(3.0, y - 4.6, 98.0, 9.2, '', edge=edge, fill=fill, lw=.7 if ours else .5)
        d.label(x_name, y - 1.5, name, MODULE, color=TEAL if ours else INK, weight='bold' if ours else 'normal', ha='left')
        d.label(x_ex, y + 2.0, ex, SMALL, color=GREY, ha='left')
        for cx, lv in zip(cols, (m, lk, inv)):
            mark(d, cx, y, lv, TEAL if ours else INK)
    # legend of marks
    ly = y0 + len(ROWS) * dy + 1.0
    mark(d, 7.0, ly, 2, INK); d.label(10.0, ly, 'yes', SMALL, ha='left')
    mark(d, 22.0, ly, 1, INK); d.label(25.0, ly, 'coarse', SMALL, ha='left')
    mark(d, 40.0, ly, 0, INK); d.label(43.0, ly, 'no', SMALL, ha='left')

    # -- panel b: grouped horizontal bars ---------------------------------------
    bx0, bw_max = 138.0, 28.0
    d.label(bx0, 14.0, '0', SMALL, color=GREY)
    d.label(bx0 + bw_max, 14.0, '100%', SMALL, color=GREY)
    d.ax.plot([bx0, bx0], [d.Y(16.0), d.Y(62.0)], color=GREY, lw=.5, zorder=3)
    gy, gdy, bh = 22.0, 9.6, 3.4
    for k, (name, nor, pb) in enumerate(SHARE):
        y = gy + k * gdy
        d.label(bx0 - 1.8, y + 2.2, name, NOTE, ha='right')
        for j, (v, col, fill) in enumerate(((nor, BLUE, BLUE_FILL), (pb, TEAL, TEAL_FILL))):
            yy = y + j * (bh + .6)
            d.ax.add_patch(Rectangle((bx0, d.Y(yy + bh)), bw_max * v / 100.0, bh, facecolor=fill, edgecolor=col, lw=.5, zorder=4))
            d.label(bx0 + bw_max * v / 100.0 + 1.0, yy + bh / 2, f'{v}%', SMALL, color=col, ha='left')
            if k == 0:   # the rank row: the reconstruction looks faithful by the C2S criterion
                d.label(bx0 + bw_max * v / 100.0 + 8.5, yy + bh / 2, f'($R^2$ {RANK_R2[j]:.2f})', SMALL, color=GREY, ha='left')
    ly = 68.0
    d.ax.add_patch(Rectangle((109.0, d.Y(ly + 2.0)), 3.0, 2.0, facecolor=BLUE_FILL, edgecolor=BLUE, lw=.5, zorder=4))
    d.label(113.0, ly + 1.0, 'Norman perturbations', SMALL, ha='left')
    d.ax.add_patch(Rectangle((142.0, d.Y(ly + 2.0)), 3.0, 2.0, facecolor=TEAL_FILL, edgecolor=TEAL, lw=.5, zorder=4))
    d.label(146.0, ly + 1.0, 'PBMC cell types', SMALL, ha='left')
    return c, font


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='paper/figs')
    a = ap.parse_args()
    c, font = compose()
    caption = ('**A rank sentence reconstructs expression rank well but keeps little of the likelihood.** (a) What each way of writing a cell '
               'keeps (Observation 1). (b) Test data: real held-out cells were rewritten in each representation and reconstructed; bars give the '
               'share of the likelihood gain of the cells\' own frequency vectors over the reference (perturbed mean for Norman, cell-type mean '
               'for PBMC), all estimated the same way. In grey, the C2S criterion for the same rank reconstruction ($R^2$ of log expression on '
               'log rank). Ceilings from decoding real cells, not model results.')
    kit.finish(c, a.out, NAME, caption, {'font_file': font, 'data_source': 'panel b: ~/cad_out/auto_cell/{norman,pbmc}_test_repr_ceiling_v2.json '
                                         '(headroom_kept_matched); R^2: auto_rev2/rank_r2_{norman,pbmc}_test.json',
                                         'model_trained': False, 'panel_a': 'Observation 1, not a measurement'})


if __name__ == '__main__':
    main()
