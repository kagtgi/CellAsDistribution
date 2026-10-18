#!/usr/bin/env python3
"""Figure 1: abstract pipeline. cell -> distribution -> decision language model -> sampled cell, two use cases.

python figures/fig1_overview.py --out paper/figs
Every drawn path exists in src/ (cell_dist.py, train_cell.py, cell_eval.py). Bar heights are illustrative (schematic), not data.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import Rectangle

SKILL = Path.home() / '.claude' / 'skills' / 'paper-schematic-figures' / 'scripts'
sys.path.insert(0, str(SKILL))
from figkit import kit  # noqa: E402
from figkit.kit import (BLUE, BLUE_FILL, PURPLE, PURPLE_FILL, TEAL, TEAL_FILL, ORANGE,  # noqa: E402
                        GREY, GREY_FILL, INK, MODULE, NOTE, SMALL)

import matplotlib  # noqa: E402
NAME = 'fig1_overview'
H = 68.0
RNG = np.random.default_rng(3)


def bars(d, x, y, w, h, heights, edge, fill, sparse=False):
    """A schematic bar glyph: one bar per (illustrative) gene, baseline at y + h."""
    n = len(heights)
    bw = w / n
    for k, v in enumerate(heights):
        bh = max(v * h, 0.25)
        d.ax.add_patch(Rectangle((x + k * bw + .12, d.Y(y + h)), bw - .24, bh, facecolor=fill, edgecolor=edge,
                                 lw=.3, zorder=4))
    d.ax.plot([x, x + w], [d.Y(y + h)] * 2, color=edge, lw=.5, zorder=5)
    return x, y, w, h


def compose():
    c, d, font = kit.new_figure(H)
    matplotlib.rcParams['mathtext.fontset'] = 'cm'   # math in TeX Computer Modern (set after the canvas resets rcParams)
    kit.single_panel(c)
    cols = (4.0, 50.0, 104.0, 148.0)
    steps = (('Cell as a distribution', BLUE), ('Decision language model', TEAL), ('Generate a cell', PURPLE),
             ('Evaluate', ORANGE))
    for n, (x, (t, col)) in enumerate(zip(cols, steps), 1):
        kit.step(d, x, 5.0, n, t, col)
    top = 12.0

    # -- 1: counts -> normalized distribution ---------------------------------
    n_g = 24
    counts = np.clip(RNG.gamma(0.45, 1.0, n_g), 0, None)
    counts = counts / counts.max()
    w1 = 38.0
    d.label(cols[0] + w1 / 2, top + 1.0, 'cell $x$: UMI counts over $G$ genes', NOTE, color=BLUE)
    bars(d, cols[0], top + 4.0, w1, 9.0, counts, BLUE, BLUE_FILL)
    kit.arrow(d, [(cols[0] + w1 / 2, top + 14.5), (cols[0] + w1 / 2, top + 21.0)], 'measurement', 'normalize', side='right')
    p = counts / counts.sum() * 4.0
    bars(d, cols[0], top + 22.0, w1, 9.0, p, BLUE, '#BBD0EA')
    d.label(cols[0] + w1 / 2, top + 34.0, r'$p=x/\sum_g x_g\in\Delta^{G-1}$', MODULE, color=INK)
    d.label(cols[0] + w1 / 2, top + 39.5, 'one distribution per cell', NOTE, color=BLUE)
    d.label(cols[0] + w1 / 2, top + 47.0, 'also: smoothed, log1p, sqrt', SMALL, color=GREY)

    # -- 2: text state -> decision language model -> decision head -------------
    cx, cw = cols[1], 44.0
    d.box(cx, top, cw, 8.0, '', edge=TEAL, fill=TEAL_FILL)
    d.label(cx + cw / 2, top + 2.6, r'use case 1: cell type', NOTE, color=TEAL, weight='bold')
    d.label(cx + cw / 2, top + 5.6, '"CD14 monocyte"', NOTE)
    d.box(cx, top + 10.0, cw, 11.0, '', edge=TEAL, fill=TEAL_FILL)
    d.label(cx + cw / 2, top + 12.6, 'use case 2: perturbation', NOTE, color=TEAL, weight='bold')
    d.label(cx + cw / 2, top + 15.6, '"KLF1" + control cell $x_0$', NOTE)
    d.label(cx + cw / 2, top + 18.6, '(top genes, log-mass levels)', SMALL, color=GREY)
    kit.arrow(d, [(cx + cw / 2, top + 21.0), (cx + cw / 2, top + 26.5)], 'conditioning')
    d.box(cx, top + 26.5, cw, 9.0, 'language model + LoRA', edge=PURPLE, fill=PURPLE_FILL, size=MODULE)
    kit.arrow(d, [(cx + cw / 2, top + 35.5), (cx + cw / 2, top + 40.0)], 'inference')
    d.box(cx, top + 40.0, cw, 8.0, 'decision head: softmax over $G$', edge=PURPLE, fill=PURPLE_FILL, size=MODULE)
    kit.arrow(d, [(cols[0] + w1 + 1.0, top + 27.0), (cx - 3.0, top + 27.0), (cx - 3.0, top + 44.0), (cx - .5, top + 44.0)], 'supervision')
    d.label(cx - 4.2, top + 43.0, 'soft CE', NOTE, color=ORANGE, ha='right')

    # -- 3: q -> sampler -> generated cell -------------------------------------
    gx, gw = cols[2], 38.0
    kit.arrow(d, [(cx + cw + .5, top + 44.0), (gx - 3.0, top + 44.0), (gx - 3.0, top + 12.0), (gx - .5, top + 12.0)], 'inference')
    q = counts * 0.6 + 0.2 * RNG.random(n_g)
    d.label(gx + gw / 2, top + 1.0, r'output $q(\cdot\mid\mathrm{state})$', NOTE, color=PURPLE)
    bars(d, gx, top + 4.0, gw, 9.0, q / q.max(), PURPLE, '#CFC7EA')
    d.label(gx + gw / 2, top + 16.0, 'exactly normalized', SMALL, color=GREY)
    kit.arrow(d, [(gx + gw / 2, top + 18.0), (gx + gw / 2, top + 23.5)], 'inference', 'sample', side='right')
    d.box(gx, top + 23.5, gw, 9.0, r'Mult$(s,q)$, library size $s$', edge=GREY, fill=GREY_FILL, size=MODULE)
    kit.arrow(d, [(gx + gw / 2, top + 32.5), (gx + gw / 2, top + 37.5)], 'inference')
    g = RNG.poisson(np.clip(q * 6, 0, None)).astype(float)
    bars(d, gx, top + 38.5, gw, 9.0, g / max(g.max(), 1), PURPLE, '#CFC7EA')
    d.label(gx + gw / 2, top + 51.0, r'generated cell $\tilde x$', MODULE, color=PURPLE)

    # -- 4: evaluation ---------------------------------------------------------
    ex, ew = cols[3], c.w - cols[3] - 4.0
    for k, t in enumerate(('held-out NLL', 'MMD, real-vs-gen AUC', 'UMAP vs real')):
        d.check(ex, top + k * 8.0, ew, 6.4, t, ORANGE, size=SMALL)
    by = top + 31.0
    d.box(ex, by, ew, 14.0, '', edge=ORANGE, fill='#FFFFFF', dashed=True)
    d.label(ex + ew / 2, by + 4.5, 'held-out real cells', NOTE, color=ORANGE)
    d.label(ex + ew / 2, by + 9.5, '(unseen donors /', SMALL, color=ORANGE)
    d.label(ex + ew / 2, by + 12.2, 'perturbations)', SMALL, color=ORANGE)
    kit.arrow(d, [(ex + ew / 2, by), (ex + ew / 2, top + 25.4)], 'supervision', 'score', side='right')
    kit.arrow(d, [(gx + gw + 1.0, top + 43.0), (ex - 3.0, top + 43.0), (ex - 3.0, top + 3.2), (ex - .5, top + 3.2)], 'measurement')

    kit.arrow_legend(c, d, H - 3.5)
    return c, font


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='paper/figs')
    a = ap.parse_args()
    c, font = compose()
    caption = ('**A cell is a distribution, and a decision language model decodes it.** (1) A cell\'s UMI count vector is normalized '
               'into a probability vector on the gene simplex (alternatives: smoothed, log1p, square-root). (2) A language model '
               'with a softmax decision head reads a text state, either a cell type (conditional generation) or a perturbation '
               'together with a control cell (perturbation response), and outputs an exactly normalized distribution $q$; '
               'training matches $q$ to the distribution of a real cell by soft cross-entropy (dashed). (3) Counts of a generated '
               'cell are drawn from a multinomial with a library size. (4) Generated cells are scored against held-out real cells '
               'of unseen donors or perturbations. Bar heights are schematic, not data.')
    kit.finish(c, a.out, NAME, caption, {'font_file': font, 'data_source': 'schematic', 'model_trained': True})


if __name__ == '__main__':
    main()
