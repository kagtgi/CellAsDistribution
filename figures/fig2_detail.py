#!/usr/bin/env python3
"""Figure 2: detailed overview of the decoder, its training and its evaluation (every block exists in src/train_cell.py,
src/cell_dist.py, src/cell_eval.py). Math is set in TeX Computer Modern.

python figures/fig2_detail.py --out paper/figs
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.patches import Rectangle

SKILL = Path.home() / '.claude' / 'skills' / 'paper-schematic-figures' / 'scripts'
sys.path.insert(0, str(SKILL))
from figkit import kit  # noqa: E402
from figkit.kit import (BLUE, BLUE_FILL, PURPLE, PURPLE_FILL, TEAL, TEAL_FILL, ORANGE,  # noqa: E402
                        GREY, GREY_FILL, MODULE, NOTE, SMALL)

NAME = 'fig2_detail'
H = 92.0
RNG = np.random.default_rng(5)


def bars(d, x, y, w, h, heights, edge, fill):
    heights = np.asarray(heights) / np.max(heights)
    n = len(heights)
    bw = w / n
    for k, v in enumerate(heights):
        d.ax.add_patch(Rectangle((x + k * bw + .1, d.Y(y + h)), bw - .2, max(v * h, .25), facecolor=fill, edgecolor=edge,
                                 lw=.3, zorder=4))
    d.ax.plot([x, x + w], [d.Y(y + h)] * 2, color=edge, lw=.5, zorder=5)


def compose():
    c, d, font = kit.new_figure(H)
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    kit.single_panel(c)
    cols = (4.0, 54.0, 106.0, 146.0)
    heads = (('State text', TEAL), ('Decision language model', PURPLE), ('Training target', BLUE), ('Generate and score', ORANGE))
    for n, (x, (t, col)) in enumerate(zip(cols, heads), 1):
        kit.step(d, x, 5.0, n, t, col)
    top = 12.0

    # -- 1 state text ----------------------------------------------------------
    w = 44.0
    d.box(cols[0], top, w, 15.0, '', edge=TEAL, fill=TEAL_FILL)
    d.label(cols[0] + w / 2, top + 2.8, 'use case 1: conditional', NOTE, color=TEAL, weight='bold')
    d.label(cols[0] + w / 2, top + 7.0, 'Cell type: CD14 monocyte', NOTE)
    d.label(cols[0] + w / 2, top + 11.0, '(held-out donors at test)', SMALL, color=GREY)
    y2 = top + 19.0
    d.box(cols[0], y2, w, 15.0, '', edge=TEAL, fill=TEAL_FILL)
    d.label(cols[0] + w / 2, y2 + 2.8, 'use case 2: perturbation', NOTE, color=TEAL, weight='bold')
    d.label(cols[0] + w / 2, y2 + 7.0, 'Perturbation: KLF1', NOTE)
    d.label(cols[0] + w / 2, y2 + 11.0, '+ control cell text', NOTE)
    cy = y2 + 20.0
    d.label(cols[0] + w / 2, cy, r'control cell $x_0$ (random)', NOTE, color=BLUE)
    bars(d, cols[0], cy + 2.0, w, 8.0, RNG.gamma(.45, 1, 22) / 3, BLUE, BLUE_FILL)
    kit.arrow(d, [(cols[0] + w / 2, cy + 11.5), (cols[0] + w / 2, cy + 16.0)], 'measurement', 'top-48 genes', side='right')
    d.box(cols[0], cy + 16.0, w, 8.0, '', edge=GREY, fill=GREY_FILL)
    # real example: first tokens of a serialized training control cell (paper/auto_rev2/biology_checks.json)
    d.label(cols[0] + w / 2, cy + 19.0, 'HBZ:10 CKS1B:9 SET:9 ...', SMALL)
    d.label(cols[0] + w / 2, cy + 22.2, r'level $\lfloor\log_2(1{+}10^4x_{0g}/N_0)\rfloor$', SMALL, color=GREY)
    kit.arrow(d, [(cols[0] + w + .5, top + 4.0), (cols[1] - .5, top + 4.0)], 'conditioning')
    kit.arrow(d, [(cols[0] + w + .5, cy + 20.0), (cols[1] - 2.0, cy + 20.0), (cols[1] - 2.0, top + 6.5), (cols[1] - .5, top + 6.5)],
              'conditioning')

    # -- 2 decision language model ---------------------------------------------
    bx, bw = cols[1], 46.0
    d.box(bx, top, bw, 8.0, 'tokenizer', edge=GREY, fill=GREY_FILL, size=MODULE)
    kit.arrow(d, [(bx + bw / 2, top + 8.0), (bx + bw / 2, top + 12.5)], 'inference')
    d.box(bx, top + 12.5, bw, 11.0, '', edge=PURPLE, fill=PURPLE_FILL)
    d.label(bx + bw / 2, top + 16.5, 'C2S-Scale-Gemma-2-2B', MODULE, weight='bold')
    d.label(bx + bw / 2, top + 20.2, 'LoRA, rank 16', NOTE)
    kit.arrow(d, [(bx + bw / 2, top + 23.5), (bx + bw / 2, top + 28.5)], 'inference')
    d.tokens(bx + bw / 2 - 6.0, top + 28.5, 3, 3.4, edge=PURPLE, fills=('#CFC7EA', '#B5AADB', '#CFC7EA'))
    d.label(bx + bw / 2, top + 35.5, r'last-token state $h$', NOTE, color=PURPLE)
    kit.arrow(d, [(bx + bw / 2, top + 37.5), (bx + bw / 2, top + 42.0)], 'inference')
    d.box(bx, top + 42.0, bw, 9.0, r'decision head: $z=Wh+\log b$', edge=PURPLE, fill=PURPLE_FILL, size=MODULE)
    d.label(bx + bw / 2, top + 55.0, r'prior $b$: none, training mean, or control cell', SMALL, color=GREY)
    kit.arrow(d, [(bx + bw / 2, top + 57.5), (bx + bw / 2, top + 62.0)], 'inference')
    d.box(bx, top + 62.0, bw, 8.0, r'softmax: $q\in\Delta^{G-1}$', edge=PURPLE, fill=PURPLE_FILL, size=MODULE)

    # -- 3 training target -----------------------------------------------------
    tx, tw = cols[2], 36.0
    d.label(tx + tw / 2, top + 1.5, r'real cell $x$ (train donor / perturbed)', NOTE, color=BLUE)
    bars(d, tx, top + 4.0, tw, 8.0, RNG.gamma(.45, 1, 22) / 3, BLUE, BLUE_FILL)
    kit.arrow(d, [(tx + tw / 2, top + 13.5), (tx + tw / 2, top + 19.0)], 'measurement', r'$T$', side='right')
    d.box(tx, top + 19.0, tw, 9.0, '', edge=BLUE, fill=BLUE_FILL)
    d.label(tx + tw / 2, top + 22.0, 'transformation', NOTE, color=BLUE, weight='bold')
    d.label(tx + tw / 2, top + 25.3, 'freq, smooth, log1p, sqrt', SMALL)
    kit.arrow(d, [(tx + tw / 2, top + 28.0), (tx + tw / 2, top + 33.5)], 'measurement')
    bars(d, tx, top + 35.0, tw, 8.0, RNG.gamma(.45, 1, 22) / 3, BLUE, '#BBD0EA')
    d.label(tx + tw / 2, top + 46.0, r'target $p=T(x)\in\Delta^{G-1}$', NOTE, color=BLUE)
    d.box(tx, top + 52.0, tw, 9.0, r'soft CE $-\sum_g p_g\log q_g$', edge=ORANGE, fill='#FFFFFF', size=MODULE, dashed=True)
    kit.arrow(d, [(tx + tw / 2, top + 48.5), (tx + tw / 2, top + 51.6)], 'supervision')
    kit.arrow(d, [(bx + bw + .5, top + 66.0), (tx + tw / 2, top + 66.0), (tx + tw / 2, top + 61.4)], 'supervision')
    d.label(tx + tw / 2, top + 65.0, '', NOTE)

    # -- 4 generate and score --------------------------------------------------
    ex, ew = cols[3], c.w - cols[3] - 4.0
    d.box(ex, top, ew, 10.0, '', edge=GREY, fill=GREY_FILL)
    d.label(ex + ew / 2, top + 3.2, r'$s\sim$ train library sizes', NOTE)
    d.label(ex + ew / 2, top + 7.0, r'Mult$(s,q)$ or DM$(s,\kappa q)$', NOTE)
    kit.arrow(d, [(ex + ew / 2, top + 10.0), (ex + ew / 2, top + 15.0)], 'inference')
    d.label(ex + ew / 2, top + 17.2, r'generated cell $\tilde x$', NOTE, color=PURPLE)
    bars(d, ex, top + 19.5, ew, 7.0, RNG.gamma(.45, 1, 18) / 3, PURPLE, '#CFC7EA')
    kit.arrow(d, [(ex + ew / 2, top + 28.0), (ex + ew / 2, top + 33.0)], 'measurement', 'score', side='right')
    for k, t in enumerate(('NLL, cell NLL', 'MMD, AUC', r'$r_\mu$, type acc.', 'UMAP')):
        d.check(ex, top + 34.0 + k * 7.0, ew, 5.6, t, ORANGE, size=SMALL)
    d.box(ex, top + 63.0, ew, 8.0, 'held-out real cells', edge=ORANGE, fill='#FFFFFF', dashed=True, size=SMALL, color=ORANGE)
    kit.arrow(d, [(ex + ew / 2, top + 63.0), (ex + ew / 2, top + 61.0)], 'supervision')

    # q -> sampler: route above the training column
    kit.arrow(d, [(bx + bw / 2 + 8.0, top + 70.0), (bx + bw / 2 + 8.0, top + 74.0), (ex - 1.8, top + 74.0), (ex - 1.8, top + 5.0), (ex - .5, top + 5.0)], 'inference')
    kit.arrow_legend(c, d, H - 3.5)
    return c, font


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='paper/figs')
    a = ap.parse_args()
    c, font = compose()
    caption = ('**Detailed overview of the decoder, its training target and its evaluation.** (1) The state is a text prompt: a cell type, '
               'or a perturbation with a randomly paired control cell serialized as its 48 most abundant genes with quantized log-mass levels. '
               '(2) A LoRA-tuned C2S-Scale-Gemma-2-2B encodes the prompt; a linear decision head with an optional fixed log-prior $b$ gives gene logits and a '
               'softmax gives the exactly normalized distribution $q$. (3) Training matches $q$ to the transformed distribution $T(x)$ of a real cell by '
               'soft cross-entropy. (4) A shared sampler (multinomial or Dirichlet-multinomial with a training-fit concentration) draws a generated cell, '
               'scored against held-out real cells by NLL, MMD, classifier AUC, gene-mean correlation, cell-type recovery and UMAP.')
    kit.finish(c, a.out, NAME, caption, {'font_file': font, 'data_source': 'schematic', 'model_trained': True})


if __name__ == '__main__':
    main()
