#!/usr/bin/env python3
"""Figure 1 (merged overview + detail): text state -> decision language model -> distribution on the gene simplex ->
shared sampler and evaluation. Replaces fig1_overview.py and fig2_detail.py.

python figures/fig1_method.py --out paper/figs
Provenance of every drawn element:
  prompts ............ src/train_cell.py prompt(); "CD14-positive monocyte" is a PBMC label (obs.csv), KLF1 a Norman target
  control-cell tokens  first tokens of a real serialized training control cell (paper/auto_rev2/biology_checks.json)
  level formula ...... src/cell_dist.py serialize_cell: floor(log2(1 + 1e4 x_0g / N_0)), N_0 = panel total of x_0
  decision head ...... src/train_cell.py: z = W h + log b, b in {none, training mean, smoothed control cell}
  target ............. src/cell_dist.py to_dist: freq (default), smooth, log1p, sqrt; soft cross-entropy in train_cell.py
  sampler ............ src/cell_dist.py sample_counts: Mult(s, q) or Dirichlet-multinomial(s, kappa q), s from train cells
  checks ............. src/cell_eval.py (NLL, cell NLL, MMD, AUC) and src/de_eval.py (delta, top-20 DE)
Bar heights are schematic, not data.
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
                        GREY, GREY_FILL)

# rev3: larger type than the kit default so that no label prints below 7 pt (RECOMB asks for >= 10 pt text; the figure
# is printed at 100% of the text width)
MODULE, NOTE, SMALL = 8.2, 7.6, 7.0
kit.HEAD, kit.NOTE = 8.6, 7.6

NAME = 'fig1_method'
H = 80.0
RNG = np.random.default_rng(7)
LIGHT_BLUE, LIGHT_PURPLE = '#BBD0EA', '#CFC7EA'


def bars(d, x, y, w, h, heights, edge, fill):
    """Schematic bar glyph (one bar per gene), baseline at y + h. Heights are illustrative."""
    v = np.asarray(heights, float)
    v = v / v.max()
    bw = w / len(v)
    for k, hk in enumerate(v):
        d.ax.add_patch(Rectangle((x + k * bw + .1, d.Y(y + h)), bw - .2, max(hk * h, .25), facecolor=fill,
                                 edgecolor=edge, lw=.3, zorder=4))
    d.ax.plot([x, x + w], [d.Y(y + h)] * 2, color=edge, lw=.5, zorder=5)


def compose():
    c, d, font = kit.new_figure(H, width_mm=165.0)   # = LaTeX text width (6.5 in), printed at 100%
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    kit.single_panel(c)
    X1, W1 = 3.0, 41.0
    X2, W2 = 49.0, 37.0
    X3, W3 = 91.0, 34.0
    X4 = 130.0
    W4 = c.w - X4 - 3.0
    for n, (x, t, col) in enumerate(((X1, 'Text state', TEAL), (X2, 'Decision language model', PURPLE),
                                     (X3, 'Cell as a distribution', BLUE), (X4, 'Sample and score', ORANGE)), 1):
        kit.step(d, x, 5.0, n, t, col)
    top = 12.0
    counts = RNG.gamma(.45, 1, 22)

    # ---- 1 text state --------------------------------------------------------------------------------------
    d.box(X1, top, W1, 10.0, '', edge=TEAL, fill=TEAL_FILL)
    d.label(X1 + W1 / 2, top + 3.0, 'cell-type generation', NOTE, color=TEAL, weight='bold')
    d.label(X1 + W1 / 2, top + 7.0, 'Cell type: CD14+ monocyte', NOTE)
    y2 = top + 13.0
    d.box(X1, y2, W1, 17.0, '', edge=TEAL, fill=TEAL_FILL)
    d.label(X1 + W1 / 2, y2 + 3.0, 'perturbation response', NOTE, color=TEAL, weight='bold')
    d.label(X1 + W1 / 2, y2 + 7.0, 'Perturbation: KLF1', NOTE)
    d.label(X1 + W1 / 2, y2 + 10.6, 'Control cell: HBZ:10 CKS1B:9 ...', NOTE)
    d.label(X1 + W1 / 2, y2 + 14.2, r'level $\lfloor\log_2(1+10^4x_{0g}/N_0)\rfloor$', SMALL, color=GREY)
    # the random control cell feeds the prompt (top-48 genes, serialized) and, as a residual, the head (log b)
    cy = y2 + 22.5
    bars(d, X1 + 6.0, cy, W1 - 18.0, 6.5, RNG.gamma(.45, 1, 16), BLUE, BLUE_FILL)
    d.label(X1 + 6.0 + (W1 - 18.0) / 2, cy + 9.4, r'random control cell $x_0$', NOTE, color=BLUE)
    kit.arrow(d, [(X1 + 6.0 + (W1 - 18.0) / 2, cy - .6), (X1 + 6.0 + (W1 - 18.0) / 2, y2 + 17.4)], 'measurement',
              'top-48 genes', side='right')
    ty = cy

    # ---- 2 decision language model ---------------------------------------------------------------------------
    lm_y = top
    d.box(X2, lm_y, W2, 13.0, '', edge=PURPLE, fill=PURPLE_FILL)
    d.label(X2 + W2 / 2, lm_y + 3.6, 'language model + LoRA', MODULE, weight='bold')
    d.label(X2 + W2 / 2, lm_y + 7.6, 'C2S-Scale-Gemma-2-2B', NOTE)
    d.label(X2 + W2 / 2, lm_y + 10.8, 'or Qwen3.5-0.8B', SMALL, color=GREY)
    kit.arrow(d, [(X2 + W2 / 2, lm_y + 13.0), (X2 + W2 / 2, lm_y + 17.5)], 'inference')
    d.tokens(X2 + W2 / 2 - 6.0, lm_y + 17.5, 3, 3.4, edge=PURPLE, fills=(LIGHT_PURPLE, '#B5AADB', LIGHT_PURPLE))
    d.label(X2 + W2 / 2, lm_y + 24.2, r'last-token state $h$', NOTE, color=PURPLE)
    kit.arrow(d, [(X2 + W2 / 2, lm_y + 26.0), (X2 + W2 / 2, lm_y + 30.5)], 'inference')
    hy = lm_y + 30.5
    d.box(X2, hy, W2, 12.0, '', edge=PURPLE, fill=PURPLE_FILL)
    d.label(X2 + W2 / 2, hy + 3.8, 'decision head', MODULE, weight='bold')
    d.label(X2 + W2 / 2, hy + 8.4, r'logits $z=Wh+\log b$', NOTE, color=GREY)
    kit.arrow(d, [(X2 + W2 / 2, hy + 12.0), (X2 + W2 / 2, hy + 16.5)], 'inference')
    sy = hy + 16.5
    d.box(X2, sy, W2, 10.0, '', edge=PURPLE, fill=PURPLE_FILL)
    d.label(X2 + W2 / 2, sy + 3.6, r'softmax  $q\in\Delta^{G-1}$', MODULE, weight='bold')
    d.label(X2 + W2 / 2, sy + 7.4, 'one pass, exactly normalized', SMALL, color=GREY)

    # conditioning arrows: both prompts into the language model; the control cell also into log b
    kit.arrow(d, [(X1 + W1 + .5, top + 5.0), (X2 - .5, top + 5.0)], 'conditioning')
    gx = X1 + W1 + 3.0
    kit.arrow(d, [(X1 + W1 + .5, y2 + 8.5), (gx, y2 + 8.5), (gx, top + 9.0), (X2 - .5, top + 9.0)], 'conditioning')
    by = cy + 3.5
    kit.arrow(d, [(X1 + W1 - 11.5, by), (X2 - .5, by)], 'conditioning')
    d.label(X1 + W1 - 4.0, by - 1.6, r'$\log b$', SMALL, color=TEAL)

    # ---- 3 cell as a distribution (training target) ---------------------------------------------------------
    d.label(X3 + W3 / 2, top + 1.0, r'real cell $x$ (UMI counts)', NOTE, color=BLUE)
    bars(d, X3, top + 3.5, W3, 8.0, counts, BLUE, BLUE_FILL)
    kit.arrow(d, [(X3 + W3 / 2, top + 12.5), (X3 + W3 / 2, top + 18.0)], 'measurement', r'normalize, $T$', side='right')
    p = counts / counts.sum()
    bars(d, X3, top + 19.0, W3, 8.0, p, BLUE, LIGHT_BLUE)
    d.label(X3 + W3 / 2, top + 30.6, r'$p=x/N\in\Delta^{G-1}$', MODULE)
    d.label(X3 + W3 / 2, top + 35.4, 'or smoothed, log1p, sqrt', SMALL, color=GREY)
    ce_y = sy
    d.box(X3, ce_y, W3, 10.0, r'soft CE  $-\sum_g p_g\log q_g$', edge=ORANGE, fill='#FFFFFF', size=MODULE, dashed=True)
    kit.arrow(d, [(X3 + W3 / 2, top + 37.5), (X3 + W3 / 2, ce_y - .4)], 'supervision', 'train', side='right')
    kit.arrow(d, [(X3 - .5, ce_y + 5.0), (X2 + W2 + .5, ce_y + 5.0)], 'supervision')

    # ---- 4 generate and evaluate -----------------------------------------------------------------------------
    d.box(X4, top, W4, 10.0, '', edge=GREY, fill=GREY_FILL)
    d.label(X4 + W4 / 2, top + 3.3, r'$s\sim$ training library sizes', SMALL)
    d.label(X4 + W4 / 2, top + 7.0, r'Mult$(s,q)$ or DM$(s,\kappa q)$', NOTE)
    kit.arrow(d, [(X4 + W4 / 2, top + 10.0), (X4 + W4 / 2, top + 14.0)], 'inference', 'sample', side='right')
    bars(d, X4 + 2.0, top + 15.0, W4 - 4.0, 6.0, RNG.gamma(.45, 1, 16), PURPLE, LIGHT_PURPLE)
    d.label(X4 + W4 / 2, top + 23.8, r'generated cell $\tilde x$', NOTE, color=PURPLE)
    ck = top + 27.0
    for k, t in enumerate(('NLL, cell NLL', 'MMD, real-vs-gen AUC', r'$\Delta$, top-20 DE genes')):
        d.check(X4, ck + k * 6.2, W4, 5.2, t, ORANGE, size=SMALL)
    hb = sy
    d.box(X4, hb, W4, 10.0, '', edge=ORANGE, fill='#FFFFFF', dashed=True)
    d.label(X4 + W4 / 2, hb + 3.4, 'held-out real cells', NOTE, color=ORANGE)
    d.label(X4 + W4 / 2, hb + 7.2, 'unseen conditions', SMALL, color=ORANGE)
    kit.arrow(d, [(X4 + 6.0, hb - .2), (X4 + 6.0, ck + 2 * 6.2 + 5.4)], 'supervision', 'score', side='right')

    # q -> sampler: under the training column, then up the gutter left of column 4
    ry = sy + 13.5
    kit.arrow(d, [(X2 + W2 / 2, sy + 10.0), (X2 + W2 / 2, ry), (X4 - 2.6, ry), (X4 - 2.6, top + 5.0), (X4 - .5, top + 5.0)],
              'inference')
    d.label(X3 + W3 / 2, ry - 1.4, r'$q$', NOTE, color=INK_LABEL)

    kit.arrow_legend(c, d, H - 3.5)
    return c, font


INK_LABEL = '#333333'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='paper/figs')
    a = ap.parse_args()
    c, font = compose()
    caption = ('**A decision language model reads a text state and returns, in one forward pass, a distribution over a fixed gene panel; '
               'every method is scored with the same likelihood and sampler.** (1) The state is a prompt: a cell type (conditional '
               'generation) or a perturbation together with a random control cell $x_0$, written as its 48 most abundant panel genes '
               'with quantized log-mass levels (the tokens shown are the first ones of a real K562 control cell). (2) A LoRA-tuned '
               'language model encodes the prompt; a linear decision head on the last-token state, with an optional fixed log-prior '
               '$b$ (none, the training mean, or the smoothed control cell itself), gives one logit per gene, and a softmax gives '
               '$q$ on the simplex. (3) A real cell is the probability vector $p=x/N$ of its counts (or a transformed version); '
               'training minimizes the soft cross-entropy between $p$ and $q$ (dashed). (4) A shared sampler draws a generated cell '
               'from $q$ with a training library size, and held-out real cells of unseen donors or perturbations score the model by '
               'likelihood, sample quality and perturbation effects. Colours: teal conditioning, purple model, blue data, orange '
               'training and evaluation only. Bar heights are schematic.')
    kit.finish(c, a.out, NAME, caption, {'font_file': font, 'data_source': 'schematic (example tokens from biology_checks.json)',
                                         'model_trained': True})


if __name__ == '__main__':
    main()
