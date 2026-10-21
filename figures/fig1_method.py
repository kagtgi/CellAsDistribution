#!/usr/bin/env python3
"""Figure 1 (rev5): why generate a sentence? The same text prompt, two ways to answer it.

python figures/fig1_method.py --out paper/figs
Top lane (prior route, Cell2Sentence): the language model writes the cell as a rank sentence token by token and a
rank-to-count map turns it back into expression. Bottom lane (this work): a decision head answers in one forward pass
with one probability per gene of a FIXED, ordered panel; that distribution q is the cell, and it is scored exactly.

Provenance of every drawn element:
  prompt ............. the real test prompt of FOXA1+FOXL2 (paper/figs/demo_main.json; src/train_cell.py prompt())
  sentence tokens .... the most abundant genes of a real K562 cell, in rank order (TMSB4X, HBZ, SET, HBG2, CKS1B, NCL;
                       paper/auto_rev2/biology_checks.json and the control-cell text of the appendix demo)
  530 tokens, 1.99 s . mean generated length and decode time of C2S-Gemma (paper/auto_rev2/speed_bench.json, Sec. 5.1)
  38% ................ share of the likelihood gain kept by the published log-rank map (paper/auto_rev3, Figure 2)
  q bars ............. CellAsDistribution's decoded distribution for FOXA1+FOXL2 (test split, Gemma, control residual,
                       mean over control cells) on the fixed 20-gene axis of Figure 3: panel genes with the highest mean
                       training frequency, in panel order (paper/figs/demo_main_plotdata.npz, row 1). Real values.
  0.004 s, 2048 ...... decode time per cell with the control residual (speed_bench) and panel size (Sec. 3.1)
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch

SKILL = Path.home() / '.claude' / 'skills' / 'paper-schematic-figures' / 'scripts'
sys.path.insert(0, str(SKILL))
from figkit import kit  # noqa: E402
from figkit.kit import TEAL, TEAL_FILL, ORANGE, GREY, GREY_FILL, INK  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
NAME = 'fig1_method'
W, H = 165.0, 70.0
HEAD, MOD, NOTE, SMALL = 8.6, 8.0, 7.4, 7.0        # pt at print size (the figure is printed at 100% of the text width)
DARK_TEAL, MUTED = '#1C6F6D', '#7A828C'
LANE_FILL = '#F2FAF9'


def load_q():
    rec = json.load(open(REPO / 'paper' / 'figs' / 'demo_main.json'))[1]
    z = np.load(REPO / 'paper' / 'figs' / 'demo_main_plotdata.npz')
    axis = z['1_axis']
    return rec['unit'], rec['prompt'], rec['top_genes'], z['1_cad'][axis]


def compose():
    unit, prompt, genes, q = load_q()
    c, d, font = kit.new_figure(H, width_mm=W)
    matplotlib.rcParams['mathtext.fontset'] = 'cm'
    kit.single_panel(c)

    # ---------------- shared input: the text state ----------------
    PX, PW, PY, PH = 3.2, 33.0, 20.0, 31.0
    d.label(PX, PY - 3.2, 'Text state (prompt)', HEAD, color=INK, weight='bold', ha='left')
    d.box(PX, PY, PW, PH, '', edge=GREY, fill='#FFFFFF')
    lines = ['Cell context: K562 cells,', 'CRISPRa perturbation.', 'Perturbation:', f'{unit}.', 'Gene expression',
             'frequency distribution', 'of the perturbed cell:']
    for k, t in enumerate(lines):
        d.label(PX + 2.0, PY + 3.9 + k * 3.9, t, SMALL, color=INK, ha='left')
    fork = PX + PW + 2.5

    # ---------------- lane A: prior route (generate a sentence) ----------------
    AY = 4.5                                   # lane top
    d.label(42.0, AY, 'Prior route: generate a sentence (Cell2Sentence)', HEAD, color=MUTED, weight='bold', ha='left')
    ay = AY + 4.0                              # boxes top
    d.box(42.0, ay, 20.0, 11.0, '', edge=MUTED, fill=GREY_FILL)
    d.label(52.0, ay + 4.0, 'language', MOD, color=INK)
    d.label(52.0, ay + 7.8, 'model', MOD, color=INK)
    toks = ['TMSB4X', 'HBZ', 'SET', 'HBG2', 'CKS1B']
    tx = 67.0
    for k, t in enumerate(toks):
        w = 4.4 + 1.25 * len(t)
        d.box(tx, ay + 2.5, w, 6.0, t, edge=MUTED, fill='#FFFFFF', size=SMALL)
        tx += w + 1.0
    d.label(tx + 2.5, ay + 5.5, '…', MOD, color=MUTED)
    tend = tx + 5.0
    d.label((67.0 + tend) / 2, ay + 12.6, 'token by token: 530 tokens, 1.99 s per cell', SMALL, color=MUTED)
    kit.arrow(d, [(62.2, ay + 5.5), (66.6, ay + 5.5)], 'measurement', color=MUTED)
    mx = tend + 3.0
    d.box(mx, ay, 163.0 - mx, 11.0, '', edge=MUTED, fill=GREY_FILL, dashed=True)
    d.label((mx + 163.0) / 2, ay + 4.0, 'rank → count map', MOD, color=INK)
    d.label((mx + 163.0) / 2, ay + 7.8, 'heuristic, after the fact', SMALL, color=MUTED)
    kit.arrow(d, [(tend + .3, ay + 5.5), (mx - .4, ay + 5.5)], 'measurement', color=MUTED)
    d.label((mx + 163.0) / 2, ay + 12.6, 'likelihood gain kept: 38%', SMALL, color=MUTED)

    # ---------------- lane B: this work (answer with a distribution) ----------------
    BY = 27.5
    d.ax.add_patch(FancyBboxPatch((39.5, d.Y(H - 1.0)), W - 40.5, H - 1.0 - BY + 1.0, boxstyle='round,pad=0,rounding_size=1.6',
                                  lw=.8, edgecolor=TEAL, facecolor=LANE_FILL, zorder=1))
    d.label(42.0, BY + 3.0, 'This work: answer with a distribution (decision head)', HEAD, color=DARK_TEAL, weight='bold', ha='left')
    by = BY + 8.0
    d.box(42.0, by, 20.0, 11.0, '', edge=TEAL, fill='#FFFFFF')
    d.label(52.0, by + 4.0, 'language', MOD, color=INK)
    d.label(52.0, by + 7.8, 'model', MOD, color=INK)
    d.label(52.0, by + 15.0, 'one forward pass', SMALL, color=DARK_TEAL)
    d.label(52.0, by + 18.6, '0.004 s per cell', SMALL, color=DARK_TEAL)
    hx = 66.0
    d.box(hx, by, 22.0, 11.0, '', edge=TEAL, fill=TEAL_FILL, lw=1.0)
    d.label(hx + 11.0, by + 4.0, 'decision head', MOD, color=DARK_TEAL, weight='bold')
    d.label(hx + 11.0, by + 7.8, r'softmax over $G$', SMALL, color=INK)
    d.label(hx + 11.0, by + 15.0, 'one probability', SMALL, color=DARK_TEAL)
    d.label(hx + 11.0, by + 18.6, 'per panel gene', SMALL, color=DARK_TEAL)
    kit.arrow(d, [(62.3, by + 5.5), (65.6, by + 5.5)], 'inference', r'$h$', size=SMALL)

    # the answer: q on the fixed, ordered gene axis (real values)
    QX, QW, QT, QH = 94.0, 45.0, by + 1.5, 18.0     # plot area: x, width, top, height (mm)
    n = len(q)
    bw = QW / n
    vmax = q.max() * 1.08
    for k, v in enumerate(q):
        h = QH * v / vmax
        d.ax.add_patch(Rectangle((QX + k * bw + .18, d.Y(QT + QH)), bw - .36, max(h, .3), facecolor=TEAL, edgecolor=DARK_TEAL,
                                 lw=.3, alpha=.85, zorder=4))
    d.ax.plot([QX, QX + QW], [d.Y(QT + QH)] * 2, color=INK, lw=.5, zorder=5)
    for k, g in enumerate(genes):
        d.ax.text(QX + (k + .5) * bw, d.Y(QT + QH + .8), g, rotation=90, ha='center', va='top', fontsize=SMALL - 1.6,
                  color=INK, zorder=6, fontfamily=matplotlib.rcParams['font.family'])
    d.label(QX + QW / 2, by - 1.6, r'$q$ = the cell: share of its transcripts per gene', SMALL, color=DARK_TEAL, weight='bold')
    kit.arrow(d, [(hx + 22.3, by + 5.5), (QX - .6, by + 5.5)], 'inference')
    d.label(QX + QW / 2, H - 3.6, 'fixed gene order (20 of 2048 shown)', SMALL, color=MUTED)

    # what the distribution gives: sampling and an exact score
    OX = QX + QW + 4.5
    OW = 163.0 - OX
    d.box(OX, by - 1.0, OW, 10.5, '', edge=TEAL, fill='#FFFFFF')
    d.label(OX + OW / 2, by + 2.6, 'sample cells', SMALL, color=INK)
    d.label(OX + OW / 2, by + 6.6, r'$x\sim\mathrm{Mult}(N,q)$', SMALL - .6, color=INK)
    d.box(OX, by + 12.5, OW, 10.5, '', edge=ORANGE, fill='#FFFFFF', dashed=True)
    d.label(OX + OW / 2, by + 16.1, 'exact', SMALL, color=ORANGE)
    d.label(OX + OW / 2, by + 20.1, 'likelihood', SMALL, color=ORANGE)

    kit.arrow(d, [(QX + QW + .6, by + 4.2), (OX - .5, by + 4.2)], 'inference')
    kit.arrow(d, [(QX + QW + .6, by + 17.7), (OX - .5, by + 17.7)], 'supervision')

    # ---------------- the prompt feeds both lanes ----------------
    kit.arrow(d, [(PX + PW + .4, PY + PH / 2), (fork, PY + PH / 2), (fork, ay + 5.5), (41.6, ay + 5.5)], 'conditioning')
    kit.arrow(d, [(fork, PY + PH / 2), (fork, by + 5.5), (41.6, by + 5.5)], 'conditioning')
    return c, font


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='paper/figs')
    a = ap.parse_args()
    c, font = compose()
    caption = ('**Why generate a sentence?** The same prompt, answered two ways. Top: a cell-sentence model writes the cell as '
               'gene names ranked by expression, token by token, and a heuristic map turns the ranks back into expression. '
               'Bottom: a decision head answers in one forward pass with one probability for every gene of a fixed, ordered '
               'panel; that distribution is the cell, can be sampled, and is scored by an exact likelihood. Bars: the '
               'distribution decoded for the test perturbation FOXA1+FOXL2 on 20 of the 2048 panel genes.')
    kit.finish(c, a.out, NAME, caption, {'font_file': font,
                                         'data_source': 'q: paper/figs/demo_main_plotdata.npz row 1; timings: speed_bench.json; '
                                                        '38%: auto_rev3 ceilings; prompt: demo_main.json',
                                         'model_trained': True})


if __name__ == '__main__':
    main()
