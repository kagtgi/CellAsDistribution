// rev3 main-text tables, generated from the result JSONs (never edited by hand). All at \normalsize (RECOMB: >= 10 pt).
// node figures/make_tables_rev3.js     reads paper/auto_rev3/*.json and paper/auto_rev2/*.json, writes paper/auto_rev3/tab_*.tex
// Every metric column declares its better direction (min: lower is better, max: higher is better, one: closest to 1);
// the header carries the arrow and the best displayed value of each column is set in bold (all tied values).
const fs = require("fs");
const D3 = "paper/auto_rev3/", D2 = "paper/auto_rev2/", D4 = "paper/auto_rev4/";
const J = f => (fs.existsSync(f) ? JSON.parse(fs.readFileSync(f, "utf8")) : null);
const sg = x => (x >= 0 ? "+" : "\u2212") + Math.abs(x).toFixed(3);
const sgs = x => (x >= 0 ? "+" : "\u2212") + Math.abs(x).toFixed(3).replace(/^0\./, ".");
const ci = d => `${sg(d[0])} [${sgs(d[1])}, ${sgs(d[2])}]`;
const write = (name, s) => { fs.writeFileSync(D3 + name, s.replace(/\u2212/g, "$-$")); console.log("wrote", D3 + name); };
const OURS = lab => lab.startsWith("\\cad");
const ARROW = { min: " $\\downarrow$", max: " $\\uparrow$", one: " ($\\to1$)" };
const BOLD_NOTE = "Best value of each column in bold; arrows give the better direction.";

// cell value helpers: {num, txt} or null
const num = (x, dp) => (x == null ? null : { num: +x.toFixed(dp), txt: x.toFixed(dp) });
const diffci = d => (d ? { num: +d[0].toFixed(3), txt: ci(d) } : null);
const zero = () => ({ num: 0, txt: "0" });

// rows: [[label, key]], cols: [{h, dir, get(key) -> cell|null}]
function table({ rows, cols, firstHead = "Model", pre = "", sep = "4pt", size = "", place = "t", caption, label, spec }) {
  const cells = rows.map(([, k]) => cols.map(c => c.get(k)));
  const best = cols.map((c, j) => {
    const v = cells.map(r => r[j]).filter(Boolean).map(x => x.num);
    if (!v.length || !c.dir) return null;
    return c.dir === "min" ? Math.min(...v) : c.dir === "max" ? Math.max(...v) : v.reduce((b, x) => (Math.abs(x - 1) < Math.abs(b - 1) ? x : b));
  });
  const isBest = (c, j, x) => x && best[j] != null && (c.dir === "one" ? Math.abs(x.num - 1) === Math.abs(best[j] - 1) : x.num === best[j]);
  let s = `\\begin{table}[${place}]\n\\centering\n${size}\\setlength{\\tabcolsep}{${sep}}\n\\begin{tabular}{${spec || "l" + "r".repeat(cols.length)}}\n\\toprule\n`;
  s += pre;
  s += [firstHead, ...cols.map(c => c.h + (c.dir ? ARROW[c.dir] : ""))].join(" & ") + " \\\\\n\\midrule\n";
  rows.forEach(([lab], i) => {
    const r = cols.map((c, j) => { const x = cells[i][j]; if (!x) return "---"; return isBest(c, j, x) ? `\\textbf{${x.txt}}` : x.txt; });
    s += (OURS(lab) ? "\\rowcolor{ours}" : "") + [lab, ...r].join(" & ") + " \\\\\n";
  });
  s += "\\bottomrule\n\\end{tabular}\n" + `\\caption{${caption} ${BOLD_NOTE}}\n\\label{${label}}\n\\end{table}\n`;
  return s;
}

// ---------- Table: Norman, test, all perturbations (likelihood, realism, downstream biology) ----------
const nm = J(D3 + "test_downstream_vs_additive.json");
if (nm) {
  const S = nm.strata.all.stats;
  const rows = [["Perturbed mean", "cb_perturbed_mean"], ["Additive (reference)", "cb_additive"],
    ["Additive + target induction", "cb_additive_ti"], ["scGen-style", "cb_scGen_style"],
    ["C2S, autoregressive", "c2sg_s0_eps0.9"], ["Additive, control residual", "cb_additive_residual"],
    ["\\cad{}, no control cell", "t2_g_noctrl_s0"], ["\\cad{}, control residual", "t2_g_ctrl_notext_s0"],
    ["\\cad{}, NB head", "t2_g_noctrl_nb_s0"], ["\\cad{}, low-rank head", "t2_g_noctrl_lowrank_s0"]].filter(([, t]) => S[t]);
  const m = (k, dp) => t => (S[t][k] ? num(S[t][k].mean, dp) : null);
  write("tab_norman_main.tex", table({
    rows, cols: [
      { h: "$\\Delta$NLL to additive", dir: "min", get: t => (t === "cb_additive" ? zero() : diffci(S[t].nll_pooled.diff)) },
      { h: "cell NLL", dir: "min", get: m("nll_mix", 3) }, { h: "MMD", dir: "min", get: m("mmd_dm", 3) },
      { h: "AUC", dir: "min", get: m("auc_dm", 3) }, { h: "$r_{\\mathrm{DE}}$", dir: "max", get: m("pearson_de20", 2) },
      { h: "sign", dir: "max", get: m("sign_de20", 2) }],
    caption: `Perturbation response, Norman K562 test split (${nm.strata.all.n} held-out perturbations, matched control cells, seed 0, C2S-Scale-Gemma-2-2B for every language model). $\\Delta$NLL: pooled NLL per UMI minus the additive baseline's [95\\% bootstrap over perturbations; the additive baseline's NLL is ${S.cb_additive.nll_pooled.mean.toFixed(3)}]. cell NLL: mixture over the $M=50$ control cells (or the low-rank head's draws). MMD and real-versus-generated AUC on Dirichlet-multinomial samples. Downstream biology, with the perturbed genes removed (added after the test results; post hoc): $r_{\\mathrm{DE}}$, Pearson correlation of predicted and real pseudobulk change over the held-out control mean on the perturbation's top-20 differentially expressed genes; sign, the fraction of those genes whose change has the right sign.`,
    label: "tab:norman_main" }));

  // ---------- Table: by stratum (NLL vs additive; downstream r_DE; direct target error) ----------
  const st = nm.strata, r4 = J(D4 + "test_r4_vs_additive.json");
  const g = (k, t) => (r4 && r4.strata[k].stats[t]) || st[k].stats[t];
  const R = [["Additive", "cb_additive"], ["Additive + target induction", "cb_additive_ti"], ["scGen-style", "cb_scGen_style"],
    ["No LM (gene embeddings)", "t3_nolm_s0"], ["\\cad{}, no control cell", "t2_g_noctrl_s0"], ["\\cad{} + target prior", "t2_g_noctrl_s0_tprior"],
    ["\\cad{}, NB head", "t2_g_noctrl_nb_s0"], ["\\cad{}, opaque gene ids", "t2_g_desc_opaque_s0"]].filter(([, t]) => g("all", t));
  const dnl = k => t => (t === "cb_additive" ? zero() : (g(k, t) ? { num: +g(k, t).nll_pooled.diff[0].toFixed(3), txt: sg(g(k, t).nll_pooled.diff[0]) } : null));
  const rde = k => t => (g(k, t) && g(k, t).pearson_de20 ? num(g(k, t).pearson_de20.mean, 2) : null);
  const ter = k => t => (g(k, t) && g(k, t).target_abs_err ? num(g(k, t).target_abs_err.mean, 2) : null);
  const pre = " & \\multicolumn{3}{c}{$\\Delta$NLL to additive $\\downarrow$} & \\multicolumn{3}{c}{downstream $r_{\\mathrm{DE}}$ $\\uparrow$} & \\multicolumn{2}{c}{target error $\\downarrow$} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-9}\n";
  write("tab_strata.tex", table({
    rows: R, sep: "3.2pt", pre, cols: [
      { h: "single", dir: "min", get: dnl("s") }, { h: "1 unseen", dir: "min", get: dnl("pu") }, { h: "2 seen", dir: "min", get: dnl("p") },
      { h: "single", dir: "max", get: rde("s") }, { h: "1 unseen", dir: "max", get: rde("pu") }, { h: "2 seen", dir: "max", get: rde("p") },
      { h: "1 unseen", dir: "min", get: ter("pu") }, { h: "2 seen", dir: "min", get: ter("p") }].map(c => ({ ...c, arrowInGroup: true })),
    caption: `Norman test split by stratum: unseen single genes ($n=${st.s.n}$), pairs with one unseen gene ($n=${st.pu.n}$) and pairs of two genes seen alone in training ($n=${st.p.n}$). $\\Delta$NLL: mean paired difference to the additive baseline (intervals in the text). Downstream $r_{\\mathrm{DE}}$ as in Table~\\ref{tab:norman_main}, perturbed genes excluded. Target error: mean absolute error of the predicted log fold change of the perturbed gene(s) themselves over the held-out control mean, for targets in the panel. Target prior: every target seen alone in training gets its own training-single frequency, unseen targets the global induction factor, applied to the saved \\cad{} output. Added after the test results (post hoc).`,
    label: "tab:strata" }).replace(/(single|1 unseen|2 seen) \$\\(downarrow|uparrow)\$/g, "$1"));   // arrows sit on the group headers
}

// ---------- Table: does the language model matter? (E1, pre-registered; test seeds 0-2) ----------
const e1r4 = [0, 1, 2].map(s => J(D4 + `test_r4_e1_vs_cad_s${s}.json`)).filter(Boolean);
const e1 = e1r4.length === 3 ? e1r4 : [0, 1, 2].map(s => J(D3 + `test_e1_vs_cad_s${s}.json`)).filter(Boolean);
if (e1.length) {
  const j = e1[0];
  const rows = [["\\cad{} (reference)", "ref"], ["No LM, learned", "t3_nolm_s0"], ["No LM, GenePT", "t4_nolm_genept_s0"],
    ["No LM, co-expression", "t4_nolm_coexpr_s0"], ["Frozen LM + MLP", "t3_g_frozen_mlp_s0"],
    ["Additive", "cb_additive"], ["Additive + target induction", "cb_additive_ti"]].filter(([, t]) => t === "ref" || j.strata.all.stats[t]);
  const d = k => t => (t === "ref" ? zero() : (k === "all" ? diffci(j.strata[k].stats[t].nll_pooled.diff)
    : { num: +j.strata[k].stats[t].nll_pooled.diff[0].toFixed(3), txt: sg(j.strata[k].stats[t].nll_pooled.diff[0]) }));
  const seeds = e1.map((jj, s) => {
    const a = jj.strata.all.stats[`t3_nolm_s${s}`], b = jj.strata.all.stats[`t3_g_frozen_mlp_s${s}`];
    return `seed ${s}: ${a ? sg(a.nll_pooled.diff[0]) : "---"} / ${b ? sg(b.nll_pooled.diff[0]) : "---"}`;
  }).join("; ");
  write("tab_e1.tex", table({
    rows, firstHead: "Model", sep: "3pt", cols: [
      { h: "$\\Delta$NLL, all", dir: "min", get: d("all") }, { h: "single", dir: "min", get: d("s") },
      { h: "1 unseen", dir: "min", get: d("pu") }, { h: "both seen", dir: "min", get: d("p") },
      { h: "$\\Delta r_{\\mathrm{DE}}$", dir: "max", get: t => (t === "ref" ? zero() : (j.strata.all.stats[t].pearson_de20 ? diffci(j.strata.all.stats[t].pearson_de20.diff) : null)) }],
    caption: `Does the language model matter? Pooled NLL per UMI of each model minus that of \\cad{} without a control cell (same seed; positive values favor \\cad{}), Norman test split, seed 0: overall with [95\\% bootstrap over perturbations], and the mean difference by stratum (single $n=21$, 1 unseen $n=29$, both seen $n=26$). $\\Delta r_{\\mathrm{DE}}$: the same paired difference in downstream $r_{\\mathrm{DE}}$ (perturbed genes excluded; negative favors \\cad{}). Every model uses the same 20{,}000 training cells, frequency target, training-mean log-prior, softmax over the panel and scoring. Overall NLL difference by seed (learned no LM / frozen LM): ${seeds}. GenePT: frozen text embeddings of NCBI gene summaries~\\citep{chen2024genept}; co-expression: loadings of training cells' principal components; both give every perturbed gene, including unseen ones, an embedding (post hoc). Pre-registered (Appendix~\\ref{app:protocol}). Differences whose interval covers zero are ties.`,
    label: "tab:e1" }));
}

// ---------- PBMC and zero-shot, re-emitted from rev2 JSONs at full size ----------
const pb = J(D2 + "test_pbmc.json");
if (pb) {
  const S = pb.stats;
  const rows = [["Global mean", "Global mean"], ["Cell-type mean (reference)", "Cell-type mean"], ["scVI", "scVI"],
    ["scVI, own NB sampling", "scVI, own NB sampling"], ["scANVI, own NB sampling", "scANVI, own NB sampling"],
    ["\\cad{}", "CAD (Gemma)"], ["\\cad{}, low-rank head", "CAD (Gemma), low-rank head"]].filter(([, k]) => S[k]);
  const own = (k, m) => { const p = pb.per_unit[k], u = Object.keys(p), key = m.replace("_dm", "_nb");
    return u.every(x => key in p[x]) ? { mean: u.reduce((a, x) => a + p[x][key], 0) / u.length } : null; };
  const get = (m, dp) => k => { let v = S[k][m]; if (k.includes("own NB") && m.endsWith("_dm")) v = own(k, m); return v ? num(v.mean, dp) : null; };
  write("tab_pbmc.tex", table({
    rows, cols: [
      { h: "$\\Delta$NLL to type mean", dir: "min", get: k => (k === "Cell-type mean" ? zero() : diffci(S[k].nll_pooled.diff)) },
      { h: "cell NLL", dir: "min", get: get("nll_mix", 3) }, { h: "MMD", dir: "min", get: get("mmd_dm", 3) },
      { h: "AUC", dir: "min", get: get("auc_dm", 3) }, { h: "type acc.", dir: "max", get: get("type_acc_gen_dm", 2) }],
    caption: `Conditional generation, human PBMC test split (unseen donors; ${pb.units.length} cell types; seed 0; Gemma backbone). $\\Delta$NLL: paired difference to the cell-type mean [95\\% bootstrap over cell types]. \\emph{Own NB sampling}: cells drawn from scVI/scANVI's own negative binomial from latent states of training cells, with cell NLL as the mixture over these draws; other rows use the shared samplers (generation metrics on Dirichlet-multinomial samples).`,
    label: "tab:pbmc_main" }));
}
const zs = J(D4 + "pbmc_zs_test.json") || J(D2 + "pbmc_zs_test.json");
if (zs) {
  const C = zs.compare["Global mean"], ref = zs.compare["Name-overlap type mean"];
  const rows = [["Nearest Cell Ontology type", "Nearest CL type mean"], ["Cell Ontology parents", "CL-parent mean"],
    ["Name-overlap type (string match)", "Name-overlap type mean"], ["\\cad{}, opaque type id", "CAD, opaque type id"],
    ["\\cad{}, type name", "CAD, type name only"], ["\\cad{}, name + definition", "CAD, CL definition"],
    ["\\cad{} (Gemma), opaque type id", "CAD (Gemma), opaque type id"], ["\\cad{} (Gemma), type name", "CAD (Gemma), type name only"],
    ["\\cad{} (Gemma), name + definition", "CAD (Gemma), CL definition"]].filter(([, k]) => C[k]);
  write("tab_zeroshot.tex", table({
    rows, firstHead: "Unseen cell type", cols: [
      { h: "$\\Delta$NLL to global mean", dir: "min", get: k => diffci(C[k].nll_pooled.diff) },
      { h: "$\\Delta$NLL to name overlap", dir: "min", get: k => (ref && ref[k] ? diffci(ref[k].nll_pooled.diff) : (k === "Name-overlap type mean" ? zero() : null)) },
      { h: "type acc.", dir: "max", get: k => num(C[k].type_acc_gen_dm.mean, 2) }],
    caption: `Zero-shot cell types, PBMC test split: ${zs.n_units} types, each held out of training in one of five leave-types-out folds (Qwen3.5-0.8B and, added after the test results, C2S-Scale-Gemma-2-2B backbones; seed 0). Paired differences [95\\% bootstrap over cell types]. Baselines use training types only.`,
    label: "tab:zeroshot" }));
}

// ---------- Supplement: full Norman table (all baselines, all-gene r_Delta) ----------
if (nm) {
  const S = nm.strata.all.stats;
  const g = J(D2 + "test_all_vs_additive.json");          // GEARS and control-mean rows (not in the rev3 downstream run)
  const rows = [["Control mean", "cb_control_mean"], ["Perturbed mean", "cb_perturbed_mean"], ["Additive (reference)", "cb_additive"],
    ["Additive + target induction", "cb_additive_ti"], ["Ridge on gene embeddings", "cb_ridge"], ["scGen-style", "cb_scGen_style"],
    ["GEARS", "cb_GEARS_trctrl"], ["C2S, autoregressive", "c2sg_s0_eps0.9"], ["Additive, control residual", "cb_additive_residual"],
    ["\\cad{}, no control cell", "t2_g_noctrl_s0"], ["\\cad{}, control residual", "t2_g_ctrl_notext_s0"], ["\\cad{}, NB head", "t2_g_noctrl_nb_s0"],
    ["\\cad{}, low-rank head", "t2_g_noctrl_lowrank_s0"], ["\\cad{}, opaque gene ids", "t2_g_desc_opaque_s0"]];
  const get = t => S[t] || (g && g.strata.all.stats[t]);
  const R = rows.filter(([, t]) => get(t));
  const m = (k, dp) => t => (get(t)[k] ? num(get(t)[k].mean, dp) : null);
  write("tab_norman_full.tex", table({
    rows: R, sep: "3pt", size: "\\small\n", place: "h", cols: [
      { h: "NLL", dir: "min", get: m("nll_pooled", 3) },
      { h: "$\\Delta$NLL to additive", dir: "min", get: t => (t === "cb_additive" ? zero() : diffci(get(t).nll_pooled.diff)) },
      { h: "cell NLL", dir: "min", get: m("nll_mix", 3) }, { h: "MMD", dir: "min", get: m("mmd_dm", 3) },
      { h: "AUC", dir: "min", get: m("auc_dm", 3) }, { h: "$r_\\Delta$", dir: "max", get: m("pearson_delta_pert", 2) },
      { h: "$r_{\\mathrm{DE}}$", dir: "max", get: m("pearson_de20", 2) }],
    caption: `All methods, Norman test split (76 perturbations, seed 0, matched control cells). $r_\\Delta$: Pearson correlation of predicted and real change relative to the training perturbed centroid over all panel genes except the perturbed ones; $r_{\\mathrm{DE}}$ as in Table~\\ref{tab:norman_main}. GEARS and control-mean rows: perturbed genes not excluded (earlier run).`,
    label: "tab:norman_full" }));
}

// ---------- Supplement: co-expression (E5-lite) ----------
const cx = J(D3 + "test_coexpr.json");
if (cx) {
  const rows = Object.keys(cx.summary).map(l => [l.replace(/^CAD/, "\\cad{}"), l]);
  const m = (k, dp) => l => num(cx.summary[l][k].mean, dp);
  write("tab_coexpr.tex", table({
    rows, firstHead: "Generator", size: "\\small\n", place: "h", cols: [
      { h: "correlation error", dir: "min", get: m("corr_err", 3) }, { h: "variance ratio", dir: "one", get: m("var_ratio", 2) },
      { h: "PC1-share error", dir: "min", get: m("pc1_err", 3) }],
    caption: `Co-expression of generated cells, Norman test split (${cx.units.length} perturbations with at least 100 cells; ${cx.H} training-selected variable genes; seed 0; post hoc). Correlation error: mean absolute difference between gene--gene correlations across real and generated cells; variance ratio: total variance generated/real (1 is ideal).`,
    label: "tab:coexpr" }));
}

// ---------- Supplement: add-on panel with every perturbed gene (rev4, post hoc) ----------
const tp = J(D4 + "tp_vs_additive.json");
if (tp) {
  const st = tp.strata, g = (k, t) => st[k].stats[t];
  const rows = [["Additive", "cb_additive"], ["Perturbed mean", "cb_perturbed_mean"], ["Additive + target induction", "cb_additive_ti"],
    ["No LM (learned)", "t4_tp_nolm_s0"], ["No LM (GenePT)", "t4_tp_nolm_genept_s0"], ["No LM (co-expression)", "t4_tp_nolm_coexpr_s0"],
    ["\\cad{}, no control cell", "t4_tp_g_noctrl_s0"]].filter(([, t]) => g("all", t));
  const dnl = k => t => (t === "cb_additive" ? zero() : (g(k, t) && g(k, t).nll_pooled.diff ? { num: +g(k, t).nll_pooled.diff[0].toFixed(3), txt: sg(g(k, t).nll_pooled.diff[0]) } : null));
  // target metrics averaged over the units whose target(s) are in the panel (3 of the 105 perturbed genes are not
  // in the dataset's gene list, so the strata summary, which needs every unit, drops them)
  const dd = J(D4 + "tp_downstream.json");
  const tavg = (k, t, m) => { const pu = dd && dd.per_unit[t]; if (!pu) return null;
    const u = st[k].units.filter(x => pu[x] && pu[x][m] != null); return u.length ? { v: u.reduce((a, x) => a + pu[x][m], 0) / u.length, n: u.length } : null; };
  const ter = k => t => { const r = tavg(k, t, "target_abs_err"); return r ? num(r.v, 2) : null; };
  const tsg = k => t => { const r = tavg(k, t, "target_sign"); return r ? num(r.v, 2) : null; };
  const nTarg = (tavg("s", "cb_additive", "target_abs_err") || { n: 0 }).n;
  const pre = " & \\multicolumn{3}{c}{$\\Delta$NLL to additive $\\downarrow$} & \\multicolumn{3}{c}{target error $\\downarrow$} & \\multicolumn{1}{c}{target sign $\\uparrow$} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-8}\n";
  write("tab_tp.tex", table({
    rows, sep: "3pt", size: "\\small\n", place: "h", pre, cols: [
      { h: "single", dir: "min", get: dnl("s") }, { h: "1 unseen", dir: "min", get: dnl("pu") }, { h: "2 seen", dir: "min", get: dnl("p") },
      { h: "single", dir: "min", get: ter("s") }, { h: "1 unseen", dir: "min", get: ter("pu") }, { h: "2 seen", dir: "min", get: ter("p") },
      { h: "single", dir: "max", get: tsg("s") }],
    caption: `Add-on gene panel with every perturbed gene forced in (${tp.strata.s.n} unseen single genes, ${nTarg} of them with their target in the panel; same splits as the main panel, verified cell by cell; Norman test split, seed 0, C2S-Scale-Gemma-2-2B for \\cad{}; post hoc). $\\Delta$NLL: mean paired difference to additive; target error: mean absolute error of the predicted log fold change of the perturbed gene(s); target sign: fraction of unseen single targets predicted to go up.`,
    label: "tab:tp" }).replace(/(single|1 unseen|2 seen) \$\\(downarrow|uparrow)\$/g, "$1"));
}
