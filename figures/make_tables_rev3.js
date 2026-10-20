// rev3 main-text tables, generated from the result JSONs (never edited by hand). All at \normalsize (RECOMB: >= 10 pt).
// node figures/make_tables_rev3.js     reads paper/auto_rev3/*.json and paper/auto_rev2/*.json, writes paper/auto_rev3/tab_*.tex
const fs = require("fs");
const D3 = "paper/auto_rev3/", D2 = "paper/auto_rev2/";
const J = f => (fs.existsSync(f) ? JSON.parse(fs.readFileSync(f, "utf8")) : null);
const f3 = x => x.toFixed(3);
const sg = x => (x >= 0 ? "+" : "\u2212") + Math.abs(x).toFixed(3);
const sgs = x => (x >= 0 ? "+" : "\u2212") + Math.abs(x).toFixed(3).replace(/^0\./, ".");
const ci = d => `${sg(d[0])} [${sgs(d[1])}, ${sgs(d[2])}]`;
const write = (name, s) => { fs.writeFileSync(D3 + name, s.replace(/\u2212/g, "$-$")); console.log("wrote", D3 + name); };
const OURS = lab => lab.startsWith("\\cad");
const head = (cols, spec) => `\\begin{table}[t]\n\\centering\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{${spec}}\n\\toprule\n` + cols + " \\\\\n\\midrule\n";

// ---------- Table: Norman, test, all perturbations (likelihood, realism, downstream biology) ----------
const nm = J(D3 + "test_downstream_vs_additive.json");
if (nm) {
  const S = nm.strata.all.stats;
  const rows = [["Perturbed mean", "cb_perturbed_mean"], ["Additive (reference)", "cb_additive"],
    ["Additive + target induction", "cb_additive_ti"], ["scGen-style", "cb_scGen_style"],
    ["C2S, autoregressive", "c2sg_s0_eps0.9"], ["Additive, control residual", "cb_additive_residual"],
    ["\\cad{}, no control cell", "t2_g_noctrl_s0"], ["\\cad{}, control residual", "t2_g_ctrl_notext_s0"],
    ["\\cad{}, NB head", "t2_g_noctrl_nb_s0"], ["\\cad{}, low-rank head", "t2_g_noctrl_lowrank_s0"]].filter(([, t]) => S[t]);
  const v = (t, m, dp = 3) => (S[t][m] ? S[t][m].mean.toFixed(dp) : "---");
  let s = head("Model & $\\Delta$NLL to additive & cell NLL & MMD & AUC & $r_{\\mathrm{DE}}$ & sign", "lrrrrrr");
  for (const [lab, t] of rows) {
    const d = t === "cb_additive" ? "0" : ci(S[t].nll_pooled.diff);
    s += (OURS(lab) ? "\\rowcolor{ours}" : "") + `${lab} & ${d} & ${v(t, "nll_mix")} & ${v(t, "mmd_dm")} & ${v(t, "auc_dm")} & ${v(t, "pearson_de20", 2)} & ${v(t, "sign_de20", 2)} \\\\\n`;
  }
  s += "\\bottomrule\n\\end{tabular}\n";
  s += `\\caption{Perturbation response, Norman K562 test split (${nm.strata.all.n} held-out perturbations, matched control cells, seed 0, C2S-Scale-Gemma-2-2B for every language model). $\\Delta$NLL: pooled NLL per UMI minus the additive baseline's [95\\% bootstrap over perturbations; lower is better; the additive baseline's NLL is ${v("cb_additive", "nll_pooled")}]. cell NLL: mixture over the $M=50$ control cells (or the low-rank head's draws). MMD and real-versus-generated AUC (lower is better) on Dirichlet-multinomial samples. Downstream biology, with the perturbed genes removed: $r_{\\mathrm{DE}}$, Pearson correlation of predicted and real pseudobulk change over the held-out control mean on the perturbation's top-20 differentially expressed genes; sign, the fraction of those genes whose change has the right sign.}\n\\label{tab:norman_main}\n\\end{table}\n`;
  write("tab_norman_main.tex", s);

  // ---------- Table: by stratum (NLL vs additive; downstream r_DE; direct target error) ----------
  const st = nm.strata;
  const K = [["s", "single"], ["pu", "1 unseen"], ["p", "2 seen"]];
  const R = [["Additive", "cb_additive"], ["Additive + target induction", "cb_additive_ti"], ["scGen-style", "cb_scGen_style"],
    ["\\cad{}, no control cell", "t2_g_noctrl_s0"], ["\\cad{}, opaque gene ids", "t2_g_desc_opaque_s0"]].filter(([, t]) => st.all.stats[t]);
  const rde = (k, t) => (st[k].stats[t] && st[k].stats[t].pearson_de20 ? st[k].stats[t].pearson_de20.mean.toFixed(2) : "---");
  const terr = (k, t) => (st[k].stats[t] && st[k].stats[t].target_abs_err ? st[k].stats[t].target_abs_err.mean.toFixed(2) : "---");
  const dnl = (k, t) => (t === "cb_additive" ? "0" : (st[k].stats[t] ? sg(st[k].stats[t].nll_pooled.diff[0]) : "---"));
  let t2 = `\\begin{table}[t]\n\\centering\n\\setlength{\\tabcolsep}{3.2pt}\n\\begin{tabular}{lrrrrrrrr}\n\\toprule\n`;
  t2 += " & \\multicolumn{3}{c}{$\\Delta$NLL to additive} & \\multicolumn{3}{c}{downstream $r_{\\mathrm{DE}}$} & \\multicolumn{2}{c}{target error} \\\\\n";
  t2 += "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-9}\n";
  t2 += "Model & " + K.map(k => k[1]).join(" & ") + " & " + K.map(k => k[1]).join(" & ") + " & 1 unseen & 2 seen \\\\\n\\midrule\n";
  for (const [lab, t] of R)
    t2 += (OURS(lab) ? "\\rowcolor{ours}" : "") + `${lab} & ` + K.map(([k]) => dnl(k, t)).join(" & ") + " & " + K.map(([k]) => rde(k, t)).join(" & ") + ` & ${terr("pu", t)} & ${terr("p", t)} \\\\\n`;
  t2 += "\\bottomrule\n\\end{tabular}\n";
  t2 += `\\caption{Norman test split by stratum: unseen single genes ($n=${st.s.n}$), pairs with one unseen gene ($n=${st.pu.n}$) and pairs of two genes seen alone in training ($n=${st.p.n}$). $\\Delta$NLL: mean paired difference to the additive baseline (intervals in the text). Downstream $r_{\\mathrm{DE}}$ as in Table~\\ref{tab:norman_main}, perturbed genes excluded. Target error: mean absolute error of the predicted log fold change of the perturbed gene(s) themselves over the held-out control mean, for targets in the panel. Added after the test results (post hoc).}\n\\label{tab:strata}\n\\end{table}\n`;
  write("tab_strata.tex", t2);
}

// ---------- Table: does the language model matter? (E1, pre-registered; test seeds 0-2) ----------
const e1 = [0, 1, 2].map(s => J(D3 + `test_e1_vs_cad_s${s}.json`)).filter(Boolean);
if (e1.length) {
  const rows = [["No LM (gene embeddings)", "t3_nolm_s"], ["Frozen LM + MLP head", "t3_g_frozen_mlp_s"],
    ["Additive", "cb_additive"], ["Additive + target induction", "cb_additive_ti"]];
  const K = [["all", "all"], ["s", "single"], ["pu", "1 unseen"], ["p", "both seen"]];
  const tg = (t, s) => (t.endsWith("_s") ? t + s : t);
  let s = head("$\\Delta$ to \\cad{} & " + K.map(k => k[1]).join(" & ") + " & $r_{\\mathrm{DE}}$", "lrrrrr");
  for (const [lab, t] of rows) {
    const j = e1[0], tt = tg(t, 0);
    if (!j.strata.all.stats[tt]) continue;
    const cells = K.map(([k]) => (k === "all" ? ci(j.strata[k].stats[tt].nll_pooled.diff) : sg(j.strata[k].stats[tt].nll_pooled.diff[0])));
    const r = j.strata.all.stats[tt].pearson_de20 ? ci(j.strata.all.stats[tt].pearson_de20.diff) : "---";
    s += `${lab} & ${cells.join(" & ")} & ${r} \\\\\n`;
  }
  // seed robustness: overall NLL difference per seed
  const seeds = e1.map((j, s) => {
    const a = j.strata.all.stats[`t3_nolm_s${s}`], b = j.strata.all.stats[`t3_g_frozen_mlp_s${s}`];
    return `seed ${s}: ${a ? sg(a.nll_pooled.diff[0]) : "---"} / ${b ? sg(b.nll_pooled.diff[0]) : "---"}`;
  }).join("; ");
  s += "\\bottomrule\n\\end{tabular}\n";
  s += `\\caption{Does the language model matter? Pooled NLL per UMI of each model minus that of \\cad{} without a control cell (same seed), Norman test split, seed 0: overall with [95\\% bootstrap over perturbations], and the mean difference by stratum (single $n=21$, 1 unseen $n=29$, both seen $n=26$); positive values favor \\cad{}. $r_{\\mathrm{DE}}$: the same paired difference in downstream $r_{\\mathrm{DE}}$ (perturbed genes excluded; negative favors \\cad{}). Every model uses the same 20{,}000 training cells, frequency target, training-mean log-prior, softmax over the panel and scoring. Overall NLL difference by seed (no LM / frozen LM): ${seeds}. Pre-registered (Appendix~\\ref{app:protocol}).}\n\\label{tab:e1}\n\\end{table}\n`;
  write("tab_e1.tex", s);
}

// ---------- PBMC and zero-shot, re-emitted from rev2 JSONs at full size ----------
const pb = J(D2 + "test_pbmc.json");
if (pb) {
  const S = pb.stats;
  const rows = [["Global mean", "Global mean"], ["Cell-type mean (reference)", "Cell-type mean"], ["scVI", "scVI"],
    ["scVI, own NB sampling", "scVI, own NB sampling"], ["scANVI, own NB sampling", "scANVI, own NB sampling"],
    ["\\cad{}", "CAD (Gemma)"], ["\\cad{}, low-rank head", "CAD (Gemma), low-rank head"]].filter(([, k]) => S[k]);
  const cols = [["nll_pooled", 3], ["nll_mix", 3], ["mmd_dm", 3], ["auc_dm", 3], ["type_acc_gen_dm", 2]];
  const own = (k, m) => { const p = pb.per_unit[k], u = Object.keys(p), key = m.replace("_dm", "_nb");
    return u.every(x => key in p[x]) ? { mean: u.reduce((a, x) => a + p[x][key], 0) / u.length } : null; };
  let s = head("Model & $\\Delta$NLL to type mean & cell NLL & MMD & AUC & type acc.", "lrrrrr");
  for (const [lab, k] of rows) {
    const c = cols.map(([m, dp]) => {
      let v = S[k][m]; if (k.includes("own NB") && m.endsWith("_dm")) v = own(k, m);
      if (!v) return "---";
      if (m === "nll_pooled") return k === "Cell-type mean" ? "0" : (v.diff ? ci(v.diff) : "---");
      return v.mean.toFixed(dp); });
    s += (OURS(lab) ? "\\rowcolor{ours}" : "") + lab + " & " + c.join(" & ") + " \\\\\n";
  }
  s += "\\bottomrule\n\\end{tabular}\n";
  s += `\\caption{Conditional generation, human PBMC test split (unseen donors; ${pb.units.length} cell types; seed 0; Gemma backbone). $\\Delta$NLL: paired difference to the cell-type mean [95\\% bootstrap over cell types]. \\emph{Own NB sampling}: cells drawn from scVI/scANVI's own negative binomial from latent states of training cells, with cell NLL as the mixture over these draws; other rows use the shared samplers (generation metrics on Dirichlet-multinomial samples).}\n\\label{tab:pbmc_main}\n\\end{table}\n`;
  write("tab_pbmc.tex", s);
}
const zs = J(D2 + "pbmc_zs_test.json");
if (zs) {
  const C = zs.compare["Global mean"], ref = zs.compare["Name-overlap type mean"];
  const rows = [["Nearest Cell Ontology type", "Nearest CL type mean"], ["Cell Ontology parents", "CL-parent mean"],
    ["Name-overlap type (string match)", "Name-overlap type mean"], ["\\cad{}, opaque type id", "CAD, opaque type id"],
    ["\\cad{}, type name", "CAD, type name only"], ["\\cad{}, name + definition", "CAD, CL definition"]].filter(([, k]) => C[k]);
  let s = head("Unseen cell type & $\\Delta$NLL to global mean & $\\Delta$NLL to name overlap & type acc.", "lrrr");
  for (const [lab, k] of rows) {
    const n = ref && ref[k] ? ci(ref[k].nll_pooled.diff) : (k === "Name-overlap type mean" ? "0" : "---");
    s += (OURS(lab) ? "\\rowcolor{ours}" : "") + `${lab} & ${ci(C[k].nll_pooled.diff)} & ${n} & ${C[k].type_acc_gen_dm.mean.toFixed(2)} \\\\\n`;
  }
  s += "\\bottomrule\n\\end{tabular}\n";
  s += `\\caption{Zero-shot cell types, PBMC test split: ${zs.n_units} types, each held out of training in one of five leave-types-out folds (Qwen3.5-0.8B backbone, seed 0). Paired differences [95\\% bootstrap over cell types]. Baselines use training types only.}\n\\label{tab:zeroshot}\n\\end{table}\n`;
  write("tab_zeroshot.tex", s);
}

// ---------- Supplement: full Norman table (all baselines, all-gene r_Delta) ----------
if (nm) {
  const S = nm.strata.all.stats;
  const g = J(D2 + "test_all_vs_additive.json");          // GEARS row (not in the rev3 downstream run)
  const rows = [["Control mean", "cb_control_mean"], ["Perturbed mean", "cb_perturbed_mean"], ["Additive (reference)", "cb_additive"],
    ["Additive + target induction", "cb_additive_ti"], ["Ridge on gene embeddings", "cb_ridge"], ["scGen-style", "cb_scGen_style"],
    ["GEARS", "cb_GEARS_trctrl"], ["C2S, autoregressive", "c2sg_s0_eps0.9"], ["Additive, control residual", "cb_additive_residual"],
    ["\\cad{}, no control cell", "t2_g_noctrl_s0"], ["\\cad{}, control residual", "t2_g_ctrl_notext_s0"], ["\\cad{}, NB head", "t2_g_noctrl_nb_s0"],
    ["\\cad{}, low-rank head", "t2_g_noctrl_lowrank_s0"], ["\\cad{}, opaque gene ids", "t2_g_desc_opaque_s0"]];
  const get = t => S[t] || (g && g.strata.all.stats[t]);
  let s = `\\begin{table}[h]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{lrrrrrrr}\n\\toprule\n`;
  s += "Model & NLL & $\\Delta$NLL to additive & cell NLL & MMD & AUC & $r_\\Delta$ & $r_{\\mathrm{DE}}$ \\\\\n\\midrule\n";
  for (const [lab, t] of rows) {
    const x = get(t); if (!x) continue;
    const v = (m, dp = 3) => (x[m] ? x[m].mean.toFixed(dp) : "---");
    s += (OURS(lab) ? "\\rowcolor{ours}" : "") + `${lab} & ${v("nll_pooled")} & ${t === "cb_additive" ? "0" : (x.nll_pooled.diff ? ci(x.nll_pooled.diff) : "---")} & ${v("nll_mix")} & ${v("mmd_dm")} & ${v("auc_dm")} & ${v("pearson_delta_pert", 2)} & ${v("pearson_de20", 2)} \\\\\n`;
  }
  s += "\\bottomrule\n\\end{tabular}\n";
  s += `\\caption{All methods, Norman test split (76 perturbations, seed 0, matched control cells). $r_\\Delta$: Pearson correlation of predicted and real change relative to the training perturbed centroid over all panel genes except the perturbed ones; $r_{\\mathrm{DE}}$ as in Table~\\ref{tab:norman_main}. GEARS and control-mean rows: perturbed genes not excluded (earlier run).}\n\\label{tab:norman_full}\n\\end{table}\n`;
  write("tab_norman_full.tex", s);
}

// ---------- Supplement: co-expression (E5-lite) ----------
const cx = J(D3 + "test_coexpr.json");
if (cx) {
  const lab = l => l.replace(/^CAD/, "\\cad{}");
  let s = `\\begin{table}[h]\n\\centering\n\\small\n\\begin{tabular}{lrrr}\n\\toprule\n`;
  s += "Generator & correlation error $\\downarrow$ & variance ratio & PC1-share error $\\downarrow$ \\\\\n\\midrule\n";
  for (const [l, m] of Object.entries(cx.summary))
    s += (l.startsWith("CAD") ? "\\rowcolor{ours}" : "") + `${lab(l)} & ${m.corr_err.mean.toFixed(3)} & ${m.var_ratio.mean.toFixed(2)} & ${m.pc1_err.mean.toFixed(3)} \\\\\n`;
  s += "\\bottomrule\n\\end{tabular}\n";
  s += `\\caption{Co-expression of generated cells, Norman test split (${cx.units.length} perturbations with at least 100 cells; ${cx.H} training-selected variable genes; seed 0; post hoc). Correlation error: mean absolute difference between gene--gene correlations across real and generated cells; variance ratio: total variance generated/real (1 is ideal).}\n\\label{tab:coexpr}\n\\end{table}\n`;
  write("tab_coexpr.tex", s);
}
