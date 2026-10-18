// Main-text and appendix tables, generated from the result JSONs (never edited by hand).
// node figures/make_tables_rev2.js        reads paper/auto_rev2/*.json, writes paper/auto_rev2/tab_*.tex
const fs = require("fs");
const D = "paper/auto_rev2/";
const J = f => (fs.existsSync(D + f) ? JSON.parse(fs.readFileSync(D + f, "utf8")) : null);
const f3 = x => x.toFixed(3), f4 = x => x.toFixed(4);
const sg = x => (x >= 0 ? "+" : "") + x.toFixed(3);
const ci = d => `${sg(d[0])}\\,[${sg(d[1])},\\,${sg(d[2])}]`;
const write = (name, s) => { fs.writeFileSync(D + name, s); console.log("wrote", D + name); };

// ---------------- Norman, test, all perturbations, paired to additive ----------------
const nm = J("test_all_vs_additive.json");
if (nm) {
  const all = nm.strata.all, S = all.stats;
  const rows = [
    ["Perturbed mean", "cb_perturbed_mean"], ["Additive (reference)", "cb_additive"], ["scGen-style", "cb_scGen_style"],
    ["GEARS", "cb_GEARS_trctrl"], ["C2S, autoregressive (Gemma)", "c2sg_s0_eps0.9"],
    ["Additive, control residual (no LM)", "cb_additive_residual"],
    ["\\cad{}, no control cell", "t2_g_noctrl_s0"], ["\\cad{}, control residual", "t2_g_ctrl_notext_s0"],
    ["\\cad{}, NB head", "t2_g_noctrl_nb_s0"], ["\\cad{}, low-rank head", "t2_g_noctrl_lowrank_s0"]];
  const cols = [["nll_pooled", "NLL $\\downarrow$ ($\\Delta$ to additive)"], ["nll_mix", "cell NLL $\\downarrow$"],
    ["mmd_dm", "MMD $\\downarrow$"], ["auc_dm", "AUC $\\downarrow$"]];
  const present = rows.filter(([, t]) => S[t]);
  const best = {};
  for (const [m] of cols) best[m] = present.reduce((b, [, t]) => (S[t][m] && (!b || S[t][m].mean < S[b][m].mean) ? t : b), null);
  let s = "\\begin{table}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{3.5pt}\n\\begin{tabular}{lrrrr}\n\\toprule\n";
  s += "Model & " + cols.map(c => c[1]).join(" & ") + " \\\\\n\\midrule\n";
  for (const [lab, t] of present) {
    const c = cols.map(([m]) => {
      const v = S[t][m]; if (!v) return "---";
      let x = m.startsWith("mmd") ? f4(v.mean) : f3(v.mean);
      if (m === "nll_pooled" && v.diff) x += ` (${ci(v.diff)})`;
      return best[m] === t ? `\\textbf{${x}}` : x;
    });
    s += (lab.startsWith("\\cad") ? "\\rowcolor{ours}" : "") + lab + " & " + c.join(" & ") + " \\\\\n";
  }
  s += "\\bottomrule\n\\end{tabular}\n";
  s += `\\caption{Perturbation response, Norman K562, test split (${all.n} held-out perturbations; matched control cells; seed 0; C2S-Scale-Gemma-2-2B backbone for every language model). NLL: cross-entropy per UMI of the pooled real cells (paired difference to the additive baseline with 95\\% bootstrap interval over perturbations). cell NLL: mixture over the $M=50$ control cells (or the low-rank head's draws). MMD and real-versus-generated AUC on cells from the shared Dirichlet-multinomial sampler. Best value in bold; \\cad{} rows shaded. Per-stratum results in Table~\\ref{tab:strata}.}\n\\label{tab:norman_main}\n\\end{table}\n`;
  write("tab_norman_main.tex", s);

  // stratified table: NLL difference to additive, and r_Delta / r_DE (pseudobulk) per stratum
  const st = nm.strata;
  const R2 = [["scGen-style", "cb_scGen_style"], ["C2S, autoregressive", "c2sg_s0_eps0.9"], ["Additive, control residual", "cb_additive_residual"],
    ["\\cad{}, no control cell", "t2_g_noctrl_s0"], ["\\cad{}, control residual", "t2_g_ctrl_notext_s0"], ["\\cad{}, NB head", "t2_g_noctrl_nb_s0"],
    ["\\cad{}, opaque gene ids", "t2_g_desc_opaque_s0"]].filter(([, t]) => st.all.stats[t]);
  const K = [["s", "single"], ["pu", "1 unseen"], ["p", "both seen"]];
  // compact interval for this wide table: -0.009 [-.037,+.010]
  const cs = d => `${sg(d[0])} [${sg(d[1]).replace("0.", ".")},${sg(d[2]).replace("0.", ".")}]`;
  let t2 = "\\begin{table}[t]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{1.8pt}\n\\begin{tabular}{l" + "r".repeat(7) + "}\n\\toprule\n";
  t2 += " & \\multicolumn{3}{c}{$\\Delta$NLL to additive $\\downarrow$} & \\multicolumn{2}{c}{$r_\\Delta\\uparrow$} & \\multicolumn{2}{c}{$r_{\\mathrm{DE}}\\uparrow$} \\\\\n";
  t2 += "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\n";
  t2 += "Model & " + K.map(k => `${k[1]} ($n=${st[k[0]].n}$)`).join(" & ") + " & 1 unseen & both & 1 unseen & both \\\\\n\\midrule\n";
  const rD = (k, t) => st[k].stats[t] && st[k].stats[t].pearson_delta_pert ? f3(st[k].stats[t].pearson_delta_pert.mean) : "---";
  const rE = (k, t) => st[k].stats[t] && st[k].stats[t].pearson_de20 ? f3(st[k].stats[t].pearson_de20.mean) : "---";
  t2 += "Additive & 0 & 0 & 0 & " + rD("pu", "cb_additive") + " & " + rD("p", "cb_additive") + " & " + rE("pu", "cb_additive") + " & " + rE("p", "cb_additive") + " \\\\\n";
  for (const [lab, t] of R2) {
    const d = K.map(([k]) => (st[k].stats[t] && st[k].stats[t].nll_pooled.diff ? cs(st[k].stats[t].nll_pooled.diff) : "---"));
    t2 += (lab.startsWith("\\cad") ? "\\rowcolor{ours}" : "") + lab + " & " + d.join(" & ") + " & " + rD("pu", t) + " & " + rD("p", t) + " & " + rE("pu", t) + " & " + rE("p", t) + " \\\\\n";
  }
  t2 += "\\bottomrule\n\\end{tabular}\n";
  t2 += `\\caption{Norman test split by stratum of the gene-level split. $\\Delta$NLL: paired difference to the additive baseline [95\\% bootstrap over perturbations]. $r_\\Delta$: Pearson correlation of predicted and real change relative to the training perturbed centroid over all panel genes; $r_{\\mathrm{DE}}$: the same relative to the held-out control mean on each perturbation's top-20 DE genes (pseudobulk). On unseen single genes the additive baseline equals the perturbed mean, so $r_\\Delta$ is not reported there.}\n\\label{tab:strata}\n\\end{table}\n`;
  write("tab_strata.tex", t2);
}

// ---------------- PBMC, test, conditional generation ----------------
const pb = J("test_pbmc.json");
if (pb) {
  const S = pb.stats;
  const rows = [["Global mean", "Global mean"], ["Cell-type mean (reference)", "Cell-type mean"], ["scVI", "scVI"],
    ["scVI, own NB sampling", "scVI, own NB sampling"], ["scANVI, own NB sampling", "scANVI, own NB sampling"],
    ["\\cad{} (Gemma)", "CAD (Gemma)"], ["\\cad{} (Qwen)", "CAD (Qwen)"], ["\\cad{}, low-rank head", "CAD (Gemma), low-rank head"]]
    .filter(([, k]) => S[k]);
  const cols = [["nll_pooled", "NLL $\\downarrow$ ($\\Delta$ to type mean)", 3], ["nll_mix", "cell NLL $\\downarrow$", 3], ["mmd_dm", "MMD $\\downarrow$", 4],
    ["auc_dm", "AUC $\\downarrow$", 3], ["type_acc_gen_dm", "type acc. $\\uparrow$", 3]];
  let s = "\\begin{table}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{3.5pt}\n\\begin{tabular}{lrrrrr}\n\\toprule\n";
  s += "Model & " + cols.map(c => c[1]).join(" & ") + " \\\\\n\\midrule\n";
  const own = (k, m) => {   // own-sampling rows: generation metrics from the model's own NB samples (per-unit *_nb)
    const pu = pb.per_unit[k], u = Object.keys(pu), key = m.replace("_dm", "_nb");
    return u.every(x => key in pu[x]) ? { mean: u.reduce((a, x) => a + pu[x][key], 0) / u.length } : null;
  };
  for (const [lab, k] of rows) {
    const c = cols.map(([m, , dp]) => {
      let v = S[k][m];
      if (k.includes("own NB") && m.endsWith("_dm")) v = own(k, m);
      if (!v) return "---"; let x = v.mean.toFixed(dp); if (m === "nll_pooled" && v.diff) x += ` (${ci(v.diff)})`; return x; });
    s += (lab.startsWith("\\cad") ? "\\rowcolor{ours}" : "") + lab + " & " + c.join(" & ") + " \\\\\n";
  }
  s += "\\bottomrule\n\\end{tabular}\n";
  s += `\\caption{Conditional generation, human PBMC, test split (unseen donors P7 and P8; ${pb.units.length} cell types with at least 30 cells; seed 0). NLL: paired difference to the cell-type mean [95\\% bootstrap over cell types]. \\emph{Own NB sampling}: cells drawn from scVI/scANVI's own negative binomial from latent states of training cells of the type, and cell NLL as the mixture over these draws; all other rows use the shared samplers. Generation metrics on Dirichlet-multinomial samples except for the own-sampling rows.}\n\\label{tab:pbmc_main}\n\\end{table}\n`;
  write("tab_pbmc.tex", s);
}

// ---------------- zero-shot cell types, test ----------------
const zs = J("pbmc_zs_test.json");
if (zs) {
  const C = zs.compare["Global mean"];
  const ref = zs.compare["Name-overlap type mean"];
  const rows = [["Nearest Cell Ontology type", "Nearest CL type mean"], ["Cell Ontology parent", "CL-parent mean"],
    ["Name-overlap type (string match)", "Name-overlap type mean"], ["\\cad{}, opaque type id", "CAD, opaque type id"],
    ["\\cad{}, type name", "CAD, type name only"], ["\\cad{}, name + CL definition", "CAD, CL definition"]].filter(([, k]) => C[k]);
  let s = "\\begin{table}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{3.5pt}\n\\begin{tabular}{lrrrr}\n\\toprule\n";
  s += "Model (unseen cell type) & $\\Delta$NLL to global mean $\\downarrow$ & $\\Delta$NLL to name-overlap $\\downarrow$ & $r_\\mu\\uparrow$ & type acc. $\\uparrow$ \\\\\n\\midrule\n";
  for (const [lab, k] of rows) {
    const g = C[k].nll_pooled, n = ref && ref[k] ? ci(ref[k].nll_pooled.diff) : (k === "Name-overlap type mean" ? "0" : "---");
    s += (lab.startsWith("\\cad") ? "\\rowcolor{ours}" : "") + `${lab} & ${ci(g.diff)} & ${n} & ${f3(C[k].mean_corr_dm.mean)} & ${f3(C[k].type_acc_gen_dm.mean)} \\\\\n`;
  }
  s += "\\bottomrule\n\\end{tabular}\n";
  s += `\\caption{Zero-shot cell types, PBMC test split (unseen donors): ${zs.n_units} cell types, each held out of training in one of five leave-types-out folds (Qwen3.5-0.8B backbone, seed 0). Paired differences [95\\% bootstrap over cell types]. Baselines use only training types: the type closest in the Cell Ontology, the pooled training types under the held-out type's Cell Ontology parents, and the training type whose name shares the most words.}\n\\label{tab:zeroshot}\n\\end{table}\n`;
  write("tab_zeroshot.tex", s);
}
