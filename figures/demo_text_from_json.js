// Regenerate the appendix demo text from figs/demo_outputs.json with the same logic as figures/demo_outputs.py
// (used when the JSON is available locally but the data needed to redraw the figure are not).
// node figures/demo_text_from_json.js
const fs = require("fs");
const recs = JSON.parse(fs.readFileSync("paper/figs/demo_outputs.json", "utf8"));
const esc = t => t.replace(/\\/g, "").replace(/_/g, "\\_").replace(/&/g, "\\&").replace(/%/g, "\\%");
const REF = { cb_type_mean: "the cell-type mean", cb_additive: "the additive baseline", cb_perturbed_mean: "the perturbed mean" };
const refname = r => REF[r] || (r.startsWith("cb_zs_ontology") ? "the nearest Cell Ontology type" : r);
const sg = x => (x >= 0 ? "+" : "") + x.toFixed(3);
const lines = ["\\begin{itemize}"];
recs.forEach((r, k) => {
  let s = `\\item[(${String.fromCharCode(97 + k)})] \\emph{${esc(r.title)}}: ${esc(r.unit)}, the median of ${r.n_units} units ` +
    `(NLL difference to ${refname(r.ref)}: $${sg(r.nll_diff_cad_minus_ref)}$ nats per UMI)`;
  if ("de_pearson_cad" in r) s += `; Pearson correlation with the real log fold change on its top-20 DE genes: $${r.de_pearson_cad.toFixed(2)}$ for CAD and $${r.de_pearson_ref.toFixed(2)}$ for the reference`;
  lines.push(s + ".");
});
lines.push("\\end{itemize}");
fs.writeFileSync("paper/figs/demo_outputs_text.tex", lines.join("\n") + "\n");
console.log(lines.join("\n"));
