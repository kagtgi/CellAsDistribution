// Fetch Cell Ontology (CL) ids, definitions and parents for the PBMC cell-type labels from the EBI OLS4 API.
// node src/fetch_cell_ontology.js > splits/pbmc_cell_ontology.json
// Definitions are the formal CL definitions (obo_definition_citation, IAO:0000115), copied verbatim; nothing is written by hand.
const LABELS = ["B cell", "CD14-low, CD16-positive monocyte", "CD14-positive monocyte",
  "CD16-negative, CD56-bright natural killer cell, human", "CD4-positive, alpha-beta T cell",
  "CD4-positive, alpha-beta cytotoxic T cell", "CD8-positive, alpha-beta T cell",
  "central memory CD4-positive, alpha-beta T cell", "central memory CD8-positive, alpha-beta T cell",
  "conventional dendritic cell", "double negative thymocyte", "effector memory CD4-positive, alpha-beta T cell",
  "effector memory CD8-positive, alpha-beta T cell", "erythrocyte", "gamma-delta T cell", "hematopoietic stem cell",
  "innate lymphoid cell", "memory B cell", "memory regulatory T cell", "mucosal invariant T cell",
  "myeloid dendritic cell, human", "naive B cell", "naive regulatory T cell",
  "naive thymus-derived CD4-positive, alpha-beta T cell", "naive thymus-derived CD8-positive, alpha-beta T cell",
  "natural killer cell", "plasmablast", "plasmacytoid dendritic cell", "platelet"];
const OLS = "https://www.ebi.ac.uk/ols4/api";
// dataset labels that are older names of a CL term (CELLxGENE label -> current CL id)
const ALIAS = { "mucosal invariant T cell": "CL:0000940" };

async function getJSON(url) {
  for (let i = 0; i < 4; i++) {
    try { const r = await fetch(url); if (r.ok) return await r.json(); } catch (e) { }
    await new Promise(res => setTimeout(res, 1500));
  }
  throw new Error("failed " + url);
}

(async () => {
  const out = {};
  for (const lab of LABELS) {
    let doc;
    if (ALIAS[lab]) {
      doc = { obo_id: ALIAS[lab], iri: "http://purl.obolibrary.org/obo/" + ALIAS[lab].replace(":", "_") };
    } else {
      const s = await getJSON(`${OLS}/search?q=${encodeURIComponent(lab)}&ontology=cl&exact=true&queryFields=label&rows=5`);
      doc = (s.response.docs || []).find(d => d.label === lab && d.obo_id && d.obo_id.startsWith("CL:"));
    }
    if (!doc) { out[lab] = { error: "no exact CL label match" }; continue; }
    const iri = encodeURIComponent(encodeURIComponent(doc.iri));
    const term = await getJSON(`${OLS}/ontologies/cl/terms/${iri}`);
    let parents = [];
    try {
      const p = await getJSON(`${OLS}/ontologies/cl/terms/${iri}/parents`);
      parents = ((p._embedded || {}).terms || []).map(t => [t.obo_id, t.label]);
    } catch (e) { }
    let anc = [];
    try {
      const a = await getJSON(`${OLS}/ontologies/cl/terms/${iri}/hierarchicalAncestors?size=200`);
      anc = ((a._embedded || {}).terms || []).map(t => t.obo_id).filter(x => x && x.startsWith("CL:"));
    } catch (e) { }
    out[lab] = { id: doc.obo_id, label: term.label, definition: ((term.obo_definition_citation || [])[0] || {}).definition || "",
                 dataset_label_is_alias: !!ALIAS[lab], parents, ancestors_cl: anc };
  }
  console.log(JSON.stringify({ source: "EBI OLS4 API, Cell Ontology", fetched: new Date().toISOString(), terms: out }, null, 1));
})();
