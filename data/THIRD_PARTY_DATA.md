# Third-party data sources

Every dataset under `data/` derives from a third-party release. This file
records, for each source, what it is, the license it was published under, and
exactly what we changed. CC BY 3.0 and CC BY 4.0 both require that
modifications be indicated, so the "Modifications" entries below are part of
the licensing obligation, not just documentation.

Licenses were checked against the upstream sources on 2026-10-07. Where a
statement could not be verified at its original source, it is marked as such.

---

## MetaQA — `data/metaqa/`

| | |
|---|---|
| Source | Zhang, Dai, Kozareva, Smola, Song. *Variational Reasoning for Question Answering with Knowledge Graph.* AAAI 2018. |
| Upstream | https://github.com/yuyuz/MetaQA |
| License | **CC BY 3.0 Unported** (see `LICENSE.txt` in the upstream repository) |
| Derived files | `import_nodes.csv`, `import_rels.csv`, `qa/hop1.jsonl`, `qa/hop2.jsonl`, `qa/hop2_w_path.jsonl`, `qa/hop3.jsonl`, `qa/hop3_w_path.jsonl` |

**Modifications.** The movie knowledge graph was converted into Neo4j
bulk-import CSVs (`:ID` / `:LABEL` / `:START_ID` / `:END_ID` / `:TYPE`
columns), and the 1/2/3-hop question files were converted into this project's
JSONL evaluation format with entity-set gold answers. No question text or
graph triple was altered.

**Upstream note.** MetaQA is itself partly based on WikiMovies (the Facebook
bAbI project). The original Facebook download page is no longer reachable;
secondary mirrors list WikiMovies as CC BY 3.0, but this could **not be
verified at the original source**. The MetaQA release we actually derive from
carries its own CC BY 3.0 grant.

---

## PrimeKG — `data/primekgqa/import_nodes.csv`

| | |
|---|---|
| Source | Chandak, Huang, Zitnik. *Building a knowledge graph to enable precision medicine.* Scientific Data 10(1):67, 2023. |
| Upstream | https://doi.org/10.7910/DVN/IXA7BM · https://github.com/mims-harvard/PrimeKG |
| License | **CC0 1.0 Universal** for the dataset (Harvard Dataverse); MIT for the PrimeKG codebase |
| Derived files | `import_nodes.csv` |

**Modifications.** Node records were extracted and reformatted as a Neo4j
bulk-import CSV. The relationship file is **not** redistributed here because
of its size; `README.md` explains how to obtain PrimeKG and build it locally.

**Upstream note.** PrimeKG aggregates roughly twenty upstream biomedical
databases, and the authors state that the CC0 dedication applies to the
PrimeKG aggregation rather than to the licenses of those individual sources
(DrugBank, for example, carries its own terms). We did **not** audit those
twenty licenses. Anyone reusing individual records should check the
originating database's terms.

---

## PrimeKGQA — `data/primekgqa/` (QA splits, built locally)

| | |
|---|---|
| Source | Yan, Westphal, Seliger, Usbeck. *Bridging the Gap: Generating a Comprehensive Biomedical Knowledge Graph Question Answering Dataset.* ECAI 2024. |
| Upstream | https://zenodo.org/records/13829395 · https://github.com/xixi019/primeKGQG |
| License | **CC BY 4.0** for the dataset (Zenodo); MIT for the generation code |
| Derived files | None redistributed. The QA splits are rebuilt from the upstream release by `app/dataset_construction/convert_primekgqa_original.py`. |

**Modifications.** PrimeKGQA questions are converted into entity-set
evaluation items (PrimeKGQA-Struct) by the converter script; the conversion
selects anchored, type-constrained questions and re-derives gold answer sets.
The converted files are produced on the user's machine and are not shipped in
this repository.

---

## KGT pan-cancer graph and QA — `data/pcqa/`

| | |
|---|---|
| Source | Feng, Zhou, Ma, Zheng, He, Li. *Knowledge graph-based thought: a knowledge graph-enhanced LLM framework for pan-cancer question answering.* GigaScience 14:giae082, 2025. |
| Upstream | https://github.com/yichun10/bioKGQA-KGT |
| License | **CC0 1.0 Universal** for the data in the upstream repository; MIT for its code; the article itself is CC BY 4.0 |
| Redistributed as-is | `PcQA.json` (the 405 free-form QA pairs) |
| Derived files | `import_nodes.csv`, `import_rels.csv` |
| Upstream documentation | `README.md` in this directory describes the upstream graph build procedure |

**Modifications.** The public portion of the pan-cancer graph was converted
into Neo4j bulk-import CSVs. Separately, the 405 free-form QA pairs were
reconstructed into entity-set gold answers — the PcQA benchmark reported in
the paper — which is this project's own contribution and is covered by
`data/LICENSE`; see Section 5.1 of the paper for the construction procedure.

**Upstream note — important.** The upstream repository releases only *a
portion* of the pan-cancer knowledge graph, "for local validation." The
complete SmartQuerier Oncology Knowledge Graph is **proprietary** and is
obtained by contacting its owner. Nothing beyond the publicly released subset
is redistributed here, and the CSVs in this directory must not be described as
the full pan-cancer graph. The upstream README also references
[SDKG-11](https://github.com/ZhuChaoY/SDKG-11) as a data source; that
repository states **no license**, so no material traceable to it is
redistributed here.

---

## Summary

| Source | Upstream license | Redistributed here | Our modification |
|---|---|---|---|
| MetaQA | CC BY 3.0 | Converted CSV + JSONL | Neo4j import format, evaluation format |
| PrimeKG | CC0 1.0 | Node CSV only | Neo4j import format |
| PrimeKGQA | CC BY 4.0 | Nothing (rebuilt locally) | Entity-set conversion script |
| KGT / PcQA | CC0 1.0 (public subset) | `PcQA.json`, converted CSVs | Neo4j import format; entity-set gold reconstruction |

If you believe any attribution here is incomplete or incorrect, please open an
issue on the repository.
