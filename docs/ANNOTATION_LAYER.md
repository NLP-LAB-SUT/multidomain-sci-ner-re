# Annotation Layer and Data Formats

The corpus is distributed in three formats:

1. The **annotation layer**: stand-off JSONL, the canonical release.
2. The **NER training format**: inline tags, read by `src/ner/*`.
3. The **RE training format**: marked entity pairs, read by `src/re/*`.

All three are available on request (see [`DATA_REQUEST.md`](../DATA_REQUEST.md)). Small samples of each are in [`data/sample/`](../data/sample).

## 1. Annotation layer: stand-off JSONL

Each line is one document:

```json
{
  "doc_id": 179,
  "domain": "advance_material",
  "split": "train",
  "text": "QMOCC will be o MCDHF, GRASP2K, FAC, CIV3: the Dirac-Hartree-Fock methodologies ...",
  "entities": [
    {"id": "…uuid…", "start": 47, "end": 79, "text": "Dirac-Hartree-Fock methodologies", "type": "Method"}
  ],
  "relations": [
    {"head": "…uuid…", "tail": "…uuid…", "type": "CREATES"}
  ]
}
```

| Field | Meaning |
|---|---|
| `doc_id` | Document identifier (unique within a domain) |
| `domain` | One of the 11 domain keys (see `schema/schema.json`) |
| `split` | `train` or `test`, the predefined document-level split. Validation is 10% of `train`. |
| `text` | Document text: the abstract plus two paragraphs. Offsets are character offsets into this string. |
| `entities[].start/end` | Character offsets, `end` exclusive: `text[start:end] == entity.text` |
| `entities[].type` | One of the 11 entity types |
| `relations[].head/tail` | Entity `id`s. The relation is directed `head → tail`. |
| `relations[].type` | One of the 5 relation types. Unlisted pairs are `no_relation`. |

Duplicate labels (the same `start`, `end` and `type`) are removed. Pre-annotation labels outside the schema (for example `CARDINAL` or `DATE`) are not part of the layer.

> **Text licensing:** the full release can include the annotation layer **without** `text`, so that the source articles remain under their publishers' licenses. The offsets then refer to the curated passage of each source article.

## 2. NER training format: `data/ner/{domain}_docs_{train|test}.json`

Each record contains:

```json
{
  "sentence": "plain document text",
  "entities": "span: Type; span: Type; ...",
  "combined_text": "... <Non_Physical_Tools>DeblurGAN</Non_Physical_Tools> was designed as a <Theory>cGAN</Theory> ..."
}
```

- The training scripts read `combined_text` and convert it to 23 BIO labels (`O` plus `B-`/`I-` for each type). Tokens are split on whitespace.
- The LLM fine-tuning scripts read `sentence` + `entities`.
- `all_train.json` and `all_test.json` concatenate all domains.

## 3. RE training format: `data/re/{domain|all}/dataset.json`

Each file has the following structure:

```json
{
  "train": [ {"text": "... [E1] EMS [/E1] are computer-aided tools ... [E2] transmission [/E2] ...",
              "label": "CREATES", "head_type": "Non_Physical_Tools", "tail_type": "Parameter",
              "domain_id": 2, "doc_id": 954, "head_id": "…", "tail_id": "…"} ],
  "val":  [ ... ],
  "test": [ ... ],
  "label_list":  ["PART_OF", "CAUSES", "RELATED_TO", "CREATES", "STUDIES", "no_relation"],
  "entity_list": [ ... ],
  "domain_map":  {"advance_material": 0, "...": 10}
}
```

How the pairs are built:

- **Candidates:** all ordered pairs of distinct, non-overlapping entities in a document. There are no sentence or distance limits (paper §3.3.1).
- **Context:** a ±400-character window around the pair, with the markers `[E1] … [/E1]` and `[E2] … [/E2]`.
- **Negatives:** `no_relation` pairs are subsampled after the document split. All positive pairs are kept. The configured ratios are:

  | Split | Negatives per positive |
  |---|---|
  | train | 1.5 |
  | val | 3 |
  | test | 10 |

  Taken together, the train and val parts give ≈ 1.7 : 1, the value reported in the paper.
- **Split:** by document within each domain, multilabel-stratified with seed 42. `all/` is the concatenation of the per-domain splits, so a test document never appears in any training set.
