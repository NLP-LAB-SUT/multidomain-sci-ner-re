# Cross-Domain Training Guide

Companion for [../src/ner/train_ner_cross_domain.py](../src/ner/train_ner_cross_domain.py).

## Purpose

Train DeBERTa-v3-large NER on one/several **source** domains. Evaluate per-domain on the remaining **target** domains (never seen during training).

## Data layout

Expected under `--data-dir` (default `./data/ner`):

```
{domain}_docs_train.json
{domain}_docs_test.json
```

Each JSON = list of records with `combined_text` field containing inline `<Tag>span</Tag>` annotations.

## Valid domains

```
advance_material  aerospace  ai       chemestry  climate  electronic
energy            genetic    laser    physics    robatics
```

## Splits

| Pool | Source (in `--train-domains`) | Target (in `--test-domains` or complement) |
|------|-------------------------------|--------------------------------------------|
| train + dev | `*_train.json` (90/10 stratified split) | — |
| in-domain test | `*_test.json` | — |
| cross-domain test | — | `*_train.json` + `*_test.json` |

Target files never touch training. Both target splits go to test since model unseen either.

## CLI flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--train-domains` | required | Comma-separated source domains |
| `--test-domains` | `""` = complement | Comma-separated target domains |
| `--data-dir` | `./data/ner` | Root of `{domain}_docs_*.json` |
| `--output-dir` | required | Results root |
| `--epochs` | 50 | Max epochs |
| `--patience` | 8 | Early-stop on dev micro F1 |
| `--batch-size` | 4 | Per-device (grad_accum=4 → eff=16) |
| `--seed` | 42 | RNG seed |
| `--dev-ratio` | 0.10 | Fraction of source train for dev |
| `--no-domain-token` | off | Disable `[DOMAIN_X]` tokens (recommended cross-domain) |
| `--run-dapt` | off | MLM pretrain on source text first |

## Domain token: use or not

- **Cross-domain → use `--no-domain-token`.** Target `[DOMAIN_X]` embedding random init at inference — hurts.
- **In-domain only → omit flag.** Domain conditioning helps.

Even without flag, tokenizer registers all listed domain tokens so no crash if reused.

## Examples

Train 3 source, auto-test other 8:

```bash
python src/ner/train_ner_cross_domain.py --train-domains ai,physics,chemestry --output-dir ./results_cross/ai_phys_chem --no-domain-token
```

Explicit source + target:

```bash
python src/ner/train_ner_cross_domain.py --train-domains ai,physics --test-domains laser,energy,robatics --output-dir ./results_cross/ai_phys__laser_energy_rob --no-domain-token
```

Single-source generalization test:

```bash
python src/ner/train_ner_cross_domain.py --train-domains ai --output-dir ./results_cross/from_ai --no-domain-token
```

Add domain-adaptive pretraining:

```bash
python src/ner/train_ner_cross_domain.py --train-domains ai,physics --output-dir ./results_cross/ai_phys_dapt --no-domain-token --run-dapt
```

Short smoke test:

```bash
python src/ner/train_ner_cross_domain.py --train-domains ai --epochs 3 --patience 2 --output-dir ./results_cross/_smoke --no-domain-token
```

## Output tree

```
<output-dir>/
├── best_model.pt              # checkpoint at best dev micro F1
├── history.json               # per-epoch losses + F1
├── training_curves.png
├── indomain_summary.txt       # source *_test.json micro/macro
├── indomain/
│   ├── indomain_overall_*.txt|png
│   └── domain_<src>/indomain_*.txt|png
├── crossdomain_summary.txt    # target overall + per-domain micro/macro
└── crossdomain/
    ├── cross_overall_*.txt|png
    └── domain_<tgt>/cross_*.txt|png
```

Each domain folder holds 5 reports + 4 confusion matrices:

1. `1_seqeval.txt` — entity-level P/R/F1
2. `2_classification_with_O.txt` — token BIO w/ O
3. `3_classification_without_O.txt` — token BIO w/o O
4. `4_classification_merged_with_O.txt` — merged entity types w/ O
5. `5_classification_merged_without_O.txt` — merged w/o O

## Interpretation quick tips

- **Cross-domain micro F1 < in-domain micro F1** by wide margin → domain shift dominant. Try more source domains or `--run-dapt`.
- **Macro much lower than micro** on target → rare classes collapse. Check per-class report.
- **Per-domain variance** in `crossdomain_summary.txt` shows which targets transfer well. Similar domains (physics↔laser) transfer better than unrelated (ai↔chemestry).

## Reproducibility

Fixed seeds (Python/NumPy/Torch/CUDA). Same `--train-domains` + `--seed` → same split. Different order of `--train-domains` gives same split (grouped by domain).

## Requirements

Same env as [deberta_train_alldomain_cluade.py](deberta_train_alldomain_cluade.py): `torch`, `transformers`, `torchcrf`, `seqeval`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Unknown train domain: X` | Typo in `--train-domains` | Match spelling above (note `chemestry`, `robatics`) |
| `FileNotFoundError ..._docs_train.json` | Wrong `--data-dir` | Point to folder with JSON files |
| OOM on GPU | `max_len=512` + batch 4 | Lower `--batch-size` to 2, keep grad_accum |
| Held-out F1 near 0 | Domain token leak on unseen domain | Add `--no-domain-token` |
| Dev F1 flat all epochs | LR too low for tiny source | Edit `bert_lr` in `deberta_train_alldomain_cluade.py:Cfg` |

## Extending

- Change label set: edit `base_labels` in `Cfg`.
- Change LR/DAPT config: edit `Cfg` fields in original file (this script inherits).
- Add new domain: drop `{new}_docs_train.json` + `{new}_docs_test.json` into data dir, append name to `Cfg.domains`.
