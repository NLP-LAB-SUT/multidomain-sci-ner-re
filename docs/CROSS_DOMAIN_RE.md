# RE Cross-Domain Training Guide

Companion for [../src/re/train_re_cross_domain.py](../src/re/train_re_cross_domain.py). Extends [re_train_v3.py](../src/re/train_re.py) for hold-out-domain evaluation.

## Purpose

Train R-BERT++ (DeBERTa-v3-large) relation extractor on selected source domains. Evaluate per-domain on the remaining target domains never seen during training.

## Data requirement

Uses the combined dump:

```
<data-dir>/all/dataset.json
```

Same file `re_train_v3.py --mode cross_domain` reads. Must contain:

- `label_list`, `entity_list`
- `domain_map` = {domain_name: domain_id}
- `train`, `val`, `test` = lists of samples each with `domain_id`

If missing, generate it with whatever preparation script produced `./data/re/all/dataset.json` previously.

## Splits

| Pool | Definition |
|------|-----------|
| train | `data["train"]` where `domain_id ∈ source` |
| val | `data["val"]`   where `domain_id ∈ source` |
| in-domain test | `data["test"]`  where `domain_id ∈ source` |
| cross-domain test | `data["train"] + data["val"] + data["test"]` where `domain_id ∈ target` |

Target domains contribute nothing to training. All their splits go to held-out test since model unseen either.

Fallback: if no val samples exist for source domains (small source set), script auto-slices 10% of train as val.

## CLI flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--train-domains` | required | Comma list of source domain names |
| `--test-domains` | `""` = complement | Target domains (else auto) |
| `--data-dir` | `./data/re` | Root containing `all/dataset.json` |
| `--output-dir` | required | Results directory |
| `--model-name` | `microsoft/deberta-v3-large` | Encoder |
| `--max-length` | 512 | Token truncation |
| `--batch-size` | 16 | Per device |
| `--eval-batch-size` | 16 | |
| `--grad-accum` | 4 | Effective batch = 64 |
| `--lr` | 2e-5 | AdamW LR |
| `--epochs` | 20 | Max epochs |
| `--warmup-ratio` | 0.1 | Linear warmup |
| `--weight-decay` | 0.01 | |
| `--label-smoothing` | 0.1 | CE only |
| `--loss` | `focal` | `ce` or `focal` |
| `--focal-gamma` | 2.0 | Focal γ |
| `--train-eval-frac` | 0.1 | Train subset for per-epoch train F1 |
| `--seed` | 42 | |
| `--precision` | `bf16` | `fp32` / `fp16` / `bf16` |
| `--gradient-checkpointing` | off | Cuts VRAM |
| `--early-stop-patience` | 4 | Epochs w/o dev F1 gain |
| `--type-emb-dim` | 64 | Entity-type embedding size |
| `--domain-emb-dim` | 64 | Domain embedding size |
| `--use-type-emb` | true | Head/tail type embeddings |
| `--use-domain-emb` | **false** | Domain embedding (default OFF here) |
| `--no-multi-dropout` | off | Disable multi-sample dropout |

## Domain embedding: keep OFF

Cross-domain default `--use-domain-emb false`. Target domain id embeddings never trained → random noise at inference. Only enable when target ids overlap source ids (unusual).

Type embeddings safe to keep on: entity types shared across domains.

## Examples

Train 3 source, auto-test other 8:

```bash
python src/re/train_re_cross_domain.py --train-domains ai,physics,chemestry --output-dir ./runs_cross/ai_phys_chem
```

Explicit source + target:

```bash
python src/re/train_re_cross_domain.py --train-domains ai,physics --test-domains laser,energy,robatics --output-dir ./runs_cross/ai_phys__laser_energy_rob
```

Single-source transfer:

```bash
python src/re/train_re_cross_domain.py --train-domains ai --output-dir ./runs_cross/from_ai
```

Ablate type embeddings:

```bash
python src/re/train_re_cross_domain.py --train-domains ai,physics --output-dir ./runs_cross/ai_phys_notype --use-type-emb false
```

Low-VRAM smoke test:

```bash
python src/re/train_re_cross_domain.py --train-domains ai --epochs 3 --early-stop-patience 2 --batch-size 8 --gradient-checkpointing --output-dir ./runs_cross/_smoke
```

## Output tree

```
<output-dir>/
├── final_model/                    # trained weights + tokenizer
├── history.json                    # trainer log_history
├── loss_curve.png
├── f1_curve.png
├── final_summary.json              # headline metrics + args
├── indomain/
│   ├── indomain_classification_report.txt|json
│   ├── indomain_confusion_matrix(_norm).png
│   ├── indomain_per_domain.json
│   └── domain_<src>/indomain_<src>_report.txt + _cm.png
└── crossdomain/
    ├── crossdomain_classification_report.txt|json
    ├── crossdomain_confusion_matrix(_norm).png
    ├── crossdomain_per_domain.json
    └── domain_<tgt>/crossdomain_<tgt>_report.txt + _cm.png
```

## Metric definitions

- **f1_macro_positive** — macro-F1 over positive labels (excludes `no_relation`). Primary headline.
- **f1_weighted** — support-weighted F1 over all labels.
- **accuracy** — plain accuracy.
- **best_metric** in summary = best `eval_f1_macro_positive` during training on val.

## Interpretation

- Cross-domain F1pos ≪ in-domain F1pos → strong domain shift. Add more source domains or richer source variety.
- One target near random → its relation distribution disjoint from source. Check per-domain `classification_report`.
- Macro much below weighted → rare positive classes collapsed to `no_relation`. Try `--loss focal --focal-gamma 3.0` or raise minority weight in `re_train_v3.compute_class_weights`.

## Reproducibility

Seeds fixed (Python/NumPy/Torch/CUDA). Same source list + seed = same train/val filter and dropout mask. Domain order in `--train-domains` doesn't matter (set membership).

## Requirements

Same env as `re_train_v3.py`: `torch`, `transformers`, `scikit-learn`, `matplotlib`, `seaborn`, `tensorboard`.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Expected combined dataset at ...` | Missing `all/dataset.json` | Regenerate the combined dump used by `--mode cross_domain` |
| `Unknown train domain: X` | Name not in `domain_map` | Print `domain_map` keys from JSON |
| `No training samples matched` | Source domains have no train split | Verify `domain_id` field in samples |
| Cross-domain F1 = 0 for a target | Domain embedding leak | Ensure `--use-domain-emb false` (default) |
| OOM | DeBERTa-large + 512 | `--gradient-checkpointing`, `--batch-size 8`, keep `--grad-accum 8` |
| Trainer warns about EarlyStopping metric | Harmless — see `RETrainer` docstring | Ignore |

## Extending

- Different metric-for-best: change `metric_for_best_model` in `TrainingArguments` (must be a key `make_compute_metrics` returns with `eval_` prefix).
- Weighted domain sampling: subclass `torch.utils.data.Sampler`, pass to `Trainer` via `get_train_sampler` override.
- Add DAPT: pretrain MLM on source samples first, then pass its checkpoint via `--model-name`.
