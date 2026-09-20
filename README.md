# A Multi-Domain Corpus for Entity and Relation Extraction from Scientific Articles

This repository contains the baseline **training code**, the **annotation schema and guidelines**, the **annotation-layer format**, and **sample data** for a multi-domain scientific corpus. The corpus supports two tasks:

- **Named Entity Recognition (NER)**
- **Relation Extraction (RE)**

The full corpus is available **on request** (see [Data access](#data-access)).

## Corpus at a glance

| | |
|---|---|
| Documents | **3,466** multi-paragraph passages (abstract + two paragraphs) from open-access articles retrieved from OpenAlex |
| Domains | **11**: Advanced Materials & Construction, Aerospace, Artificial Intelligence, Chemistry, Climate & Environment, Electronics, Energy, Genetics, Laser & Photonics, Physics, Robotics |
| Entity annotations | **105,640** in **11 types** |
| Relation annotations | **37,824** in **5 directed types**, including cross-sentence relations |
| Split | Document-level, per domain: 80% train (2,763 docs) and 20% test (703 docs). 10% of train is held out for validation. |
| Annotation | Hybrid human–LLM. Humans label entities; an LLM adjudicator (GPT-4.1) resolves disagreements; a domain expert confirms. GPT-4.1 and DeepSeek propose relations; humans correct and finalize them. |
| Agreement (Krippendorff's α) | Independent entity agreement ≈ 0.38. After consolidation: entities ≈ 0.73, relations ≈ 0.60. |

### Schema

**Entity types (11):** `Subject`, `Theory`, `Criteria`, `Policy`, `Material`, `Physical_Tools`, `Non_Physical_Tools`, `Data`, `Process`, `Method`, `Parameter`

**Relation types (5, directed head → tail):** `PART_OF`, `CREATES`, `STUDIES`, `CAUSES`, `RELATED_TO` (plus `no_relation` for unrelated candidate pairs)

Definitions: [`schema/schema.json`](schema/schema.json). Annotation rules: [`docs/ANNOTATION_GUIDELINES.md`](docs/ANNOTATION_GUIDELINES.md).

## Repository structure

```
.
├── README.md
├── DATA_REQUEST.md               how to request the full corpus
├── LICENSE                       MIT (code)
├── DATA_LICENSE.md               CC BY-NC 4.0 (annotations)
├── CITATION.cff
├── requirements.txt
├── schema/
│   ├── schema.json               entity/relation types, definitions, domains, categories
│   └── label_studio_config.xml   Label Studio labeling interface
├── docs/
│   ├── ANNOTATION_GUIDELINES.md  annotation rules and workflow
│   ├── ANNOTATION_LAYER.md       data formats (annotation layer, NER, RE)
│   ├── CROSS_DOMAIN_NER.md       leave-out-domain NER training
│   └── CROSS_DOMAIN_RE.md        leave-out-domain RE training
├── data/
│   └── sample/
│       ├── annotation_layer_sample.jsonl   22 documents (2 per domain), stand-off format
│       ├── ner/{domain}_docs_{train,test}.json
│       └── re/all/dataset.json
└── src/
    ├── ner/
    │   ├── train_ner.py              DeBERTa-v3-large + CRF, joint 11-domain (+ DAPT)
    │   ├── train_ner_single_domain.py
    │   └── train_ner_cross_domain.py leave-one-category-out
    ├── re/
    │   ├── train_re.py               DeBERTa-v3-large + R-BERT++
    │   └── train_re_cross_domain.py  leave-one-category-out
    └── llm_finetune/
        ├── finetune_llm_full.py      Qwen3-8B / Mistral-7B token classifier, full fine-tuning
        └── finetune_llm_lora.py      same, LoRA (r = 4, α = 16, dropout 0.1)
```

## Installation

```bash
git clone https://github.com/<user>/multidomain-sci-ner-re.git
cd multidomain-sci-ner-re
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The experiments used one NVIDIA RTX 4090 (24 GB) for the encoders and one NVIDIA RTX A6000 (48 GB) for LLM fine-tuning.

## Quick start with the sample data

Run all commands from the repository root. The sample is only big enough to check that the pipeline runs; it cannot reproduce the paper's scores.

```bash
# NER: train on 2 domains, test on 1 unseen domain (smoke test)
python src/ner/train_ner_cross_domain.py --data-dir data/sample/ner \
    --train-domains ai,physics --test-domains laser \
    --epochs 2 --patience 1 --output-dir results/_smoke_ner --no-domain-token

# RE: all domains, sample pairs (smoke test)
python src/re/train_re.py --mode cross_domain --data-dir data/sample/re \
    --epochs 1 --output-dir results/_smoke_re
```

## Reproducing the paper baselines (full data)

### 1. Place the data

The full corpus is delivered as training-ready files, with the predefined splits already applied. Put them here:

```
data/ner/{domain}_docs_{train,test}.json   and   data/ner/all_{train,test}.json
data/re/{domain}/dataset.json              and   data/re/all/dataset.json
```

The formats are described in [`docs/ANNOTATION_LAYER.md`](docs/ANNOTATION_LAYER.md).

### 2. NER: DeBERTa-v3-large + CRF

```bash
python src/ner/train_ner.py
```

Settings are in the `Cfg` class at the top of the script. The defaults match the paper's reference model (Table A4):

- Domain-adaptive pretraining (DAPT) on: MLM, 8 epochs, lr 5e-5, 15% masking
- Encoder / head / CRF learning rates: 1e-5 / 5e-5 / 1e-3, with layer-wise LR decay 0.9
- Batch size 4 × gradient accumulation 4
- Up to 50 epochs, with patience 8
- Loss: CRF NLL + 0.2 × focal loss (γ = 2)
- Sliding window of 512 tokens with stride 128
- Seed 42

To train the SciBERT baseline, set `model_name = "allenai/scibert_scivocab_cased"`.

### 3. RE: DeBERTa-v3-large + R-BERT++

```bash
python src/re/train_re.py --mode cross_domain --loss ce --label-smoothing 0.1 \
    --lr 2e-5 --batch-size 16 --grad-accum 4 --epochs 20 --early-stop-patience 3 --seed 42
# per-domain model:
python src/re/train_re.py --mode per_domain --domain ai --loss ce
```

### 4. Cross-domain (leave-one-category-out)

Categories (paper Table 8): **Science** = physics, chemestry, laser, advance_material · **Engineering** = energy, aerospace · **Intelligent & Autonomous Systems** = ai, electronic, robatics · **Life & Earth Sciences** = genetic, climate.

```bash
# e.g. hold out "Engineering"
python src/ner/train_ner_cross_domain.py \
    --train-domains physics,chemestry,laser,advance_material,ai,electronic,robatics,genetic,climate \
    --test-domains energy,aerospace --output-dir results/cross_ner_E --no-domain-token

python src/re/train_re_cross_domain.py \
    --train-domains physics,chemestry,laser,advance_material,ai,electronic,robatics,genetic,climate \
    --test-domains energy,aerospace --output-dir results/cross_re_E
```

See [`docs/CROSS_DOMAIN_NER.md`](docs/CROSS_DOMAIN_NER.md) and [`docs/CROSS_DOMAIN_RE.md`](docs/CROSS_DOMAIN_RE.md) for all options.

### 5. LLM token classifiers (Qwen3-8B, Mistral-7B-Instruct-v0.3)

```bash
MODEL_NAME=Qwen/Qwen3-8B TRAIN_FILE=data/ner/all_train.json TEST_FILE=data/ner/all_test.json \
OUTPUT_DIR=results/qwen_full python src/llm_finetune/finetune_llm_full.py

MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3 OUTPUT_DIR=results/mistral_lora \
python src/llm_finetune/finetune_llm_lora.py
```

## Baseline results (test set)

| Task | Model | Metric | Score |
|---|---|---|---|
| NER | DeBERTa-v3-large + CRF + DAPT | micro-F1 / macro-F1 | **0.589** / 0.519 |
| NER | DeBERTa-v3-large + CRF (no DAPT) | micro-F1 | 0.581 |
| NER | SciBERT + CRF | micro-F1 | 0.542 |
| NER | Qwen3-8B, retrieval-augmented prompting, 10 shots | micro-F1 | 0.488 |
| NER | Qwen3-8B, full fine-tuning | micro-F1 | 0.408 |
| RE  | DeBERTa-v3-large + R-BERT++ | positive-only macro-F1 | **0.416** |
| NER | Leave-one-category-out (mean) | F1, cross- vs. in-category | 0.537 vs. 0.574 |
| RE  | Leave-one-category-out (mean) | positive-only macro-F1, cross- vs. in-category | 0.487 vs. 0.480 |

Each configuration was run once with seed 42.

## Data access

The annotation layer, guidelines, predefined splits, and training-ready files are available **from the corresponding author on reasonable request, for non-commercial research use**. See **[DATA_REQUEST.md](DATA_REQUEST.md)** for the request template.

Contact: **mohammadreza.jafari@shiftiai.com**


## License

- Code: [MIT](LICENSE)
- Annotations and samples: [CC BY-NC 4.0](DATA_LICENSE.md)
- Source texts: remain under the licenses of their original publishers
