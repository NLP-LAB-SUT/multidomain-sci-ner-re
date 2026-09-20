"""
================================================================================
 re_train_v2.py  —  Relation Extraction Trainer (with toggleable embeddings)
================================================================================

Same architecture as re_train.py, but with two new BOOL CLI flags:

    --use-type-emb   true|false   (default: true)
        Toggle the head/tail entity-type embeddings.

    --use-domain-emb true|false   (default: auto -> true for cross_domain,
                                                    false for per_domain)
        Toggle the domain embedding.

Architecture: DeBERTa-v3-large + R-BERT++  (state-of-the-art for sentence-level RE)

Representation per pair:
    [ h_CLS, h_E1, h_E2, h_E1 ⊙ h_E2, |h_E1 - h_E2|,
      (type_emb(head), type_emb(tail))?,
      (domain_emb)? ]
        → Linear → GELU → LayerNorm → multi-sample-dropout → classifier

Span pooling: MAX-pool over the inclusive span [E1] … [/E1] (vectorized).
Loss        : CrossEntropy with class weights (sqrt inv-freq) + label smoothing
              (Focal loss available via --loss focal)
Best metric : macro-F1 over POSITIVE classes only (no_relation excluded)

Run examples:
    # both ON (default for cross_domain)
    python re_train_v2.py --mode cross_domain

    # disable type embedding
    python re_train_v2.py --mode cross_domain --use-type-emb false

    # disable both (pure encoder + R-BERT++ pooling)
    python re_train_v2.py --mode cross_domain --use-type-emb false --use-domain-emb false

    # per-domain with domain emb forced on (unusual)
    python re_train_v2.py --mode per_domain --domain ai --use-domain-emb true

Outputs per run (under runs/<name>/):
    final_model/                     trained weights + tokenizer
    test_classification_report.txt   sklearn classification report
    test_classification_report.json  same, structured
    test_per_domain.json             cross-domain only: per-domain breakdown
    confusion_matrix.png             on test
    loss_curve.png                   train-step loss + epoch eval loss
    f1_curve.png                     train F1 + eval F1 (positive macro)
    history.json                     full metric history
    final_summary.json               headline metrics + args
    tensorboard logs                 under logs/
================================================================================
"""

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import (
    AutoConfig, AutoModel, AutoTokenizer,
    PreTrainedModel, Trainer, TrainingArguments,
    EarlyStoppingCallback,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from sklearn.metrics import (
    f1_score, precision_recall_fscore_support,
    accuracy_score, classification_report, confusion_matrix,
)

# ============================================================
#                     CONSTANTS
# ============================================================
SPECIAL_TOKENS = ["[E1]", "[/E1]", "[E2]", "[/E2]"]
NO_RELATION = "no_relation"
DEFAULT_MODEL = "microsoft/deberta-v3-large"


def _str2bool(x):
    """Robust bool parser for CLI."""
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"1", "true", "yes", "y", "t"}


# ============================================================
#                     DATASET
# ============================================================
class REDataset(Dataset):
    """
    Returns per-sample dict with:
        input_ids, attention_mask,
        e1_s, e1_e, e2_s, e2_e   (marker token positions)
        head_type_id, tail_type_id, domain_id, labels
    """
    def __init__(self, samples: List[Dict], tokenizer, label2id: Dict[str, int],
                 type2id: Dict[str, int], max_length: int = 384):
        self.samples = samples
        self.tok = tokenizer
        self.label2id = label2id
        self.type2id = type2id
        self.max_length = max_length

        # Pre-fetch special token ids (set AFTER add_special_tokens in main)
        self.e1_id  = tokenizer.convert_tokens_to_ids("[E1]")
        self.e1e_id = tokenizer.convert_tokens_to_ids("[/E1]")
        self.e2_id  = tokenizer.convert_tokens_to_ids("[E2]")
        self.e2e_id = tokenizer.convert_tokens_to_ids("[/E2]")
        assert -1 not in (self.e1_id, self.e1e_id, self.e2_id, self.e2e_id), \
            "Special tokens not registered in tokenizer!"

        # Unknown-type fallback id
        self.unk_type_id = type2id.get("<UNK>", 0)

    def __len__(self):
        return len(self.samples)

    def _find_first(self, ids: torch.Tensor, tok_id: int) -> int:
        pos = (ids == tok_id).nonzero(as_tuple=True)[0]
        return pos[0].item() if pos.numel() > 0 else -1

    def __getitem__(self, idx):
        s = self.samples[idx]
        enc = self.tok(
            s["text"],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )
        ids = enc["input_ids"].squeeze(0)
        mask = enc["attention_mask"].squeeze(0)

        e1_s = self._find_first(ids, self.e1_id)
        e1_e = self._find_first(ids, self.e1e_id)
        e2_s = self._find_first(ids, self.e2_id)
        e2_e = self._find_first(ids, self.e2e_id)

        # Fallback for missing markers (truncation edge case) → use CLS index 0
        if e1_s < 0: e1_s = 0
        if e1_e < 0 or e1_e < e1_s: e1_e = e1_s
        if e2_s < 0: e2_s = 0
        if e2_e < 0 or e2_e < e2_s: e2_e = e2_s

        return {
            "input_ids":     ids,
            "attention_mask": mask,
            "e1_s": e1_s, "e1_e": e1_e,
            "e2_s": e2_s, "e2_e": e2_e,
            "head_type_id": self.type2id.get(s["head_type"], self.unk_type_id),
            "tail_type_id": self.type2id.get(s["tail_type"], self.unk_type_id),
            "domain_id":    s.get("domain_id", 0),
            "labels":       self.label2id[s["label"]],
        }


class REDataCollator:
    """Pads input_ids/attention_mask, stacks scalar fields."""
    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        B = len(batch)
        ids = torch.full((B, max_len), self.pad_id, dtype=torch.long)
        msk = torch.zeros((B, max_len), dtype=torch.long)
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            ids[i, :L] = b["input_ids"]
            msk[i, :L] = b["attention_mask"]
        out = {
            "input_ids":      ids,
            "attention_mask": msk,
            "e1_s": torch.tensor([b["e1_s"] for b in batch], dtype=torch.long),
            "e1_e": torch.tensor([b["e1_e"] for b in batch], dtype=torch.long),
            "e2_s": torch.tensor([b["e2_s"] for b in batch], dtype=torch.long),
            "e2_e": torch.tensor([b["e2_e"] for b in batch], dtype=torch.long),
            "head_type_id": torch.tensor([b["head_type_id"] for b in batch], dtype=torch.long),
            "tail_type_id": torch.tensor([b["tail_type_id"] for b in batch], dtype=torch.long),
            "domain_id":    torch.tensor([b["domain_id"]    for b in batch], dtype=torch.long),
            "labels":       torch.tensor([b["labels"]       for b in batch], dtype=torch.long),
        }
        return out


# ============================================================
#                     LOSSES
# ============================================================
def focal_loss(logits, target, weight=None, gamma=2.0):
    ce = F.cross_entropy(logits, target, weight=weight, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


# ============================================================
#                     MODEL (R-BERT++ with toggleable embeddings)
# ============================================================
class REModel(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config, model_name: str,
                 num_types: int, num_domains: int,
                 use_type_emb: bool = True,
                 use_domain_emb: bool = False,
                 class_weights: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.1,
                 type_emb_dim: int = 64,
                 domain_emb_dim: int = 64,
                 use_focal: bool = False,
                 focal_gamma: float = 2.0):
        super().__init__(config)
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        H = config.hidden_size

        self.use_type_emb = use_type_emb
        self.use_domain_emb = use_domain_emb

        extra = 0
        if use_type_emb:
            self.type_emb = nn.Embedding(num_types, type_emb_dim)
            extra += 2 * type_emb_dim
        if use_domain_emb:
            self.domain_emb = nn.Embedding(num_domains, domain_emb_dim)
            extra += domain_emb_dim

        # Concat: [h_CLS, h_E1, h_E2, h_E1*h_E2, |h_E1-h_E2|, (head_emb, tail_emb)?, (dom_emb)?]
        feat_size = 5 * H + extra

        self.feature_layer = nn.Sequential(
            nn.Linear(feat_size, H),
            nn.GELU(),
            nn.LayerNorm(H),
        )
        # Multi-sample dropout — common SOTA trick for stability + small +F1
        self.ms_dropouts = nn.ModuleList(
            [nn.Dropout(p) for p in [0.1, 0.2, 0.3, 0.4, 0.5]])
        self.classifier = nn.Linear(H, config.num_labels)

        self.class_weights = class_weights
        self.label_smoothing = label_smoothing
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma

    def _span_max_pool(self, seq, s_idx, e_idx, attention_mask, fallback):
        """
        Vectorized max-pool over inclusive [s_idx, e_idx] per sample.
        seq:      [B, L, H]
        s_idx,e_idx: [B]   (long)
        attention_mask: [B, L]
        fallback: [B, H] used when span is empty/invalid
        returns: [B, H]
        """
        B, L, H = seq.shape
        positions = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, -1)  # [B, L]
        span_mask = (positions >= s_idx.unsqueeze(1)) & (positions <= e_idx.unsqueeze(1))
        span_mask = span_mask & attention_mask.bool()  # don't pool over padding

        neg_inf = torch.finfo(seq.dtype).min / 2
        masked = seq.masked_fill(~span_mask.unsqueeze(-1), neg_inf)
        pooled = masked.max(dim=1).values  # [B, H]

        empty = ~span_mask.any(dim=1)
        if empty.any():
            pooled = torch.where(empty.unsqueeze(-1), fallback, pooled)
        return pooled

    def forward(self, input_ids, attention_mask,
                e1_s, e1_e, e2_s, e2_e,
                head_type_id, tail_type_id, domain_id,
                labels=None, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        seq = out.last_hidden_state  # [B, L, H]
        h_cls = seq[:, 0, :]

        h_e1 = self._span_max_pool(seq, e1_s, e1_e, attention_mask, h_cls)
        h_e2 = self._span_max_pool(seq, e2_s, e2_e, attention_mask, h_cls)

        prod = h_e1 * h_e2
        diff = torch.abs(h_e1 - h_e2)

        parts = [h_cls, h_e1, h_e2, prod, diff]
        if self.use_type_emb:
            parts.append(self.type_emb(head_type_id))
            parts.append(self.type_emb(tail_type_id))
        if self.use_domain_emb:
            parts.append(self.domain_emb(domain_id))
        feat = torch.cat(parts, dim=-1)
        feat = self.feature_layer(feat)

        if self.training and len(self.ms_dropouts) > 0:
            logits = torch.stack(
                [self.classifier(do(feat)) for do in self.ms_dropouts]
            ).mean(dim=0)
        else:
            logits = self.classifier(feat)

        loss = None
        if labels is not None:
            if self.use_focal:
                loss = focal_loss(logits, labels,
                                  weight=self.class_weights,
                                  gamma=self.focal_gamma)
            else:
                loss = F.cross_entropy(
                    logits, labels,
                    weight=self.class_weights,
                    label_smoothing=self.label_smoothing,
                )

        return SequenceClassifierOutput(loss=loss, logits=logits)


# ============================================================
#                 CLASS WEIGHTS + METRICS
# ============================================================
def compute_class_weights(samples: List[Dict],
                          label2id: Dict[str, int]) -> torch.Tensor:
    counts = Counter(s["label"] for s in samples)
    n_classes = len(label2id)
    total = sum(counts.values())
    w = torch.zeros(n_classes, dtype=torch.float)
    for lbl, c in counts.items():
        i = label2id[lbl]
        if c > 0:
            # sqrt inverse-frequency — proven safer than raw inv-freq
            w[i] = (total / (n_classes * c)) ** 0.5
        else:
            w[i] = 1.0
    # Optionally dampen no_relation weight a bit (it's the easy class)
    if NO_RELATION in label2id:
        w[label2id[NO_RELATION]] = max(0.5, w[label2id[NO_RELATION]] * 0.85)
    return w


def make_compute_metrics(label2id: Dict[str, int]):
    pos_ids = [i for l, i in label2id.items() if l != NO_RELATION]

    def fn(pred):
        preds = np.argmax(pred.predictions, axis=1)
        labels = pred.label_ids
        f1_pos = f1_score(labels, preds, labels=pos_ids,
                          average="macro", zero_division=0)
        f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
        f1_weighted = f1_score(labels, preds, average="weighted",
                               zero_division=0)
        acc = accuracy_score(labels, preds)
        return {
            "f1_macro_positive": f1_pos,
            "f1_macro_all": f1_macro,
            "f1_weighted": f1_weighted,
            "accuracy": acc,
        }
    return fn


# ============================================================
#               CUSTOM TRAINER (double-eval)
# ============================================================
class RETrainer(Trainer):
    """Adds per-epoch evaluation on a fixed train subset for train/val curves.

    IMPORTANT: train-eval uses `evaluation_loop` directly (NOT `super().evaluate`)
    so that it does NOT trigger callbacks (EarlyStopping, BestMetric, etc.).
    This avoids the spurious 'early stopping required metric_for_best_model' warning
    and prevents EarlyStopping from being confused by missing eval_ keys.
    """
    def __init__(self, train_eval_dataset=None, **kw):
        super().__init__(**kw)
        self.train_eval_dataset = train_eval_dataset

    def evaluate(self, eval_dataset=None, ignore_keys=None,
                 metric_key_prefix="eval"):
        merged = {}

        # 1) Train-eval first, WITHOUT firing callbacks
        if (self.train_eval_dataset is not None
                and metric_key_prefix == "eval"):
            dataloader = self.get_eval_dataloader(self.train_eval_dataset)
            output = self.evaluation_loop(
                dataloader,
                description="Train-Eval",
                prediction_loss_only=None,
                ignore_keys=ignore_keys,
                metric_key_prefix="train",
            )
            # log to history (visible in plots / tensorboard) — but no callbacks
            self.log(output.metrics)
            merged.update(output.metrics)

        # 2) Real val evaluation (this DOES fire callbacks — EarlyStopping etc.)
        val_metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        merged.update(val_metrics)
        return merged


# ============================================================
#                     PLOTTING
# ============================================================
def plot_history(log_history: List[Dict], out_dir: Path):
    """Parse Trainer.state.log_history → loss / F1 curves."""
    train_step_loss = [(l["step"], l["loss"])
                       for l in log_history
                       if "loss" in l and "eval_loss" not in l
                       and "train_loss" not in l]
    eval_loss   = [(l["step"], l["eval_loss"])
                   for l in log_history if "eval_loss" in l]
    train_loss  = [(l["step"], l["train_loss"])
                   for l in log_history if "train_loss" in l]
    eval_f1     = [(l["step"], l["eval_f1_macro_positive"])
                   for l in log_history if "eval_f1_macro_positive" in l]
    train_f1    = [(l["step"], l["train_f1_macro_positive"])
                   for l in log_history if "train_f1_macro_positive" in l]

    # --- Loss curves ---
    fig, ax = plt.subplots(figsize=(11, 5.5))
    if train_step_loss:
        s, v = zip(*train_step_loss)
        ax.plot(s, v, alpha=0.4, label="train loss (per step)")
    if train_loss:
        s, v = zip(*train_loss)
        ax.plot(s, v, "o-", color="C0", label="train loss (epoch)",
                linewidth=2)
    if eval_loss:
        s, v = zip(*eval_loss)
        ax.plot(s, v, "s-", color="C3", label="eval loss (epoch)",
                linewidth=2)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Loss curves")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=110)
    plt.close(fig)

    # --- F1 curves ---
    fig, ax = plt.subplots(figsize=(11, 5.5))
    if train_f1:
        s, v = zip(*train_f1)
        ax.plot(s, v, "o-", color="C0",
                label="train F1 (macro, positive)", linewidth=2)
    if eval_f1:
        s, v = zip(*eval_f1)
        ax.plot(s, v, "s-", color="C3",
                label="eval F1 (macro, positive)", linewidth=2)
    ax.set_xlabel("step")
    ax.set_ylabel("Macro-F1 (positive classes)")
    ax.set_ylim(0, 1.0)
    ax.set_title("F1 curves")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "f1_curve.png", dpi=110)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, labels: List[str], path: Path,
                          normalize: bool = False, title: str = ""):
    if normalize:
        cm_n = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        data, fmt = cm_n, ".2f"
    else:
        data, fmt = cm, "d"
    fig, ax = plt.subplots(figsize=(max(8, 1.0 * len(labels)),
                                    max(7, 0.9 * len(labels))))
    sns.heatmap(data, annot=True, fmt=fmt,
                xticklabels=labels, yticklabels=labels,
                cmap="Blues", ax=ax, cbar=True)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ============================================================
#                        MAIN
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["per_domain", "cross_domain"], required=True)
    p.add_argument("--domain", type=str, default=None,
                   help="Required when --mode per_domain")
    p.add_argument("--data-dir", type=str, default="./data/re")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--model-name", type=str, default=DEFAULT_MODEL)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--loss", choices=["ce", "focal"], default="focal")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--train-eval-frac", type=float, default=0.1,
                   help="Fraction of train used for per-epoch train F1 (set 0 to disable)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16",
                   help="bf16 recommended for DeBERTa-v3 on Ampere+")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--early-stop-patience", type=int, default=4)
    p.add_argument("--type-emb-dim", type=int, default=64)
    p.add_argument("--domain-emb-dim", type=int, default=64)
    p.add_argument("--use-type-emb", type=_str2bool, default=True,
                   help="Use head/tail entity-type embeddings (bool: true/false). Default: true")
    p.add_argument("--use-domain-emb", type=_str2bool, default=None,
                   help="Use domain embedding (bool: true/false). Default: auto "
                        "(true for cross_domain, false for per_domain)")
    p.add_argument("--no-multi-dropout", action="store_true")
    args = p.parse_args()

    # ---- seeding ----
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ---- resolve paths ----
    if args.mode == "per_domain":
        if not args.domain:
            raise SystemExit("--domain is required for --mode per_domain")
        data_path = Path(args.data_dir) / args.domain / "dataset.json"
        run_name = f"per_domain_{args.domain}"
    else:
        data_path = Path(args.data_dir) / "all" / "dataset.json"
        run_name = "cross_domain"
    out_dir = Path(args.output_dir) if args.output_dir else (Path("runs") / run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[CONFIG] mode={args.mode}  data={data_path}  out={out_dir}")
    print(f"[CONFIG] model={args.model_name}  precision={args.precision}")

    # ---- load data ----
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    label_list = data["label_list"]
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    entity_list = data["entity_list"] + ["<UNK>"]
    type2id = {t: i for i, t in enumerate(entity_list)}
    num_types = len(type2id)

    domain_map = data["domain_map"]
    num_domains = max(len(domain_map), 1)
    # use_domain_emb: explicit CLI overrides; otherwise auto (cross_domain only)
    if args.use_domain_emb is None:
        use_domain_emb = (args.mode == "cross_domain")
    else:
        use_domain_emb = args.use_domain_emb
    use_type_emb = args.use_type_emb
    print(f"[ARCH]  use_type_emb={use_type_emb}  use_domain_emb={use_domain_emb}")

    print(f"[DATA] labels={label_list}  entity_types={len(entity_list)-1}  "
          f"domains={num_domains}")
    print(f"[DATA] sizes: train={len(data['train']):,}  "
          f"val={len(data['val']):,}  test={len(data['test']):,}")

    # ---- tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": SPECIAL_TOKENS})
    print(f"[TOKENIZER] added {added} special tokens "
          f"→ vocab={len(tokenizer)}")

    # ---- datasets ----
    train_ds = REDataset(data["train"], tokenizer, label2id, type2id, args.max_length)
    val_ds   = REDataset(data["val"],   tokenizer, label2id, type2id, args.max_length)
    test_ds  = REDataset(data["test"],  tokenizer, label2id, type2id, args.max_length)

    # train-eval subset for per-epoch train F1
    train_eval_ds = None
    if args.train_eval_frac > 0:
        k = max(50, int(len(train_ds) * args.train_eval_frac))
        k = min(k, len(train_ds))
        idx = random.Random(args.seed).sample(range(len(train_ds)), k)
        train_eval_ds = Subset(train_ds, idx)
        print(f"[TRAIN-EVAL] using {k} samples ({100*k/len(train_ds):.1f}%) "
              f"for per-epoch train F1")

    # ---- class weights ----
    cw = compute_class_weights(data["train"], label2id)
    print(f"[WEIGHTS] {dict(zip(label_list, [round(x.item(),3) for x in cw]))}")

    # ---- model ----
    config = AutoConfig.from_pretrained(args.model_name)
    config.num_labels = len(label_list)
    config.id2label = id2label
    config.label2id = label2id

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cw_dev = cw.to(device)

    model = REModel(
        config=config,
        model_name=args.model_name,
        num_types=num_types,
        num_domains=num_domains,
        use_type_emb=use_type_emb,
        use_domain_emb=use_domain_emb,
        class_weights=cw_dev,
        label_smoothing=args.label_smoothing,
        type_emb_dim=args.type_emb_dim,
        domain_emb_dim=args.domain_emb_dim,
        use_focal=(args.loss == "focal"),
        focal_gamma=args.focal_gamma,
    )
    model.encoder.resize_token_embeddings(len(tokenizer))
    if args.no_multi_dropout:
        model.ms_dropouts = nn.ModuleList([nn.Dropout(0.1)])
    if args.gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.to(device)

    # ---- training args ----
    fp16 = args.precision == "fp16"
    bf16 = args.precision == "bf16"
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro_positive",
        greater_is_better=True,
        fp16=fp16,
        bf16=bf16,
        logging_dir=str(out_dir / "logs"),
        logging_steps=25,
        save_total_limit=2,
        seed=args.seed,
        report_to=["tensorboard"],
        dataloader_num_workers=2,
        remove_unused_columns=False,   # keep our custom keys (e1_s, etc.)
    )

    trainer = RETrainer(
        train_eval_dataset=train_eval_ds,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=REDataCollator(tokenizer),
        compute_metrics=make_compute_metrics(label2id),
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=args.early_stop_patience)],
    )

    # ============================================================
    #                       TRAIN
    # ============================================================
    print("\n[TRAIN] starting ...")
    trainer.train()

    # save artifacts
    final_dir = out_dir / "final_model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # save metric history
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2,
                  default=str)

    # ============================================================
    #                       EVALUATE TEST
    # ============================================================
    print("\n[TEST] running predictions ...")
    test_out = trainer.predict(test_ds)
    y_pred = np.argmax(test_out.predictions, axis=1)
    y_true = test_out.label_ids

    label_names = [id2label[i] for i in range(len(label_list))]
    pos_ids = [label2id[l] for l in label_list if l != NO_RELATION]
    pos_names = [l for l in label_list if l != NO_RELATION]

    # ---- classification report (all classes) ----
    report_str = classification_report(
        y_true, y_pred, target_names=label_names, digits=4, zero_division=0)
    report_dict = classification_report(
        y_true, y_pred, target_names=label_names,
        output_dict=True, zero_division=0)
    print("\n" + report_str)
    (out_dir / "test_classification_report.txt").write_text(report_str, encoding="utf-8")
    with open(out_dir / "test_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report_dict, f, ensure_ascii=False, indent=2)

    # ---- positive-only macro F1 (headline metric) ----
    f1_pos = f1_score(y_true, y_pred, labels=pos_ids,
                      average="macro", zero_division=0)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    print(f"\n[HEADLINE] macro-F1 (positive) = {f1_pos:.4f}   "
          f"weighted-F1 = {f1_w:.4f}   acc = {acc:.4f}")

    # ---- confusion matrices ----
    cm = confusion_matrix(y_true, y_pred,
                          labels=list(range(len(label_list))))
    plot_confusion_matrix(
        cm, label_names, out_dir / "confusion_matrix.png",
        normalize=False, title="Test Confusion Matrix")
    plot_confusion_matrix(
        cm, label_names, out_dir / "confusion_matrix_normalized.png",
        normalize=True, title="Test Confusion Matrix (row-normalized)")

    # ---- per-domain breakdown (cross-domain only) ----
    if args.mode == "cross_domain":
        print("\n[PER-DOMAIN BREAKDOWN]")
        per_dom = {}
        for d_name, d_id in domain_map.items():
            mask = np.array(
                [s["domain_id"] == d_id for s in data["test"]], dtype=bool)
            n = int(mask.sum())
            if n == 0:
                continue
            yt = y_true[mask]; yp = y_pred[mask]
            f1p = f1_score(yt, yp, labels=pos_ids,
                           average="macro", zero_division=0)
            f1w = f1_score(yt, yp, average="weighted", zero_division=0)
            ac  = accuracy_score(yt, yp)
            rep = classification_report(yt, yp, target_names=label_names,
                                        output_dict=True, zero_division=0)
            per_dom[d_name] = {
                "n_test": n,
                "f1_macro_positive": f1p,
                "f1_weighted": f1w,
                "accuracy": ac,
                "classification_report": rep,
            }
            print(f"  {d_name:20s}  n={n:7d}  "
                  f"F1pos={f1p:.4f}  F1w={f1w:.4f}  acc={ac:.4f}")
        with open(out_dir / "test_per_domain.json", "w", encoding="utf-8") as f:
            json.dump(per_dom, f, ensure_ascii=False, indent=2)

    # ============================================================
    #                       PLOTS
    # ============================================================
    plot_history(trainer.state.log_history, out_dir)
    print(f"\n[PLOTS] saved: loss_curve.png, f1_curve.png, "
          f"confusion_matrix.png, confusion_matrix_normalized.png")

    # ============================================================
    #                       SUMMARY
    # ============================================================
    summary = {
        "mode": args.mode,
        "domain": args.domain,
        "model": args.model_name,
        "data_path": str(data_path),
        "output_dir": str(out_dir),
        "use_type_emb": use_type_emb,
        "use_domain_emb": use_domain_emb,
        "best_metric": trainer.state.best_metric,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "headline_test": {
            "f1_macro_positive": float(f1_pos),
            "f1_weighted": float(f1_w),
            "accuracy": float(acc),
        },
        "args": vars(args),
        "label_list": label_list,
    }
    with open(out_dir / "final_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n[DONE] All outputs under: {out_dir}")
    print(f"       Headline: macro-F1 (positive) = {f1_pos:.4f}")


if __name__ == "__main__":
    main()

