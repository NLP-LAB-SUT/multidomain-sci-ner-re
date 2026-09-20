"""
DeBERTa-v3-large NER for a SINGLE scientific domain.

Strategy:
- DeBERTa backbone + Linear head + CRF, with Focal Loss assist (alpha = sqrt(N/n_c))
- Layer-wise learning rate decay (LLRD) on backbone
- Optional domain-adaptive pretraining (DAPT) via MLM on the single domain
- Subword handling: only first subword of each word receives a label,
  the rest get -100 (ignored by both CRF and focal loss)
- Model selection by macro-F1 on dev (not micro)
- BIO post-processing for consistency
- Per-class evaluation on test
"""

import os
import json
import re
import random
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    get_cosine_schedule_with_warmup,
)
from torchcrf import CRF

from seqeval.metrics import classification_report as seqeval_report
from seqeval.metrics import f1_score as seqeval_f1
from sklearn.metrics import classification_report as sklearn_report
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# Configuration
# =========================================================
class Cfg:
    # Pick one domain
    domain = "genetic"           # change this for each domain

    # Paths
    data_dir = r"./data/ner/"
    output_dir = rf"./results/ner_deberta_single_{domain}"

    # 11 base entity labels
    base_labels = [
        "Data", "Material", "Subject", "Parameter", "Criteria",
        "Theory", "Process", "Physical_Tools",
        "Non_Physical_Tools", "Method", "Policy",
    ]

    # Model
    model_name = "microsoft/deberta-v3-large"
    max_len = 512
    stride = 128

    # Training
    seed = 42
    batch_size = 4
    grad_accum = 4               # effective batch = 16
    epochs = 50
    patience = 8

    bert_lr = 1e-5
    head_lr = 5e-5
    crf_lr = 1e-3
    weight_decay = 0.01
    warmup_ratio = 0.1
    llrd_decay = 0.9
    dropout = 0.1
    max_grad_norm = 1.0

    # Focal loss (auxiliary to CRF)
    focal_gamma = 2.0
    focal_alpha_O = 0.2
    focal_weight = 0.2           # total = CRF + focal_weight * focal

    # DAPT (single-domain text is small, but still helps a bit)
    run_dapt = False
    dapt_epochs = 9
    dapt_lr = 5e-5
    dapt_mlm_prob = 0.15

    # Split
    dev_ratio = 0.10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# Utilities
# =========================================================
def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def parse_combined_text(text):
    pattern = r'<(.*?)>(.*?)</\1>|([^<>\s]+)'
    matches = re.findall(pattern, text)
    tokens, labels = [], []
    for tag, tagged, plain in matches:
        if tag:
            words = tagged.strip().split()
            for i, w in enumerate(words):
                tokens.append(w)
                labels.append(f"B-{tag}" if i == 0 else f"I-{tag}")
        elif plain:
            tokens.append(plain)
            labels.append("O")
    return tokens, labels

def debug_token_label_alignment(ds, tokenizer, id2tag, n_samples=5, seed=0):
    """
    n_samples نمونه‌ی تصادفی از یک NERDataset برمیداره و
    token <-> label alignment رو چاپ میکنه.
    """
    import random as _rd
    rng = _rd.Random(seed)
    n = min(n_samples, len(ds))
    idxs = rng.sample(range(len(ds)), n)

    for k, i in enumerate(idxs):
        s = ds.samples[i]
        ids = s["input_ids"]
        lbs = s["labels"]
        toks = tokenizer.convert_ids_to_tokens(ids)

        print(f"\n=== Sample {k+1}  (dataset idx={i}, length={len(ids)}) ===")
        print(f"{'POS':<5}{'TOKEN':<30}{'LABEL_ID':<10}{'TAG'}")
        print("-" * 70)
        for j, (t, lid) in enumerate(zip(toks, lbs)):
            if lid == -100:
                tag = "<ignored: subword/special/pad>"
            else:
                tag = id2tag.get(int(lid), "?")
            print(f"{j:<5}{t:<30}{str(lid):<10}{tag}")

        # خلاصه: کنترل اینکه مجموع تگ‌های non-O داخل این نمونه با
        # تعداد لیبل‌های B-/I- در داده‌ی خام جور درمیاد یا نه
        n_entities_tokens = sum(1 for x in lbs if x != -100 and id2tag[int(x)] != "O")
        n_first_subwords = sum(1 for x in lbs if x != -100)
        print(f"  -> labeled positions (first-subword): {n_first_subwords}")
        print(f"  -> non-O labels among them          : {n_entities_tokens}")
        
def load_single_domain(data_dir, domain):
    items = {"train": [], "test": []}
    for split in ("train", "test"):
        path = Path(data_dir) / f"{domain}_docs_{split}.json"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for r in data:
            if "combined_text" not in r:
                continue
            tokens, labels = parse_combined_text(r["combined_text"])
            if len(tokens) == 0:
                continue
            items[split].append({"tokens": tokens, "labels": labels})
    return items["train"], items["test"]


def split_train_dev(train_items, dev_ratio, seed):
    rng = random.Random(seed)
    lst = train_items[:]
    rng.shuffle(lst)
    n_dev = max(1, int(len(lst) * dev_ratio))
    return lst[n_dev:], lst[:n_dev]


def fix_bio(seq):
    fixed = []
    prev = "O"
    for tag in seq:
        if tag.startswith("I-"):
            ent = tag[2:]
            if prev == "O" or (prev.startswith(("B-", "I-")) and prev[2:] != ent):
                tag = "B-" + ent
        fixed.append(tag)
        prev = tag
    return fixed


def merge_bio_tags(tags):
    """Strip B-/I- prefixes (e.g. B-Material -> Material). 'O' stays 'O'."""
    return [t.split("-", 1)[1] if t != "O" else "O" for t in tags]


def _plot_cm(true, pred, labels, fname, title):
    if not labels:
        return
    cm = confusion_matrix(true, pred, labels=labels)
    size = max(6, len(labels) * 0.55)
    plt.figure(figsize=(size + 2, size))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(fname, dpi=120)
    plt.close()


def save_all_reports(true_seqs, pred_seqs, flat_true, flat_pred, output_dir, prefix=""):
    """Save the 5 requested reports + their confusion matrices."""
    os.makedirs(output_dir, exist_ok=True)
    p = prefix

    # 1) seqeval entity-level
    with open(os.path.join(output_dir, f"{p}1_seqeval.txt"), "w", encoding="utf-8") as f:
        f.write("=== Seqeval (entity-level) ===\n")
        f.write(seqeval_report(true_seqs, pred_seqs, digits=4, zero_division=0))

    # 2) classification with O
    with open(os.path.join(output_dir, f"{p}2_classification_with_O.txt"), "w", encoding="utf-8") as f:
        f.write("=== Token-level classification report (with O) ===\n")
        f.write(sklearn_report(flat_true, flat_pred, digits=4, zero_division=0))

    # 3) classification without O
    all_labels = sorted(set(flat_true) | set(flat_pred))
    labels_no_o = [l for l in all_labels if l != "O"]
    with open(os.path.join(output_dir, f"{p}3_classification_without_O.txt"), "w", encoding="utf-8") as f:
        f.write("=== Token-level classification report (without O) ===\n")
        f.write(sklearn_report(flat_true, flat_pred, labels=labels_no_o,
                               digits=4, zero_division=0))

    # 4) merged BIO with O
    merged_t = merge_bio_tags(flat_true)
    merged_p = merge_bio_tags(flat_pred)
    with open(os.path.join(output_dir, f"{p}4_classification_merged_with_O.txt"), "w", encoding="utf-8") as f:
        f.write("=== Merged BIO classification (with O) ===\n")
        f.write(sklearn_report(merged_t, merged_p, digits=4, zero_division=0))

    # 5) merged BIO without O
    merged_all = sorted(set(merged_t) | set(merged_p))
    merged_no_o = [l for l in merged_all if l != "O"]
    with open(os.path.join(output_dir, f"{p}5_classification_merged_without_O.txt"), "w", encoding="utf-8") as f:
        f.write("=== Merged BIO classification (without O) ===\n")
        f.write(sklearn_report(merged_t, merged_p, labels=merged_no_o,
                               digits=4, zero_division=0))

    # Confusion matrices
    _plot_cm(flat_true, flat_pred, all_labels,
             os.path.join(output_dir, f"{p}cm_2_with_O.png"),
             "Token-level (with O)")
    _plot_cm(flat_true, flat_pred, labels_no_o,
             os.path.join(output_dir, f"{p}cm_3_without_O.png"),
             "Token-level (without O)")
    _plot_cm(merged_t, merged_p, merged_all,
             os.path.join(output_dir, f"{p}cm_4_merged_with_O.png"),
             "Merged BIO (with O)")
    _plot_cm(merged_t, merged_p, merged_no_o,
             os.path.join(output_dir, f"{p}cm_5_merged_without_O.png"),
             "Merged BIO (without O)")


def plot_curves(history, output_dir):
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    train_loss = [h["loss"] for h in history]
    dev_loss = [h.get("dev_loss", float("nan")) for h in history]
    dev_micro = [h["dev_micro"] for h in history]
    dev_macro = [h["dev_macro"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, train_loss, marker="o", label="Train Loss")
    ax1.plot(epochs, dev_loss, marker="s", label="Dev Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Train / Dev Loss")
    ax1.legend(); ax1.grid(True)

    ax2.plot(epochs, dev_micro, marker="o", color="green", label="Dev Micro F1")
    ax2.plot(epochs, dev_macro, marker="s", color="red", label="Dev Macro F1")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("F1")
    ax2.set_title("Dev F1 (entity-level)")
    ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=120)
    plt.close()


# =========================================================
# Dataset
# =========================================================
class NERDataset(Dataset):
    def __init__(self, items, tokenizer, tag2id, max_len, stride):
        self.samples = []
        for it in items:
            tokens = list(it["tokens"])
            labels = list(it["labels"])
            enc = tokenizer(
                tokens,
                is_split_into_words=True,
                truncation=True,
                max_length=max_len,
                stride=stride,
                return_overflowing_tokens=True,
                padding=False,
            )
            for bi in range(len(enc["input_ids"])):
                input_ids = enc["input_ids"][bi]
                attention_mask = enc["attention_mask"][bi]
                word_ids = enc.word_ids(batch_index=bi)

                label_ids = []
                prev = None
                for w in word_ids:
                    if w is None:
                        label_ids.append(-100)
                    elif w != prev:
                        label_ids.append(tag2id[labels[w]])
                    else:
                        label_ids.append(-100)
                    prev = w

                self.samples.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": label_ids,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        L = max(len(x["input_ids"]) for x in batch)
        ids, am, lb = [], [], []
        for x in batch:
            pad = L - len(x["input_ids"])
            ids.append(x["input_ids"] + [self.pad_id] * pad)
            am.append(x["attention_mask"] + [0] * pad)
            lb.append(x["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(am),
            "labels": torch.tensor(lb),
        }


# =========================================================
# Losses
# =========================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0, ignore_index=-100):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        mask = targets != self.ignore_index
        if not mask.any():
            return logits.sum() * 0.0
        logits = logits[mask]
        targets = targets[mask]
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        a = self.alpha.to(logits.device)[targets]
        loss = a * loss
        return loss.mean()


# =========================================================
# Model
# =========================================================
class DebertaCRF(nn.Module):
    def __init__(self, backbone_name_or_path, num_labels, vocab_size, dropout=0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name_or_path)
        self.backbone.resize_token_embeddings(vocab_size)
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.focal = None

    def emissions(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h = self.dropout(out.last_hidden_state)
        return self.classifier(h)

    def forward(self, input_ids, attention_mask, labels=None):
        emis = self.emissions(input_ids, attention_mask)
        if labels is not None:
            crf_mask = (labels != -100).clone()
            crf_mask[:, 0] = True
            labels_clean = labels.clone()
            labels_clean[labels_clean == -100] = 0
            crf_loss = -self.crf(emis, labels_clean, mask=crf_mask, reduction="mean")
            focal_loss = torch.tensor(0.0, device=emis.device)
            if self.focal is not None:
                focal_loss = self.focal(emis.reshape(-1, emis.size(-1)),
                                        labels.reshape(-1))
            return crf_loss, focal_loss
        else:
            crf_mask = attention_mask.bool().clone()
            crf_mask[:, 0] = True
            preds = self.crf.decode(emis, mask=crf_mask)
            return preds, crf_mask


# =========================================================
# LLRD optimizer groups
# =========================================================
def build_param_groups(model, cfg):
    no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias",
                "layer_norm.weight", "layer_norm.bias")

    def split(named):
        decay, nodecay = [], []
        for n, p in named:
            (nodecay if any(nd in n for nd in no_decay) else decay).append(p)
        return decay, nodecay

    groups = []
    groups.append({"params": list(model.crf.parameters()),
                   "lr": cfg.crf_lr, "weight_decay": 0.0})

    cls_d, cls_nd = split(list(model.classifier.named_parameters()))
    if cls_d:
        groups.append({"params": cls_d, "lr": cfg.head_lr, "weight_decay": cfg.weight_decay})
    if cls_nd:
        groups.append({"params": cls_nd, "lr": cfg.head_lr, "weight_decay": 0.0})

    backbone = model.backbone
    n_layers = backbone.config.num_hidden_layers

    emb_named = list(backbone.embeddings.named_parameters())
    emb_lr = cfg.bert_lr * (cfg.llrd_decay ** (n_layers + 1))
    emb_d, emb_nd = split(emb_named)
    if emb_d:
        groups.append({"params": emb_d, "lr": emb_lr, "weight_decay": cfg.weight_decay})
    if emb_nd:
        groups.append({"params": emb_nd, "lr": emb_lr, "weight_decay": 0.0})

    for i, layer in enumerate(backbone.encoder.layer):
        depth = n_layers - i
        lr_i = cfg.bert_lr * (cfg.llrd_decay ** depth)
        l_d, l_nd = split(list(layer.named_parameters()))
        if l_d:
            groups.append({"params": l_d, "lr": lr_i, "weight_decay": cfg.weight_decay})
        if l_nd:
            groups.append({"params": l_nd, "lr": lr_i, "weight_decay": 0.0})

    handled = set()
    for p in backbone.embeddings.parameters():
        handled.add(id(p))
    for layer in backbone.encoder.layer:
        for p in layer.parameters():
            handled.add(id(p))
    extras = [(n, p) for n, p in backbone.named_parameters() if id(p) not in handled]
    if extras:
        e_d, e_nd = split(extras)
        if e_d:
            groups.append({"params": e_d, "lr": cfg.bert_lr, "weight_decay": cfg.weight_decay})
        if e_nd:
            groups.append({"params": e_nd, "lr": cfg.bert_lr, "weight_decay": 0.0})
    return groups


# =========================================================
# DAPT
# =========================================================
def run_dapt(tokenizer, texts, cfg):
    save_dir = os.path.join(cfg.output_dir, "dapt_backbone")
    if os.path.isdir(save_dir) and os.path.isfile(os.path.join(save_dir, "config.json")):
        print(f"[DAPT] reusing existing backbone at {save_dir}")
        return save_dir

    print(f"[DAPT] training MLM on {len(texts)} texts")
    lm = AutoModelForMaskedLM.from_pretrained(cfg.model_name)
    lm.resize_token_embeddings(len(tokenizer))
    lm.to(cfg.device)

    class TextDS(Dataset):
        def __init__(self, texts, tok, max_len):
            self.enc = [tok(t, truncation=True, max_length=max_len, padding=False)
                        for t in texts]

        def __len__(self):
            return len(self.enc)

        def __getitem__(self, i):
            return {k: self.enc[i][k] for k in ("input_ids", "attention_mask")}

    ds = TextDS(texts, tokenizer, cfg.max_len)
    coll = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=cfg.dapt_mlm_prob)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=coll)

    opt = AdamW(lm.parameters(), lr=cfg.dapt_lr, weight_decay=0.01)
    total = len(loader) * cfg.dapt_epochs
    sch = get_cosine_schedule_with_warmup(opt, int(0.1 * total), total)
    scaler = GradScaler()

    lm.train()
    for ep in range(cfg.dapt_epochs):
        running = 0.0
        for batch in tqdm(loader, desc=f"DAPT epoch {ep+1}"):
            batch = {k: v.to(cfg.device) for k, v in batch.items()}
            opt.zero_grad()
            with autocast():
                loss = lm(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(lm.parameters(), cfg.max_grad_norm)
            scaler.step(opt)
            scaler.update()
            sch.step()
            running += loss.item()
        print(f"[DAPT epoch {ep+1}] loss={running/len(loader):.4f}")

    os.makedirs(save_dir, exist_ok=True)
    lm.base_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"[DAPT] saved backbone to {save_dir}")

    del lm
    torch.cuda.empty_cache()
    return save_dir


# =========================================================
# Evaluation
# =========================================================
def evaluate(model, loader, id2tag, device, compute_loss=False, focal_weight=0.0):
    model.eval()
    all_true, all_pred = [], []
    flat_t, flat_p = [], []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            am = batch["attention_mask"].to(device)
            labels = batch["labels"]

            if compute_loss:
                lb = labels.to(device)
                crf_loss, focal_loss = model(ids, am, lb)
                total_loss += (crf_loss + focal_weight * focal_loss).item()
                n_batches += 1

            preds, crf_mask = model(ids, am)
            labels_np = labels.cpu().numpy()
            crf_mask_np = crf_mask.cpu().numpy()

            for i in range(len(preds)):
                pred_full = [None] * crf_mask_np.shape[1]
                ptr = 0
                for pos, m in enumerate(crf_mask_np[i]):
                    if m:
                        pred_full[pos] = id2tag[preds[i][ptr]]
                        ptr += 1
                valid = np.where(labels_np[i] != -100)[0]
                t_seq = [id2tag[int(labels_np[i][k])] for k in valid]
                p_seq = [pred_full[k] if pred_full[k] is not None else "O" for k in valid]

                t_seq = fix_bio(t_seq)
                p_seq = fix_bio(p_seq)

                all_true.append(t_seq); all_pred.append(p_seq)
                flat_t.extend(t_seq); flat_p.extend(p_seq)

    micro = seqeval_f1(all_true, all_pred, average="micro", zero_division=0)
    macro = seqeval_f1(all_true, all_pred, average="macro", zero_division=0)
    out = {
        "micro_f1": float(micro),
        "macro_f1": float(macro),
        "all_true": all_true,
        "all_pred": all_pred,
        "flat_true": flat_t,
        "flat_pred": flat_p,
    }
    if compute_loss:
        out["loss"] = total_loss / max(1, n_batches)
    return out


# =========================================================
# Main
# =========================================================
def main():
    cfg = Cfg()
    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)

    print(f"Domain: {cfg.domain}")
    train_full, test_items = load_single_domain(cfg.data_dir, cfg.domain)
    train_items, dev_items = split_train_dev(train_full, cfg.dev_ratio, cfg.seed)
    print(f"Train={len(train_items)} Dev={len(dev_items)} Test={len(test_items)}")

    unique = ["O"]
    for b in cfg.base_labels:
        unique.append(f"B-{b}")
        unique.append(f"I-{b}")
    tag2id = {t: i for i, t in enumerate(unique)}
    id2tag = {i: t for t, i in tag2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

    backbone_path = cfg.model_name
    if cfg.run_dapt:
        dapt_texts = [" ".join(it["tokens"]) for it in train_items + dev_items]
        backbone_path = run_dapt(tokenizer, dapt_texts, cfg)

    train_ds = NERDataset(train_items, tokenizer, tag2id, cfg.max_len, cfg.stride)
    dev_ds = NERDataset(dev_items, tokenizer, tag2id, cfg.max_len, cfg.stride)
    test_ds = NERDataset(test_items, tokenizer, tag2id, cfg.max_len, cfg.stride)
    debug_token_label_alignment(train_ds, tokenizer, id2tag, n_samples=5, seed=42)

    coll = Collator(tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=coll)
    dev_loader = DataLoader(dev_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=coll)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=coll)

    model = DebertaCRF(backbone_path, num_labels=len(tag2id),
                       vocab_size=len(tokenizer), dropout=cfg.dropout).to(cfg.device)

    cnt = Counter()
    for it in train_items:
        cnt.update(it["labels"])
    total = sum(cnt.values())
    alpha = torch.ones(len(tag2id))
    for t, i in tag2id.items():
        alpha[i] = (total / max(cnt.get(t, 1), 1)) ** 0.5
    alpha[tag2id["O"]] = cfg.focal_alpha_O
    alpha = alpha / alpha.mean()
    model.focal = FocalLoss(alpha=alpha, gamma=cfg.focal_gamma).to(cfg.device)

    groups = build_param_groups(model, cfg)
    optimizer = AdamW(groups)
    steps_per_epoch = max(1, len(train_loader) // cfg.grad_accum)
    total_steps = steps_per_epoch * cfg.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(cfg.warmup_ratio * total_steps), total_steps
    )
    scaler = GradScaler()

    best_micro = -1.0
    best_path = os.path.join(cfg.output_dir, "best_model.pt")
    patience = 0
    history = []

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        for step, batch in enumerate(pbar):
            ids = batch["input_ids"].to(cfg.device)
            am = batch["attention_mask"].to(cfg.device)
            lb = batch["labels"].to(cfg.device)

            with autocast():
                crf_loss, focal_loss = model(ids, am, lb)
                loss = (crf_loss + cfg.focal_weight * focal_loss) / cfg.grad_accum

            scaler.scale(loss).backward()
            running += loss.item() * cfg.grad_accum
            pbar.set_postfix(loss=f"{loss.item()*cfg.grad_accum:.3f}")

            if (step + 1) % cfg.grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

        avg_loss = running / max(1, len(train_loader))
        res = evaluate(model, dev_loader, id2tag, cfg.device,
                       compute_loss=True, focal_weight=cfg.focal_weight)
        dev_loss = res.get("loss", float("nan"))
        print(f"Epoch {epoch+1}: train_loss={avg_loss:.4f}  dev_loss={dev_loss:.4f}  "
              f"dev_micro={res['micro_f1']:.4f}  dev_macro={res['macro_f1']:.4f}")
        history.append({"epoch": epoch+1, "loss": avg_loss, "dev_loss": dev_loss,
                        "dev_micro": res["micro_f1"], "dev_macro": res["macro_f1"]})

        if res["micro_f1"] > best_micro:
            best_micro = float(res["micro_f1"])
            patience = 0
            torch.save({"state_dict": model.state_dict(),
                        "tag2id": tag2id,
                        "epoch": epoch+1,
                        "micro_f1": float(best_micro)}, best_path)
            print(f"  -> saved new best (dev micro F1 = {best_micro:.4f})")
        else:
            patience += 1
            print(f"  patience {patience}/{cfg.patience}")
            if patience >= cfg.patience:
                print("Early stopping.")
                break

    with open(os.path.join(cfg.output_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"\nLoading best model (dev micro F1 = {best_micro:.4f})...")
    ckpt = torch.load(best_path, map_location=cfg.device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    print("Final evaluation on TEST...")
    res = evaluate(model, test_loader, id2tag, cfg.device)
    print(f"\nTEST  micro={res['micro_f1']:.4f}  macro={res['macro_f1']:.4f}")

    with open(os.path.join(cfg.output_dir, "test_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"domain: {cfg.domain}\n")
        f.write(f"micro_f1: {res['micro_f1']:.4f}\n")
        f.write(f"macro_f1: {res['macro_f1']:.4f}\n")

    save_all_reports(res["all_true"], res["all_pred"],
                     res["flat_true"], res["flat_pred"],
                     cfg.output_dir, prefix="test_")
    plot_curves(history, cfg.output_dir)
    print(f"Saved all reports + plots to {cfg.output_dir}")


if __name__ == "__main__":
    main()

