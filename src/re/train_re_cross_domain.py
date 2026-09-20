"""
Cross-domain RE trainer built on top of re_train_v3.py.

Train on user-specified source domains, test per-domain on held-out target
domains. Reuses REModel / REDataset / RETrainer / plotting from re_train_v3.

Data source: ./prepared/all/dataset.json (cross-domain-compatible dump with
`domain_map` and `domain_id` on every sample).

Splits:
- Train = samples in data["train"] whose domain is in --train-domains
- Val   = samples in data["val"]   whose domain is in --train-domains
- In-domain test    = samples in data["test"] whose domain is in --train-domains
- Cross-domain test = ALL samples (train+val+test) whose domain is in --test-domains
                      (model never saw them)

Usage:
    python re_train_crossdomain.py --train-domains ai,physics,chemestry \
        --output-dir ./runs_cross/ai_phys_chem \
        --use-domain-emb false

    python re_train_crossdomain.py --train-domains ai --test-domains laser,energy \
        --output-dir ./runs_cross/ai__laser_energy --use-domain-emb false

Notes:
- `--use-domain-emb false` recommended: target domain ids never seen during
  training, embedding for them is random.
- Domain names must be keys of `domain_map` in dataset.json.
"""

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_re as base

from transformers import (
    AutoConfig, AutoTokenizer, TrainingArguments, EarlyStoppingCallback,
)
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, confusion_matrix,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-domains", required=True,
                   help="Comma-separated source domains (must exist in domain_map).")
    p.add_argument("--test-domains", default="",
                   help="Comma-separated target domains. "
                        "Default = every domain not in --train-domains.")
    p.add_argument("--data-dir", default="./data/re")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model-name", default=base.DEFAULT_MODEL)
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
    p.add_argument("--train-eval-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--early-stop-patience", type=int, default=4)
    p.add_argument("--type-emb-dim", type=int, default=64)
    p.add_argument("--domain-emb-dim", type=int, default=64)
    p.add_argument("--use-type-emb", type=base._str2bool, default=True)
    p.add_argument("--use-domain-emb", type=base._str2bool, default=False,
                   help="Default False for cross-domain (target ids never seen).")
    p.add_argument("--no-multi-dropout", action="store_true")
    return p.parse_args()


def filter_by_domain(samples, allowed_ids):
    return [s for s in samples if s.get("domain_id", 0) in allowed_ids]


def main():
    args = parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_path = Path(args.data_dir) / "all" / "dataset.json"
    if not data_path.exists():
        raise SystemExit(f"Expected combined dataset at {data_path}. "
                         f"Cross-domain script needs the 'all' dump.")

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    domain_map = data["domain_map"]                  # name -> id
    id_to_domain = {v: k for k, v in domain_map.items()}

    train_domains = [d.strip() for d in args.train_domains.split(",") if d.strip()]
    for d in train_domains:
        if d not in domain_map:
            raise SystemExit(f"Unknown train domain: {d}. Valid: {list(domain_map)}")

    if args.test_domains.strip():
        test_domains = [d.strip() for d in args.test_domains.split(",") if d.strip()]
        for d in test_domains:
            if d not in domain_map:
                raise SystemExit(f"Unknown test domain: {d}. Valid: {list(domain_map)}")
    else:
        test_domains = [d for d in domain_map if d not in train_domains]

    train_ids = {domain_map[d] for d in train_domains}
    test_ids = {domain_map[d] for d in test_domains}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[CONFIG] train_domains={train_domains}")
    print(f"[CONFIG] test_domains ={test_domains}")
    print(f"[CONFIG] data={data_path}  out={out_dir}")

    label_list = data["label_list"]
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    entity_list = data["entity_list"] + ["<UNK>"]
    type2id = {t: i for i, t in enumerate(entity_list)}
    num_types = len(type2id)
    num_domains = max(len(domain_map), 1)

    train_samples = filter_by_domain(data["train"], train_ids)
    val_samples = filter_by_domain(data["val"], train_ids)
    indomain_test_samples = filter_by_domain(data["test"], train_ids)
    heldout_test_samples = (filter_by_domain(data["train"], test_ids)
                            + filter_by_domain(data["val"], test_ids)
                            + filter_by_domain(data["test"], test_ids))

    print(f"[DATA] train={len(train_samples)}  val={len(val_samples)}  "
          f"in_test={len(indomain_test_samples)}  held_test={len(heldout_test_samples)}")

    if not train_samples:
        raise SystemExit("No training samples matched the source domains.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": base.SPECIAL_TOKENS})
    print(f"[TOKENIZER] added {added} special tokens; vocab={len(tokenizer)}")

    train_ds = base.REDataset(train_samples, tokenizer, label2id, type2id, args.max_length)
    val_ds = base.REDataset(val_samples, tokenizer, label2id, type2id, args.max_length) \
             if val_samples else None
    indom_ds = base.REDataset(indomain_test_samples, tokenizer, label2id, type2id,
                              args.max_length) if indomain_test_samples else None
    held_ds = base.REDataset(heldout_test_samples, tokenizer, label2id, type2id,
                             args.max_length) if heldout_test_samples else None

    if val_ds is None:
        # No val samples in source domains → fall back to a slice of train
        k = max(50, len(train_ds) // 10)
        idx = random.Random(args.seed).sample(range(len(train_ds)), min(k, len(train_ds)))
        val_ds = Subset(train_ds, idx)
        print(f"[WARN] no val samples in source domains; using {len(idx)} "
              f"random train samples as val.")

    train_eval_ds = None
    if args.train_eval_frac > 0:
        k = max(50, int(len(train_ds) * args.train_eval_frac))
        k = min(k, len(train_ds))
        idx = random.Random(args.seed).sample(range(len(train_ds)), k)
        train_eval_ds = Subset(train_ds, idx)

    cw = base.compute_class_weights(train_samples, label2id)
    print(f"[WEIGHTS] {dict(zip(label_list, [round(x.item(),3) for x in cw]))}")

    config = AutoConfig.from_pretrained(args.model_name)
    config.num_labels = len(label_list)
    config.id2label = id2label
    config.label2id = label2id
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = base.REModel(
        config=config,
        model_name=args.model_name,
        num_types=num_types,
        num_domains=num_domains,
        use_type_emb=args.use_type_emb,
        use_domain_emb=args.use_domain_emb,
        class_weights=cw.to(device),
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
        remove_unused_columns=False,
    )

    trainer = base.RETrainer(
        train_eval_dataset=train_eval_ds,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=base.REDataCollator(tokenizer),
        compute_metrics=base.make_compute_metrics(label2id),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stop_patience)],
    )

    print("\n[TRAIN] starting ...")
    trainer.train()

    final_dir = out_dir / "final_model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2, default=str)

    label_names = [id2label[i] for i in range(len(label_list))]
    pos_ids = [label2id[l] for l in label_list if l != base.NO_RELATION]

    def _eval_and_dump(ds, samples, tag):
        if ds is None:
            return None
        print(f"\n[{tag.upper()}] predict on {len(samples)} samples ...")
        out = trainer.predict(ds)
        yp = np.argmax(out.predictions, axis=1)
        yt = out.label_ids
        rep_str = classification_report(yt, yp, target_names=label_names,
                                        digits=4, zero_division=0)
        rep_dict = classification_report(yt, yp, target_names=label_names,
                                         output_dict=True, zero_division=0)
        f1p = f1_score(yt, yp, labels=pos_ids, average="macro", zero_division=0)
        f1w = f1_score(yt, yp, average="weighted", zero_division=0)
        acc = accuracy_score(yt, yp)
        print(rep_str)
        print(f"[{tag}] F1pos={f1p:.4f}  F1w={f1w:.4f}  acc={acc:.4f}")
        tag_dir = out_dir / tag
        tag_dir.mkdir(parents=True, exist_ok=True)
        (tag_dir / f"{tag}_classification_report.txt").write_text(rep_str, encoding="utf-8")
        with open(tag_dir / f"{tag}_classification_report.json", "w", encoding="utf-8") as f:
            json.dump(rep_dict, f, ensure_ascii=False, indent=2)
        cm = confusion_matrix(yt, yp, labels=list(range(len(label_list))))
        base.plot_confusion_matrix(cm, label_names,
                                   tag_dir / f"{tag}_confusion_matrix.png",
                                   normalize=False,
                                   title=f"{tag} Confusion Matrix")
        base.plot_confusion_matrix(cm, label_names,
                                   tag_dir / f"{tag}_confusion_matrix_norm.png",
                                   normalize=True,
                                   title=f"{tag} Confusion Matrix (normalized)")

        # per-domain
        per_dom = {}
        for d_name in (train_domains if tag == "indomain" else test_domains):
            d_id = domain_map[d_name]
            mask = np.array([s.get("domain_id", 0) == d_id for s in samples], dtype=bool)
            n = int(mask.sum())
            if n == 0:
                continue
            yt_d, yp_d = yt[mask], yp[mask]
            f1p_d = f1_score(yt_d, yp_d, labels=pos_ids, average="macro", zero_division=0)
            f1w_d = f1_score(yt_d, yp_d, average="weighted", zero_division=0)
            ac_d = accuracy_score(yt_d, yp_d)
            rep_d = classification_report(yt_d, yp_d, target_names=label_names,
                                          output_dict=True, zero_division=0)
            per_dom[d_name] = {
                "n": n,
                "f1_macro_positive": f1p_d,
                "f1_weighted": f1w_d,
                "accuracy": ac_d,
                "classification_report": rep_d,
            }
            dd = tag_dir / f"domain_{d_name}"
            dd.mkdir(parents=True, exist_ok=True)
            rep_d_str = classification_report(yt_d, yp_d, target_names=label_names,
                                              digits=4, zero_division=0)
            (dd / f"{tag}_{d_name}_report.txt").write_text(rep_d_str, encoding="utf-8")
            cm_d = confusion_matrix(yt_d, yp_d, labels=list(range(len(label_list))))
            base.plot_confusion_matrix(cm_d, label_names,
                                       dd / f"{tag}_{d_name}_cm.png",
                                       normalize=False,
                                       title=f"{tag} {d_name}")
            print(f"  {d_name:20s}  n={n:6d}  F1pos={f1p_d:.4f}  F1w={f1w_d:.4f}  acc={ac_d:.4f}")
        with open(tag_dir / f"{tag}_per_domain.json", "w", encoding="utf-8") as f:
            json.dump(per_dom, f, ensure_ascii=False, indent=2)
        return {"f1_macro_positive": float(f1p),
                "f1_weighted": float(f1w),
                "accuracy": float(acc),
                "per_domain": per_dom}

    indom_headline = _eval_and_dump(indom_ds, indomain_test_samples, "indomain")
    cross_headline = _eval_and_dump(held_ds, heldout_test_samples, "crossdomain")

    base.plot_history(trainer.state.log_history, out_dir)

    summary = {
        "mode": "cross_domain_holdout",
        "train_domains": train_domains,
        "test_domains": test_domains,
        "model": args.model_name,
        "data_path": str(data_path),
        "output_dir": str(out_dir),
        "use_type_emb": args.use_type_emb,
        "use_domain_emb": args.use_domain_emb,
        "best_metric": trainer.state.best_metric,
        "indomain": indom_headline,
        "crossdomain": cross_headline,
        "args": vars(args),
        "label_list": label_list,
    }
    with open(out_dir / "final_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n[DONE] outputs under: {out_dir}")
    if cross_headline:
        print(f"  cross-domain F1pos = {cross_headline['f1_macro_positive']:.4f}")
    if indom_headline:
        print(f"  in-domain    F1pos = {indom_headline['f1_macro_positive']:.4f}")


if __name__ == "__main__":
    main()
