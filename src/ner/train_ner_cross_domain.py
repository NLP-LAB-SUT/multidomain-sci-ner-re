"""
Cross-domain DeBERTa-v3-large NER.

Difference vs deberta_train_alldomain_cluade.py:
- Train (and dev split) uses ONLY user-specified `--train-domains`.
- Test uses the REMAINING domains (or explicit `--test-domains`), evaluated
  per-domain. For held-out test domains, BOTH the train and test JSON files
  of that domain are treated as unseen test data.
- Optional `--no-domain-token` to run a domain-agnostic model (recommended
  when test domains are unseen so the model does not see a new [DOMAIN_*]
  token at inference time).

Usage:
    # train on 3 domains, auto-test on the other 8
    python deberta_train_crossdomain.py \
        --train-domains ai,physics,chemestry \
        --output-dir ./results_cross/ai_phys_chem \
        --no-domain-token

    # explicit train + test lists
    python deberta_train_crossdomain.py \
        --train-domains ai,physics \
        --test-domains laser,energy,robatics \
        --output-dir ./results_cross/ai_phys__laser_energy_rob \
        --no-domain-token

    # single-domain source
    python deberta_train_crossdomain.py --train-domains ai --no-domain-token

Notes:
- Domain names must match keys in Cfg.domains from the original script.
- With --no-domain-token the model has no leakage of domain identity.
  If you keep domain tokens on, unseen test domains still work (unknown
  [DOMAIN_*] token embedding is randomly initialised) but performance is
  usually worse than the agnostic setting.
"""

import argparse
import os
import sys
from pathlib import Path

# reuse everything from the original training script
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_ner as base

import json
import random
from collections import Counter, defaultdict

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-domains", required=True,
                    help="Comma-separated list of source domains.")
    ap.add_argument("--test-domains", default="",
                    help="Comma-separated list of target domains. "
                         "Default = all domains not in --train-domains.")
    ap.add_argument("--data-dir", default=base.Cfg.data_dir)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=base.Cfg.epochs)
    ap.add_argument("--patience", type=int, default=base.Cfg.patience)
    ap.add_argument("--batch-size", type=int, default=base.Cfg.batch_size)
    ap.add_argument("--seed", type=int, default=base.Cfg.seed)
    ap.add_argument("--dev-ratio", type=float, default=base.Cfg.dev_ratio)
    ap.add_argument("--no-domain-token", action="store_true",
                    help="Disable [DOMAIN_X] special tokens (recommended for "
                         "cross-domain).")
    ap.add_argument("--run-dapt", action="store_true")
    return ap.parse_args()


def load_domain_split(data_dir, train_domains, test_domains):
    """
    train_domains: build train+dev pool from their *_train.json AND *_test.json
                   files? -> No. Follow original convention:
                   train+dev = *_train.json of train_domains
                   in-domain test (for reference) = *_test.json of train_domains
    test_domains : held-out. Use BOTH *_train.json and *_test.json as test data
                   (no leakage — model never trained on them).
    """
    train_pool = []
    indomain_test = []
    heldout_test = []

    for d in train_domains:
        p_tr = Path(data_dir) / f"{d}_docs_train.json"
        p_te = Path(data_dir) / f"{d}_docs_test.json"
        for path, bucket in ((p_tr, train_pool), (p_te, indomain_test)):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for r in data:
                if "combined_text" not in r:
                    continue
                toks, labs = base.parse_combined_text(r["combined_text"])
                if not toks:
                    continue
                bucket.append({"tokens": toks, "labels": labs,
                               "domain": d, "split": "train" if bucket is train_pool else "test"})

    for d in test_domains:
        for split in ("train", "test"):
            path = Path(data_dir) / f"{d}_docs_{split}.json"
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for r in data:
                if "combined_text" not in r:
                    continue
                toks, labs = base.parse_combined_text(r["combined_text"])
                if not toks:
                    continue
                heldout_test.append({"tokens": toks, "labels": labs,
                                     "domain": d, "split": split})

    return train_pool, indomain_test, heldout_test


def stratified_train_dev(train_pool, dev_ratio, seed):
    rng = random.Random(seed)
    by_dom = defaultdict(list)
    for it in train_pool:
        by_dom[it["domain"]].append(it)
    train, dev = [], []
    for d, lst in by_dom.items():
        lst = lst[:]
        rng.shuffle(lst)
        n_dev = max(1, int(len(lst) * dev_ratio))
        dev.extend(lst[:n_dev])
        train.extend(lst[n_dev:])
    rng.shuffle(train)
    return train, dev


def main():
    args = parse_args()

    train_domains = [d.strip() for d in args.train_domains.split(",") if d.strip()]
    all_doms = base.Cfg.domains
    for d in train_domains:
        if d not in all_doms:
            raise SystemExit(f"Unknown train domain: {d}. Valid: {all_doms}")

    if args.test_domains.strip():
        test_domains = [d.strip() for d in args.test_domains.split(",") if d.strip()]
        for d in test_domains:
            if d not in all_doms:
                raise SystemExit(f"Unknown test domain: {d}. Valid: {all_doms}")
    else:
        test_domains = [d for d in all_doms if d not in train_domains]

    print(f"Train domains ({len(train_domains)}): {train_domains}")
    print(f"Test  domains ({len(test_domains)}): {test_domains}")

    # patch config
    cfg = base.Cfg()
    cfg.data_dir = args.data_dir
    cfg.output_dir = args.output_dir
    cfg.domains = train_domains + test_domains
    cfg.epochs = args.epochs
    cfg.patience = args.patience
    cfg.batch_size = args.batch_size
    cfg.seed = args.seed
    cfg.dev_ratio = args.dev_ratio
    cfg.use_domain_token = not args.no_domain_token
    cfg.run_dapt = args.run_dapt

    os.makedirs(cfg.output_dir, exist_ok=True)
    base.set_seed(cfg.seed)

    # data
    train_pool, indomain_test, heldout_test = load_domain_split(
        cfg.data_dir, train_domains, test_domains)
    train_items, dev_items = stratified_train_dev(train_pool, cfg.dev_ratio, cfg.seed)
    print(f"Train={len(train_items)} Dev={len(dev_items)} "
          f"InDomainTest={len(indomain_test)} HeldOutTest={len(heldout_test)}")

    # labels
    unique = ["O"]
    for b in cfg.base_labels:
        unique.append(f"B-{b}")
        unique.append(f"I-{b}")
    tag2id = {t: i for i, t in enumerate(unique)}
    id2tag = {i: t for t, i in tag2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if cfg.use_domain_token:
        # add tokens for ALL domains (train + test) so no crash at inference
        dom_tokens = [base.domain_token(d) for d in cfg.domains]
        added = tokenizer.add_tokens(dom_tokens, special_tokens=True)
        print(f"Added {added} domain tokens; tokenizer size = {len(tokenizer)}")
    else:
        print(f"Domain tokens disabled. Tokenizer size = {len(tokenizer)}")

    # DAPT (only on train+dev of source domains)
    backbone_path = cfg.model_name
    if cfg.run_dapt:
        dapt_texts = []
        for it in train_items + dev_items:
            toks = ([base.domain_token(it["domain"])] if cfg.use_domain_token else []) + it["tokens"]
            dapt_texts.append(" ".join(toks))
        backbone_path = base.run_dapt(tokenizer, dapt_texts, cfg)

    # datasets
    train_ds = base.NERDataset(train_items, tokenizer, tag2id, cfg.max_len, cfg.stride,
                               with_domain=cfg.use_domain_token)
    dev_ds = base.NERDataset(dev_items, tokenizer, tag2id, cfg.max_len, cfg.stride,
                             with_domain=cfg.use_domain_token)
    indom_ds = base.NERDataset(indomain_test, tokenizer, tag2id, cfg.max_len, cfg.stride,
                               with_domain=cfg.use_domain_token) if indomain_test else None
    held_ds = base.NERDataset(heldout_test, tokenizer, tag2id, cfg.max_len, cfg.stride,
                              with_domain=cfg.use_domain_token) if heldout_test else None

    coll = base.Collator(tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=coll)
    dev_loader = DataLoader(dev_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=coll)
    indom_loader = DataLoader(indom_ds, batch_size=cfg.batch_size, shuffle=False,
                              collate_fn=coll) if indom_ds else None
    held_loader = DataLoader(held_ds, batch_size=cfg.batch_size, shuffle=False,
                             collate_fn=coll) if held_ds else None

    # model
    model = base.DebertaCRF(backbone_path, num_labels=len(tag2id),
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
    model.focal = base.FocalLoss(alpha=alpha, gamma=cfg.focal_gamma).to(cfg.device)

    groups = base.build_param_groups(model, cfg)
    optimizer = AdamW(groups)
    steps_per_epoch = max(1, len(train_loader) // cfg.grad_accum)
    total_steps = steps_per_epoch * cfg.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(cfg.warmup_ratio * total_steps), total_steps)
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
        res = base.evaluate(model, dev_loader, id2tag, cfg.device,
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

    # in-domain test (source domains, held-out test files)
    if indom_loader is not None:
        print("\n=== IN-DOMAIN TEST (source domains, official test files) ===")
        res_in = base.evaluate(model, indom_loader, id2tag, cfg.device)
        print(f"IN-DOMAIN  micro={res_in['micro_f1']:.4f}  macro={res_in['macro_f1']:.4f}")
        with open(os.path.join(cfg.output_dir, "indomain_summary.txt"), "w") as f:
            f.write(f"micro_f1: {res_in['micro_f1']:.4f}\nmacro_f1: {res_in['macro_f1']:.4f}\n")
        base.save_all_reports(res_in["all_true"], res_in["all_pred"],
                              res_in["flat_true"], res_in["flat_pred"],
                              os.path.join(cfg.output_dir, "indomain"),
                              prefix="indomain_overall_")
        for d in train_domains:
            t = res_in["by_domain_true"].get(d, [])
            p = res_in["by_domain_pred"].get(d, [])
            ft = res_in["by_domain_flat_true"].get(d, [])
            fp = res_in["by_domain_flat_pred"].get(d, [])
            if not t:
                continue
            base.save_all_reports(t, p, ft, fp,
                                  os.path.join(cfg.output_dir, "indomain", f"domain_{d}"),
                                  prefix="indomain_")

    # held-out cross-domain test (per-domain reports)
    if held_loader is not None:
        print("\n=== CROSS-DOMAIN TEST (held-out target domains) ===")
        res_out = base.evaluate(model, held_loader, id2tag, cfg.device)
        print(f"CROSS-DOMAIN  micro={res_out['micro_f1']:.4f}  macro={res_out['macro_f1']:.4f}")
        with open(os.path.join(cfg.output_dir, "crossdomain_summary.txt"), "w") as f:
            f.write(f"micro_f1: {res_out['micro_f1']:.4f}\nmacro_f1: {res_out['macro_f1']:.4f}\n")
            f.write("\nPer-domain micro / macro F1:\n")
            for d in test_domains:
                t = res_out["by_domain_true"].get(d, [])
                p = res_out["by_domain_pred"].get(d, [])
                if not t:
                    f.write(f"  {d}: (no samples)\n"); continue
                from seqeval.metrics import f1_score as sf1
                mi = sf1(t, p, average="micro", zero_division=0)
                ma = sf1(t, p, average="macro", zero_division=0)
                f.write(f"  {d}: micro={mi:.4f}  macro={ma:.4f}\n")
                print(f"  {d}: micro={mi:.4f}  macro={ma:.4f}")

        base.save_all_reports(res_out["all_true"], res_out["all_pred"],
                              res_out["flat_true"], res_out["flat_pred"],
                              os.path.join(cfg.output_dir, "crossdomain"),
                              prefix="cross_overall_")
        for d in test_domains:
            t = res_out["by_domain_true"].get(d, [])
            p = res_out["by_domain_pred"].get(d, [])
            ft = res_out["by_domain_flat_true"].get(d, [])
            fp = res_out["by_domain_flat_pred"].get(d, [])
            if not t:
                continue
            base.save_all_reports(t, p, ft, fp,
                                  os.path.join(cfg.output_dir, "crossdomain", f"domain_{d}"),
                                  prefix="cross_")

    base.plot_curves(history, cfg.output_dir)
    print(f"\nSaved everything to {cfg.output_dir}")


if __name__ == "__main__":
    main()
