"""
Train relation-conditioned span tagging for DrugProt:
  (text, relation) -> BIO tags over sentence tokens

Expected files in --data_dir:
  - span_train.jsonl
  - span_valid.jsonl
  - span_test.jsonl

Optional:
  - --ood_file path/to/span_ood_test.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def upload_folder_to_hub(
    folder_path: Path,
    repo_id: str,
    private: bool = False,
    commit_message: str = "Upload best model",
) -> str:
    from huggingface_hub import HfApi, create_repo

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True, token=token)
    api = HfApi(token=token)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(folder_path),
        path_in_repo=".",
        commit_message=commit_message,
    )
    return f"https://huggingface.co/{repo_id}"


class SpanDataset(torch.utils.data.Dataset):
    def __init__(self, rows: List[dict], tokenizer, label2id: Dict[str, int], max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        rel_tokens = r["relation"].replace("-", " ").split()
        sent_tokens = r["tokens"]
        sent_labels = [self.label2id[t] for t in r["bio_tags"]]

        enc = self.tokenizer(
            rel_tokens,
            sent_tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
        )

        word_ids = enc.word_ids()
        seq_ids = enc.sequence_ids()
        labels = []
        prev_word_idx = None
        for pos, word_idx in enumerate(word_ids):
            sid = seq_ids[pos]
            if sid != 1 or word_idx is None:
                labels.append(-100)
            else:
                # Label only the first subtoken of each word.
                if word_idx != prev_word_idx:
                    labels.append(sent_labels[word_idx])
                else:
                    labels.append(-100)
            prev_word_idx = word_idx

        enc["labels"] = labels
        return enc


def token_metrics(eval_pred, id2label: Dict[int, str]) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    y_true_all: List[int] = []
    y_pred_all: List[int] = []
    for p_row, l_row in zip(preds, labels):
        for p, l in zip(p_row, l_row):
            if l == -100:
                continue
            y_true_all.append(int(l))
            y_pred_all.append(int(p))

    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true_all, y_pred_all, average="micro", zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_all, y_pred_all, average="macro", zero_division=0
    )

    o_id = next((k for k, v in id2label.items() if v == "O"), None)
    out = {
        "token_micro_f1": float(f1_micro),
        "token_macro_f1": float(f1_macro),
        "token_micro_precision": float(p_micro),
        "token_micro_recall": float(r_micro),
        "token_macro_precision": float(p_macro),
        "token_macro_recall": float(r_macro),
    }
    if o_id is not None:
        true_pos = [t for t in y_true_all if t != o_id]
        pred_pos = [p for p, t in zip(y_pred_all, y_true_all) if t != o_id]
        if len(true_pos) > 0:
            p_pos, r_pos, f1_pos, _ = precision_recall_fscore_support(
                true_pos, pred_pos, average="micro", zero_division=0
            )
            out["token_positive_micro_f1"] = float(f1_pos)
            out["token_positive_micro_precision"] = float(p_pos)
            out["token_positive_micro_recall"] = float(r_pos)
        else:
            out["token_positive_micro_f1"] = 0.0
            out["token_positive_micro_precision"] = 0.0
            out["token_positive_micro_recall"] = 0.0
    return out


def decode_bio_spans(tags: List[str], role: str) -> List[Tuple[int, int]]:
    """Decode BIO spans for a role (returns [start, end) token offsets)."""
    b_tag = f"B-{role}"
    i_tag = f"I-{role}"
    spans: List[Tuple[int, int]] = []
    start = None

    for i, t in enumerate(tags):
        if t == b_tag:
            if start is not None:
                spans.append((start, i))
            start = i
        elif t == i_tag:
            if start is None:
                # Robustness: treat orphan I as a B.
                start = i
        else:
            if start is not None:
                spans.append((start, i))
                start = None
    if start is not None:
        spans.append((start, len(tags)))
    return spans


class SpanSampleEvalCallback(TrainerCallback):
    """Print prediction on one validation example every k steps."""

    def __init__(
        self,
        valid_rows: List[dict],
        tokenizer,
        max_length: int,
        every_steps: int,
        sample_index: int = 0,
    ):
        self.valid_rows = valid_rows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.every_steps = every_steps
        self.sample_index = sample_index

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None or self.every_steps <= 0 or not self.valid_rows:
            return control
        if state.global_step <= 0 or (state.global_step % self.every_steps != 0):
            return control

        row = self.valid_rows[self.sample_index % len(self.valid_rows)]
        rel_tokens = row["relation"].replace("-", " ").split()
        sent_tokens = row["tokens"]

        enc = self.tokenizer(
            [rel_tokens],
            [sent_tokens],
            is_split_into_words=True,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        device = next(model.parameters()).device
        enc_dev = {k: v.to(device) for k, v in enc.items()}

        was_training = model.training
        model.eval()
        with torch.no_grad():
            logits = model(**enc_dev).logits
            pred_ids = torch.argmax(logits, dim=-1).cpu().tolist()[0]
        if was_training:
            model.train()

        id2label = {
            int(k): v for k, v in model.config.id2label.items()
        } if isinstance(model.config.id2label, dict) else model.config.id2label

        word_ids = enc.word_ids(batch_index=0)
        seq_ids = enc.sequence_ids(0)
        sent_tag_ids = [-100] * len(sent_tokens)
        prev_w = None
        for pos, w in enumerate(word_ids):
            if seq_ids[pos] != 1 or w is None:
                continue
            if w != prev_w and sent_tag_ids[w] == -100:
                sent_tag_ids[w] = pred_ids[pos]
            prev_w = w

        pred_tags = [id2label[t] if t != -100 else "O" for t in sent_tag_ids]
        pred_chem = decode_bio_spans(pred_tags, "CHEM")
        pred_gene = decode_bio_spans(pred_tags, "GENE")
        gold_chem = decode_bio_spans(row["bio_tags"], "CHEM")
        gold_gene = decode_bio_spans(row["bio_tags"], "GENE")

        print(
            f"[sample-eval step {state.global_step}] rel={row['relation']} "
            f"gold_chem={gold_chem} pred_chem={pred_chem} "
            f"gold_gene={gold_gene} pred_gene={pred_gene}"
        )
        return control


def _span_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def _match_overlap(pred_spans: List[Tuple[int, int]], gold_spans: List[Tuple[int, int]]) -> int:
    """Maximum one-to-one matches where spans overlap at least one token."""
    matched_gold = set()
    tp = 0
    for p in pred_spans:
        best_j = None
        best_ov = 0
        for j, g in enumerate(gold_spans):
            if j in matched_gold:
                continue
            ov = _span_overlap(p, g)
            if ov > best_ov:
                best_ov = ov
                best_j = j
        if best_j is not None and best_ov > 0:
            matched_gold.add(best_j)
            tp += 1
    return tp


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def span_argument_metrics(eval_pred, id2label: Dict[int, str]) -> Dict[str, float]:
    """
    Argument-level span metrics from BIO sequences:
    - exact match (strict)
    - overlap match (>=1 token overlap)
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    exact_tp = exact_fp = exact_fn = 0
    overlap_tp = overlap_fp = overlap_fn = 0

    for p_row, l_row in zip(preds, labels):
        valid = [i for i, l in enumerate(l_row) if l != -100]
        if not valid:
            continue

        pred_tags = [id2label[int(p_row[i])] for i in valid]
        gold_tags = [id2label[int(l_row[i])] for i in valid]

        for role in ("CHEM", "GENE"):
            pred_spans = decode_bio_spans(pred_tags, role)
            gold_spans = decode_bio_spans(gold_tags, role)

            # Strict exact matching.
            pred_set = set(pred_spans)
            gold_set = set(gold_spans)
            tp_e = len(pred_set & gold_set)
            fp_e = len(pred_set - gold_set)
            fn_e = len(gold_set - pred_set)
            exact_tp += tp_e
            exact_fp += fp_e
            exact_fn += fn_e

            # Overlap-based one-to-one matching.
            tp_o = _match_overlap(pred_spans, gold_spans)
            fp_o = len(pred_spans) - tp_o
            fn_o = len(gold_spans) - tp_o
            overlap_tp += tp_o
            overlap_fp += fp_o
            overlap_fn += fn_o

    ep, er, ef1 = _prf(exact_tp, exact_fp, exact_fn)
    op, or_, of1 = _prf(overlap_tp, overlap_fp, overlap_fn)
    return {
        "span_exact_precision": ep,
        "span_exact_recall": er,
        "span_exact_f1": ef1,
        "span_overlap_precision": op,
        "span_overlap_recall": or_,
        "span_overlap_f1": of1,
    }


def eval_split(trainer: Trainer, ds: SpanDataset, id2label: Dict[int, str], split_name: str) -> dict:
    pred_out = trainer.predict(ds)
    logits = pred_out.predictions
    labels = pred_out.label_ids
    token_scores = token_metrics((logits, labels), id2label)
    span_scores = span_argument_metrics((logits, labels), id2label)
    out = {f"{split_name}_{k}": float(v) for k, v in pred_out.metrics.items() if isinstance(v, (int, float))}
    out.update({f"{split_name}_{k}": v for k, v in token_scores.items()})
    out.update({f"{split_name}_{k}": v for k, v in span_scores.items()})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="drugprot_dual")
    parser.add_argument("--output_dir", type=str, default="outputs/span_re")
    parser.add_argument("--model_name", type=str, default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--num_train_epochs", type=float, default=5.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_strategy", type=str, default="epoch")
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--ood_file", type=str, default="")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default="")
    parser.add_argument("--hub_private", action="store_true")
    parser.add_argument("--hub_commit_message", type=str, default="Upload best span RE model")
    parser.add_argument(
        "--sample_eval_steps",
        type=int,
        default=200,
        help="Run one-example validation inference every k training steps (<=0 disables).",
    )
    parser.add_argument(
        "--sample_eval_index",
        type=int,
        default=0,
        help="Validation example index used for periodic sample inference.",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    train_rows = read_jsonl(data_dir / "span_train.jsonl")
    valid_rows = read_jsonl(data_dir / "span_valid.jsonl")
    test_rows = read_jsonl(data_dir / "span_test.jsonl")
    n_valid_raw = len(valid_rows)
    n_test_raw = len(test_rows)

    tag_list = sorted({t for r in train_rows for t in r["bio_tags"]})
    # Keep O first for readability.
    if "O" in tag_list:
        tag_list = ["O"] + [t for t in tag_list if t != "O"]
    label2id = {t: i for i, t in enumerate(tag_list)}
    id2label = {i: t for t, i in label2id.items()}

    # Prevent crashes if unexpected tags appear outside train split.
    valid_rows = [r for r in valid_rows if all(t in label2id for t in r["bio_tags"])]
    test_rows = [r for r in test_rows if all(t in label2id for t in r["bio_tags"])]
    if len(valid_rows) != n_valid_raw or len(test_rows) != n_test_raw:
        print(
            "Filtered rows with unseen BIO tags: "
            f"valid dropped={n_valid_raw - len(valid_rows)}, "
            f"test dropped={n_test_raw - len(test_rows)}"
        )

    # Select best checkpoint by argument-level quality, not token-only quality.
    best_metric_name = "eval_span_overlap_f1"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(tag_list),
        label2id=label2id,
        id2label=id2label,
    )

    train_ds = SpanDataset(train_rows, tokenizer, label2id, args.max_length)
    valid_ds = SpanDataset(valid_rows, tokenizer, label2id, args.max_length)
    test_ds = SpanDataset(test_rows, tokenizer, label2id, args.max_length)
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model=best_metric_name,
        greater_is_better=True,
        save_total_limit=2,
        fp16=args.fp16,
        bf16=args.bf16,
        seed=args.seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=lambda p: {
            **token_metrics(p, id2label),
            **span_argument_metrics(p, id2label),
        },
        callbacks=[
            SpanSampleEvalCallback(
                valid_rows=valid_rows,
                tokenizer=tokenizer,
                max_length=args.max_length,
                every_steps=args.sample_eval_steps,
                sample_index=args.sample_eval_index,
            )
        ],
    )

    trainer.train()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_payload = {}
    eval_payload.update(eval_split(trainer, valid_ds, id2label, "valid"))
    eval_payload.update(eval_split(trainer, test_ds, id2label, "test"))

    if args.ood_file:
        ood_rows = read_jsonl(Path(args.ood_file))
        n_ood_raw = len(ood_rows)
        # Keep only known tags.
        ood_rows = [r for r in ood_rows if all(t in label2id for t in r["bio_tags"])]
        if len(ood_rows) != n_ood_raw:
            print(
                "Filtered OOD rows with unseen BIO tags: "
                f"dropped={n_ood_raw - len(ood_rows)}"
            )
        ood_ds = SpanDataset(ood_rows, tokenizer, label2id, args.max_length)
        eval_payload.update(eval_split(trainer, ood_ds, id2label, "ood"))

    metadata = {
        "model_name": args.model_name,
        "tag_list": tag_list,
        "n_train": len(train_rows),
        "n_valid": len(valid_rows),
        "n_test": len(test_rows),
        "args": vars(args),
    }
    write_json(out_dir / "metadata.json", metadata)
    write_json(out_dir / "eval_metrics.json", eval_payload)
    best_model_dir = out_dir / "best_model"
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))

    if args.push_to_hub:
        if not args.hub_model_id:
            raise ValueError("--hub_model_id is required when --push_to_hub is set.")
        hub_url = upload_folder_to_hub(
            folder_path=best_model_dir,
            repo_id=args.hub_model_id,
            private=args.hub_private,
            commit_message=args.hub_commit_message,
        )
        print(f"Uploaded model to: {hub_url}")

    print("Training complete.")
    print(f"Saved results to: {out_dir}")


if __name__ == "__main__":
    main()
