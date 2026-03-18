"""
Train classic DrugProt RE:
  (text, entity pair) -> relation label

Expected files in --data_dir:
  - classic_train.jsonl
  - classic_valid.jsonl
  - classic_test.jsonl

Optional:
  - --ood_file path/to/classic_ood_test.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
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


def add_entity_markers(text: str, e1s: int, e1e: int, e2s: int, e2e: int) -> str:
    # Ensure consistent left-to-right insertion order.
    spans = [
        ("e1", e1s, e1e, "[E1]", "[/E1]"),
        ("e2", e2s, e2e, "[E2]", "[/E2]"),
    ]
    spans.sort(key=lambda x: x[1], reverse=True)
    out = text
    for _name, s, e, open_tag, close_tag in spans:
        s = max(0, min(s, len(out)))
        e = max(0, min(e, len(out)))
        if s >= e:
            continue
        out = out[:e] + f" {close_tag} " + out[e:]
        out = out[:s] + f" {open_tag} " + out[s:]
    return " ".join(out.split())


@dataclass
class ClassicExample:
    text: str
    label: int


class ClassicDataset(torch.utils.data.Dataset):
    def __init__(self, rows: List[dict], tokenizer, label2id: Dict[str, int], max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        marked = add_entity_markers(
            r["text"], r["e1_start"], r["e1_end"], r["e2_start"], r["e2_end"]
        )
        enc = self.tokenizer(
            marked,
            truncation=True,
            max_length=self.max_length,
        )
        enc["labels"] = self.label2id[r["label"]]
        return enc


def compute_metrics_builder(id2label: Dict[int, str]):
    no_rel_id = next((i for i, l in id2label.items() if l == "no_relation"), None)

    def _compute(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
            labels, preds, average="micro", zero_division=0
        )
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            labels, preds, average="macro", zero_division=0
        )
        out = {
            "accuracy": float(acc),
            "micro_f1": float(f1_micro),
            "macro_f1": float(f1_macro),
            "micro_precision": float(p_micro),
            "micro_recall": float(r_micro),
            "macro_precision": float(p_macro),
            "macro_recall": float(r_macro),
        }
        if no_rel_id is not None:
            keep = labels != no_rel_id
            if keep.any():
                p_pos, r_pos, f1_pos, _ = precision_recall_fscore_support(
                    labels[keep], preds[keep], average="micro", zero_division=0
                )
                out["positive_micro_f1"] = float(f1_pos)
                out["positive_micro_precision"] = float(p_pos)
                out["positive_micro_recall"] = float(r_pos)
        return out

    return _compute


def eval_split(trainer: Trainer, ds, id2label: Dict[int, str], split_name: str) -> dict:
    pred_out = trainer.predict(ds)
    logits = pred_out.predictions
    labels = pred_out.label_ids
    preds = np.argmax(logits, axis=-1)
    report = classification_report(
        labels,
        preds,
        labels=sorted(id2label.keys()),
        target_names=[id2label[i] for i in sorted(id2label.keys())],
        zero_division=0,
        output_dict=True,
    )
    metrics = {f"{split_name}_{k}": float(v) for k, v in pred_out.metrics.items() if isinstance(v, (int, float))}
    metrics[f"{split_name}_classification_report"] = report
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="drugprot_dual")
    parser.add_argument("--output_dir", type=str, default="outputs/classic_re")
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
    args = parser.parse_args()

    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    train_rows = read_jsonl(data_dir / "classic_train.jsonl")
    valid_rows = read_jsonl(data_dir / "classic_valid.jsonl")
    test_rows = read_jsonl(data_dir / "classic_test.jsonl")

    labels = sorted({r["label"] for r in (train_rows + valid_rows + test_rows)})
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]})

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
    )
    model.resize_token_embeddings(len(tokenizer))

    train_ds = ClassicDataset(train_rows, tokenizer, label2id, args.max_length)
    valid_ds = ClassicDataset(valid_rows, tokenizer, label2id, args.max_length)
    test_ds = ClassicDataset(test_rows, tokenizer, label2id, args.max_length)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

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
        metric_for_best_model="eval_macro_f1",
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
        compute_metrics=compute_metrics_builder(id2label),
    )

    trainer.train()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_payload = {}
    eval_payload.update(eval_split(trainer, valid_ds, id2label, "valid"))
    eval_payload.update(eval_split(trainer, test_ds, id2label, "test"))

    if args.ood_file:
        ood_rows = read_jsonl(Path(args.ood_file))
        # Keep only labels seen during train.
        ood_rows = [r for r in ood_rows if r["label"] in label2id]
        ood_ds = ClassicDataset(ood_rows, tokenizer, label2id, args.max_length)
        eval_payload.update(eval_split(trainer, ood_ds, id2label, "ood"))

    metadata = {
        "model_name": args.model_name,
        "label_list": labels,
        "n_train": len(train_rows),
        "n_valid": len(valid_rows),
        "n_test": len(test_rows),
        "args": vars(args),
    }
    write_json(out_dir / "metadata.json", metadata)
    write_json(out_dir / "eval_metrics.json", eval_payload)
    trainer.save_model(str(out_dir / "best_model"))
    tokenizer.save_pretrained(str(out_dir / "best_model"))

    print("Training complete.")
    print(f"Saved results to: {out_dir}")


if __name__ == "__main__":
    main()
