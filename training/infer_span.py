"""
Batch inference for relation-conditioned span RE model.

Input JSONL rows should include:
  - text
  - relation

Optional:
  - tokens (if omitted, whitespace tokenization is used)
  - id, sentence_id

Output JSONL rows include:
  - pred_tags (token-level BIO tags)
  - pred_chem_spans / pred_gene_spans (token index spans [start,end))
  - pred_is_null (True if no non-O tags)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def id2label_from_config(model) -> Dict[int, str]:
    cfg_map = model.config.id2label
    out: Dict[int, str] = {}
    for k, v in cfg_map.items():
        out[int(k)] = v
    return out


def decode_bio_spans(tags: List[str], role: str) -> List[Tuple[int, int]]:
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
                start = i
        else:
            if start is not None:
                spans.append((start, i))
                start = None
    if start is not None:
        spans.append((start, len(tags)))
    return spans


class SpanPredictor:
    """Simple callable predictor for single-example span RE inference."""

    def __init__(self, model_dir: str, max_length: int = 256, device: str = ""):
        self.model_dir = model_dir
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.model.eval()

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                else "cpu"
            )
        self.model.to(self.device)
        self._id2label = id2label_from_config(self.model)

    def predict(self, text: str, relation: str, tokens: List[str] | None = None) -> dict:
        sent_tokens = tokens if tokens is not None else text.split()
        rel_tokens = relation.replace("-", " ").split()

        enc = self.tokenizer(
            [rel_tokens],
            [sent_tokens],
            is_split_into_words=True,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc_dev = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.model(**enc_dev).logits
            pred_ids = torch.argmax(logits, dim=-1).cpu().tolist()[0]

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

        tags = [self._id2label[t] if t != -100 else "O" for t in sent_tag_ids]
        chem_spans = decode_bio_spans(tags, "CHEM")
        gene_spans = decode_bio_spans(tags, "GENE")
        pred_is_null = not any(t != "O" for t in tags)
        return {
            "relation": relation,
            "tokens": sent_tokens,
            "pred_tags": tags,
            "pred_chem_spans": chem_spans,
            "pred_gene_spans": gene_spans,
            "pred_is_null": pred_is_null,
        }


def predict_span(
    model_dir: str,
    text: str,
    relation: str,
    tokens: List[str] | None = None,
    max_length: int = 256,
    device: str = "",
) -> dict:
    """Convenience function for one-off single-example span prediction."""
    predictor = SpanPredictor(model_dir=model_dir, max_length=max_length, device=device)
    return predictor.predict(text=text, relation=relation, tokens=tokens)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Path to best_model dir")
    parser.add_argument("--input_file", type=str, required=True, help="JSONL input file")
    parser.add_argument("--output_file", type=str, required=True, help="JSONL prediction output")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="", help="e.g., cuda, mps, cpu")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    inp = Path(args.input_file)
    out = Path(args.output_file)

    predictor = SpanPredictor(
        model_dir=str(model_dir),
        max_length=args.max_length,
        device=args.device,
    )

    rows = read_jsonl(inp)

    outputs: List[dict] = []
    for i in range(0, len(rows), args.batch_size):
        batch_rows = rows[i : i + args.batch_size]
        for r in batch_rows:
            pred = predictor.predict(
                text=r["text"],
                relation=r["relation"],
                tokens=r.get("tokens"),
            )
            out_row = dict(pred)
            for k in ("id", "sentence_id", "text"):
                if k in r:
                    out_row[k] = r[k]
            outputs.append(out_row)

    write_jsonl(outputs, out)
    print(f"Wrote {len(outputs)} predictions -> {out}")


if __name__ == "__main__":
    main()
