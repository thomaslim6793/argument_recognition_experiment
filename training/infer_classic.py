"""
Batch inference for classic pair-classification RE model.

Input JSONL rows should include:
  - text
  - e1_start, e1_end
  - e2_start, e2_end

Optional pass-through fields (preserved in output if present):
  - id, sentence_id, pair_id, e1_text, e2_text

Output JSONL rows include:
  - pred_label
  - pred_score (max softmax prob)
  - pred_probs (label->prob map)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


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


def add_entity_markers(text: str, e1s: int, e1e: int, e2s: int, e2e: int) -> str:
    spans = [
        ("E1", e1s, e1e),
        ("E2", e2s, e2e),
    ]
    spans.sort(key=lambda x: x[1], reverse=True)
    out = text
    for name, s, e in spans:
        s = max(0, min(s, len(out)))
        e = max(0, min(e, len(out)))
        if s >= e:
            continue
        out = out[:e] + f" [/{name}] " + out[e:]
        out = out[:s] + f" [{name}] " + out[s:]
    return " ".join(out.split())


def id2label_from_config(model) -> Dict[int, str]:
    cfg_map = model.config.id2label
    # Transformers can store keys as str or int.
    out: Dict[int, str] = {}
    for k, v in cfg_map.items():
        out[int(k)] = v
    return out


class ClassicPredictor:
    """Simple callable predictor for single-example classic RE inference."""

    def __init__(self, model_dir: str, max_length: int = 256, device: str = ""):
        self.model_dir = model_dir
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
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
        self._label_ids = sorted(self._id2label.keys())
        self._labels = [self._id2label[i] for i in self._label_ids]

    def predict(
        self, text: str, e1_start: int, e1_end: int, e2_start: int, e2_end: int
    ) -> dict:
        marked = add_entity_markers(text, e1_start, e1_end, e2_start, e2_end)
        enc = self.tokenizer(
            marked, return_tensors="pt", truncation=True, max_length=self.max_length
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
        best_idx = int(max(range(len(probs)), key=lambda x: probs[x]))
        pred_label = self._id2label[best_idx]
        return {
            "pred_label": pred_label,
            "pred_score": float(probs[best_idx]),
            "pred_probs": {self._labels[j]: float(probs[j]) for j in range(len(self._labels))},
        }


def predict_classic(
    model_dir: str,
    text: str,
    e1_start: int,
    e1_end: int,
    e2_start: int,
    e2_end: int,
    max_length: int = 256,
    device: str = "",
) -> dict:
    """Convenience function for one-off single-example prediction."""
    predictor = ClassicPredictor(model_dir=model_dir, max_length=max_length, device=device)
    return predictor.predict(text, e1_start, e1_end, e2_start, e2_end)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Path to best_model dir")
    parser.add_argument("--input_file", type=str, required=True, help="JSONL input file")
    parser.add_argument("--output_file", type=str, required=True, help="JSONL prediction output")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="", help="e.g., cuda, mps, cpu")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    inp = Path(args.input_file)
    out = Path(args.output_file)

    predictor = ClassicPredictor(
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
                e1_start=r["e1_start"],
                e1_end=r["e1_end"],
                e2_start=r["e2_start"],
                e2_end=r["e2_end"],
            )
            out_row = dict(pred)
            for k in ("id", "sentence_id", "pair_id", "text", "e1_text", "e2_text"):
                if k in r:
                    out_row[k] = r[k]
            outputs.append(out_row)

    write_jsonl(outputs, out)
    print(f"Wrote {len(outputs)} predictions -> {out}")


if __name__ == "__main__":
    main()
