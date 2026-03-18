"""
Evaluate saved prediction JSONL files against gold JSONL files.

Supports:
  - classic: label classification metrics
  - span: token + span + null metrics

Examples:
  python training/eval_predictions.py \
    --task classic \
    --gold_file drugprot_dual/classic_test.jsonl \
    --pred_file outputs/classic_test_preds.jsonl \
    --out_file outputs/classic_test_eval.json

  python training/eval_predictions.py \
    --task span \
    --gold_file drugprot_dual/span_test.jsonl \
    --pred_file outputs/span_test_preds.jsonl \
    --out_file outputs/span_test_eval.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support


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


def key_for_row(r: dict, fallback_idx: int) -> str:
    if "id" in r:
        return f"id::{r['id']}"
    if "sentence_id" in r and "relation" in r:
        return f"sentrel::{r['sentence_id']}::{r['relation']}"
    if "sentence_id" in r and "pair_id" in r:
        return f"sentpair::{r['sentence_id']}::{r['pair_id']}"
    return f"idx::{fallback_idx}"


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


def _span_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def _match_overlap(pred_spans: List[Tuple[int, int]], gold_spans: List[Tuple[int, int]]) -> int:
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


def eval_classic(gold_rows: List[dict], pred_rows: List[dict]) -> dict:
    pred_by_key = {key_for_row(r, i): r for i, r in enumerate(pred_rows)}

    y_true: List[str] = []
    y_pred: List[str] = []
    missing = 0

    for i, g in enumerate(gold_rows):
        k = key_for_row(g, i)
        p = pred_by_key.get(k)
        if p is None:
            missing += 1
            continue
        if "pred_label" not in p:
            missing += 1
            continue
        y_true.append(g["label"])
        y_pred.append(p["pred_label"])

    labels = sorted(set(y_true) | set(y_pred))
    if not y_true:
        return {"error": "No aligned rows for evaluation.", "missing_predictions": missing}

    acc = accuracy_score(y_true, y_pred)
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)

    no_rel = "no_relation"
    out = {
        "n_gold": len(gold_rows),
        "n_scored": len(y_true),
        "missing_predictions": missing,
        "accuracy": float(acc),
        "micro_f1": float(f1_micro),
        "macro_f1": float(f1_macro),
        "micro_precision": float(p_micro),
        "micro_recall": float(r_micro),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "classification_report": report,
    }
    if no_rel in labels:
        pos_idx = [i for i, t in enumerate(y_true) if t != no_rel]
        if pos_idx:
            yt = [y_true[i] for i in pos_idx]
            yp = [y_pred[i] for i in pos_idx]
            p_pos, r_pos, f1_pos, _ = precision_recall_fscore_support(
                yt, yp, average="micro", zero_division=0
            )
            out["positive_micro_f1"] = float(f1_pos)
            out["positive_micro_precision"] = float(p_pos)
            out["positive_micro_recall"] = float(r_pos)
    return out


def eval_span(gold_rows: List[dict], pred_rows: List[dict]) -> dict:
    pred_by_key = {key_for_row(r, i): r for i, r in enumerate(pred_rows)}

    token_true: List[str] = []
    token_pred: List[str] = []
    missing = 0

    exact_tp = exact_fp = exact_fn = 0
    overlap_tp = overlap_fp = overlap_fn = 0
    null_true: List[int] = []
    null_pred: List[int] = []

    for i, g in enumerate(gold_rows):
        k = key_for_row(g, i)
        p = pred_by_key.get(k)
        if p is None or "pred_tags" not in p:
            missing += 1
            continue

        g_tags = g["bio_tags"]
        p_tags = p["pred_tags"]
        if len(g_tags) != len(p_tags):
            # Align by truncation to avoid hard failure.
            n = min(len(g_tags), len(p_tags))
            g_tags = g_tags[:n]
            p_tags = p_tags[:n]

        token_true.extend(g_tags)
        token_pred.extend(p_tags)

        for role in ("CHEM", "GENE"):
            g_sp = decode_bio_spans(g_tags, role)
            p_sp = decode_bio_spans(p_tags, role)

            g_set = set(g_sp)
            p_set = set(p_sp)
            exact_tp += len(g_set & p_set)
            exact_fp += len(p_set - g_set)
            exact_fn += len(g_set - p_set)

            tp_o = _match_overlap(p_sp, g_sp)
            overlap_tp += tp_o
            overlap_fp += len(p_sp) - tp_o
            overlap_fn += len(g_sp) - tp_o

        null_true.append(1 if g.get("is_null", False) else 0)
        null_pred.append(1 if p.get("pred_is_null", False) else 0)

    if not token_true:
        return {"error": "No aligned rows for evaluation.", "missing_predictions": missing}

    labels = sorted(set(token_true) | set(token_pred))
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        token_true, token_pred, average="micro", zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        token_true, token_pred, average="macro", zero_division=0
    )

    o_label = "O"
    out = {
        "n_gold": len(gold_rows),
        "n_scored": len(null_true),
        "missing_predictions": missing,
        "token_micro_f1": float(f1_micro),
        "token_macro_f1": float(f1_macro),
        "token_micro_precision": float(p_micro),
        "token_micro_recall": float(r_micro),
        "token_macro_precision": float(p_macro),
        "token_macro_recall": float(r_macro),
    }
    if o_label in labels:
        keep = [i for i, t in enumerate(token_true) if t != o_label]
        if keep:
            yt = [token_true[i] for i in keep]
            yp = [token_pred[i] for i in keep]
            p_pos, r_pos, f1_pos, _ = precision_recall_fscore_support(
                yt, yp, average="micro", zero_division=0
            )
            out["token_positive_micro_f1"] = float(f1_pos)
            out["token_positive_micro_precision"] = float(p_pos)
            out["token_positive_micro_recall"] = float(r_pos)

    ep, er, ef1 = _prf(exact_tp, exact_fp, exact_fn)
    op, or_, of1 = _prf(overlap_tp, overlap_fp, overlap_fn)
    out.update(
        {
            "span_exact_precision": ep,
            "span_exact_recall": er,
            "span_exact_f1": ef1,
            "span_overlap_precision": op,
            "span_overlap_recall": or_,
            "span_overlap_f1": of1,
        }
    )

    p_n, r_n, f1_n, _ = precision_recall_fscore_support(
        null_true, null_pred, average="binary", zero_division=0
    )
    out["null_precision"] = float(p_n)
    out["null_recall"] = float(r_n)
    out["null_f1"] = float(f1_n)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["classic", "span"])
    parser.add_argument("--gold_file", type=str, required=True)
    parser.add_argument("--pred_file", type=str, required=True)
    parser.add_argument("--out_file", type=str, required=True)
    args = parser.parse_args()

    gold_rows = read_jsonl(Path(args.gold_file))
    pred_rows = read_jsonl(Path(args.pred_file))

    if args.task == "classic":
        metrics = eval_classic(gold_rows, pred_rows)
    else:
        metrics = eval_span(gold_rows, pred_rows)

    write_json(Path(args.out_file), metrics)
    print(f"Wrote eval metrics -> {args.out_file}")
    if "error" not in metrics:
        print(json.dumps({k: v for k, v in metrics.items() if isinstance(v, (int, float))}, indent=2))
    else:
        print(metrics["error"])


if __name__ == "__main__":
    main()
