"""
Build paired DrugProt datasets for two RE formulations:

1) Classic pair classification:
   (text, chem span, gene span) -> label (including no_relation)

2) Relation-conditioned span extraction:
   (text, relation query) -> token-level BIO tags
   where null is represented as all "O".

Notes:
- Uses bigbio/drugprot (kb schema).
- Keeps only CHEMICAL vs GENE pairs.
- Ignores directionality: (chem, gene) and (gene, chem) are treated as the same pair.
- Restricts to within-sentence relations/pairs.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

from datasets import load_dataset


def normalize_type(t: str) -> str:
    t_low = t.lower()
    if "chem" in t_low or "drug" in t_low or "compound" in t_low:
        return "CHEMICAL"
    if "gene" in t_low or "protein" in t_low or "geneorgeneproduct" in t_low:
        return "GENE"
    return t


def is_chem(t: str) -> bool:
    return normalize_type(t) == "CHEMICAL"


def is_gene(t: str) -> bool:
    return normalize_type(t) == "GENE"


def sent_tokenize_offsets(text: str) -> List[Tuple[int, int]]:
    """Simple punctuation-based sentence splitter."""
    spans: List[Tuple[int, int]] = []
    start = 0
    n = len(text)
    for i, ch in enumerate(text):
        if ch in ".!?":
            end = i + 1
            while end < n and text[end].isspace():
                end += 1
            spans.append((start, end))
            start = end
    if start < n:
        spans.append((start, n))
    return [s for s in spans if s[0] < s[1]]


def find_sentence_index(
    sent_spans: List[Tuple[int, int]], start: int, end: int
) -> Optional[int]:
    for idx, (s, e) in enumerate(sent_spans):
        if start >= s and end <= e:
            return idx
    return None


def parse_entity(entity: dict) -> Optional[dict]:
    offsets = entity.get("offsets")
    if not offsets:
        return None

    first = offsets[0]
    if isinstance(first, dict):
        start = first["start"]
        end = first["end"]
    else:
        start, end = first

    ent_type = entity.get("type")
    if isinstance(ent_type, list):
        ent_type = ent_type[0]

    ent_text = entity.get("text")
    if isinstance(ent_text, list):
        ent_text = " ".join(ent_text)

    return {
        "id": entity["id"],
        "text": ent_text,
        "start": start,
        "end": end,
        "type": normalize_type(ent_type),
    }


def parse_relation(rel: dict) -> Tuple[str, str, str]:
    rtype = rel["type"]
    if isinstance(rtype, list):
        rtype = rtype[0]

    if "arg1_id" in rel and "arg2_id" in rel:
        return rel["arg1_id"], rel["arg2_id"], rtype

    args = rel.get("arguments", [])
    if len(args) >= 2:
        a1 = args[0].get("ref_id", args[0].get("arg_id"))
        a2 = args[1].get("ref_id", args[1].get("arg_id"))
        if a1 is not None and a2 is not None:
            return a1, a2, rtype

    raise ValueError(f"Could not parse relation format: {rel}")


def undirected_pair_id(a: str, b: str) -> Tuple[str, str]:
    return tuple(sorted((a, b)))


def collect_relation_vocab(ds_split) -> Set[str]:
    rels: Set[str] = set()
    for doc in ds_split:
        for rel in doc["relations"]:
            rtype = rel.get("type")
            if isinstance(rtype, list):
                if rtype:
                    rtype = rtype[0]
                else:
                    continue
            if rtype is not None:
                rels.add(rtype)
    return rels


def tokenize_with_offsets(text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    tokens: List[str] = []
    offsets: List[Tuple[int, int]] = []
    for match in re.finditer(r"\S+", text):
        tokens.append(match.group(0))
        offsets.append((match.start(), match.end()))
    return tokens, offsets


def token_indices_for_span(
    token_offsets: List[Tuple[int, int]], start: int, end: int
) -> List[int]:
    idxs: List[int] = []
    for i, (ts, te) in enumerate(token_offsets):
        if ts < end and te > start:
            idxs.append(i)
    return idxs


def apply_bio(tags: List[str], token_idxs: Iterable[int], label: str) -> None:
    sorted_idxs = sorted(set(token_idxs))
    if not sorted_idxs:
        return
    prev = None
    for i in sorted_idxs:
        prefix = "B" if prev is None or i != prev + 1 else "I"
        new_tag = f"{prefix}-{label}"
        if tags[i] == "O":
            tags[i] = new_tag
        prev = i


def write_jsonl(rows: List[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def choose_test_sentence_ids(
    classic_rows: List[dict], test_frac: float, seed: int
) -> Set[str]:
    by_sentence: Dict[str, List[dict]] = defaultdict(list)
    for row in classic_rows:
        by_sentence[row["sentence_id"]].append(row)

    pos_ids: List[str] = []
    neg_ids: List[str] = []
    for sid, rows in by_sentence.items():
        has_positive = any(r["label"] != "no_relation" for r in rows)
        if has_positive:
            pos_ids.append(sid)
        else:
            neg_ids.append(sid)

    rng = random.Random(seed)
    rng.shuffle(pos_ids)
    rng.shuffle(neg_ids)

    n_pos_test = int(round(len(pos_ids) * test_frac))
    n_neg_test = int(round(len(neg_ids) * test_frac))
    if len(pos_ids) > 0 and n_pos_test == 0:
        n_pos_test = 1
    if len(neg_ids) > 0 and n_neg_test == 0:
        n_neg_test = 1

    return set(pos_ids[:n_pos_test]) | set(neg_ids[:n_neg_test])


def split_rows_by_sentence_id(
    rows: List[dict], test_sentence_ids: Set[str]
) -> Tuple[List[dict], List[dict]]:
    train_rows: List[dict] = []
    test_rows: List[dict] = []
    for row in rows:
        if row["sentence_id"] in test_sentence_ids:
            test_rows.append(row)
        else:
            train_rows.append(row)
    return train_rows, test_rows


def classic_stats(rows: List[dict]) -> dict:
    total = len(rows)
    n_no_rel = sum(1 for r in rows if r["label"] == "no_relation")
    return {
        "rows": total,
        "no_relation_rows": n_no_rel,
        "positive_rows": total - n_no_rel,
        "no_relation_ratio": (n_no_rel / total) if total else 0.0,
    }


def span_stats(rows: List[dict]) -> dict:
    total = len(rows)
    n_null = sum(1 for r in rows if r.get("is_null", False))
    return {
        "rows": total,
        "null_rows": n_null,
        "nonnull_rows": total - n_null,
        "null_ratio": (n_null / total) if total else 0.0,
    }


def process_split(
    ds_split, split_name: str, relation_vocab: Optional[Set[str]] = None
) -> Tuple[List[dict], List[dict], Set[str]]:
    classic_rows: List[dict] = []
    span_rows: List[dict] = []
    observed_relations: Set[str] = set()

    for doc_idx, doc in enumerate(ds_split):
        doc_id = str(doc.get("document_id", doc.get("id", f"doc_{doc_idx}")))
        passage_texts: List[str] = []
        for p in doc["passages"]:
            p_text = p["text"]
            if isinstance(p_text, list):
                p_text = " ".join(p_text)
            passage_texts.append(p_text)

        full_text = "\n".join(passage_texts)
        sent_spans = sent_tokenize_offsets(full_text)

        entities: List[dict] = []
        for ent in doc["entities"]:
            parsed = parse_entity(ent)
            if parsed is None:
                continue
            sent_idx = find_sentence_index(sent_spans, parsed["start"], parsed["end"])
            if sent_idx is None:
                continue
            parsed["sent_idx"] = sent_idx
            entities.append(parsed)

        ent_by_id: Dict[str, dict] = {e["id"]: e for e in entities}
        ents_by_sent: Dict[int, List[dict]] = defaultdict(list)
        for e in entities:
            ents_by_sent[e["sent_idx"]].append(e)

        # Build undirected relation map with sentence alignment.
        # pair_to_rels[(id_a,id_b)] -> set(relation labels)
        pair_to_rels: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        for rel in doc["relations"]:
            try:
                arg1, arg2, rtype = parse_relation(rel)
            except Exception:
                continue
            if arg1 not in ent_by_id or arg2 not in ent_by_id:
                continue
            e1 = ent_by_id[arg1]
            e2 = ent_by_id[arg2]
            if e1["sent_idx"] != e2["sent_idx"]:
                continue
            # Keep only chem-gene relations for this setup.
            if not (
                (is_chem(e1["type"]) and is_gene(e2["type"]))
                or (is_gene(e1["type"]) and is_chem(e2["type"]))
            ):
                continue

            pid = undirected_pair_id(arg1, arg2)
            pair_to_rels[pid].add(rtype)
            observed_relations.add(rtype)

        for sent_idx, sent_ents in ents_by_sent.items():
            s_start, s_end = sent_spans[sent_idx]
            sent_text = full_text[s_start:s_end]
            sentence_id = f"{split_name}:{doc_id}:{sent_idx}"

            chems = [e for e in sent_ents if is_chem(e["type"])]
            genes = [e for e in sent_ents if is_gene(e["type"])]
            if not chems or not genes:
                continue

            # Build pair list once for this sentence.
            pairs: List[Tuple[dict, dict, Tuple[str, str], Set[str]]] = []
            for chem in chems:
                for gene in genes:
                    pid = undirected_pair_id(chem["id"], gene["id"])
                    rels = pair_to_rels.get(pid, set())
                    pairs.append((chem, gene, pid, rels))

            # Classic rows (single label, no directionality).
            multi_label_pairs = 0
            for chem, gene, pid, rels in pairs:
                if len(rels) == 0:
                    label = "no_relation"
                elif len(rels) == 1:
                    label = next(iter(rels))
                else:
                    # Keep a deterministic single label for classic formulation.
                    multi_label_pairs += 1
                    label = sorted(rels)[0]

                classic_rows.append(
                    {
                        "id": f"{sentence_id}|pair:{pid[0]}::{pid[1]}",
                        "sentence_id": sentence_id,
                        "doc_id": doc_id,
                        "sent_idx": sent_idx,
                        "text": sent_text,
                        "e1_start": chem["start"] - s_start,
                        "e1_end": chem["end"] - s_start,
                        "e2_start": gene["start"] - s_start,
                        "e2_end": gene["end"] - s_start,
                        "e1_text": chem["text"],
                        "e2_text": gene["text"],
                        "pair_id": list(pid),
                        "label": label,
                    }
                )
            if multi_label_pairs:
                # Intentionally silent in rows; aggregated count can be added later if needed.
                pass

            # Relation-conditioned span rows (existence-conditioned):
            # create rows only for relations that are actually present in this sentence.
            # This avoids query expansion into all-O negatives.
            present_relations: Set[str] = set()
            for _chem, _gene, _pid, rels in pairs:
                present_relations.update(rels)
            if relation_vocab is not None:
                # Keep only relations seen in train vocab for stability.
                present_relations = {r for r in present_relations if r in relation_vocab}
            query_relations = sorted(present_relations)

            tokens, tok_offsets = tokenize_with_offsets(sent_text)
            for rel_name in query_relations:
                tags = ["O"] * len(tokens)
                pos_pair_count = 0

                for chem, gene, _pid, rels in pairs:
                    if rel_name not in rels:
                        continue
                    pos_pair_count += 1

                    chem_start = chem["start"] - s_start
                    chem_end = chem["end"] - s_start
                    gene_start = gene["start"] - s_start
                    gene_end = gene["end"] - s_start

                    chem_tok_idxs = token_indices_for_span(tok_offsets, chem_start, chem_end)
                    gene_tok_idxs = token_indices_for_span(tok_offsets, gene_start, gene_end)
                    # Typed arguments while keeping non-directional relation matching.
                    apply_bio(tags, chem_tok_idxs, "CHEM")
                    apply_bio(tags, gene_tok_idxs, "GENE")

                if pos_pair_count > 0:
                    span_rows.append(
                        {
                            "id": f"{sentence_id}|rel:{rel_name}",
                            "sentence_id": sentence_id,
                            "doc_id": doc_id,
                            "sent_idx": sent_idx,
                            "text": sent_text,
                            "relation": rel_name,
                            "tokens": tokens,
                            "bio_tags": tags,
                            "is_null": False,
                            "positive_pair_count": pos_pair_count,
                        }
                    )

    return classic_rows, span_rows, observed_relations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument(
        "--dataset_name", type=str, default="bigbio/drugprot", help="HF dataset name"
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="drugprot_bigbio_kb",
        help="HF dataset config",
    )
    parser.add_argument(
        "--test_frac",
        type=float,
        default=0.1,
        help="Fraction of train sentence_ids reserved for test",
    )
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ds = load_dataset(args.dataset_name, args.dataset_config, trust_remote_code=True)

    # Build relation vocabulary from full train split for stable query-space.
    train_rel_vocab = collect_relation_vocab(ds["train"])
    full_train_classic, full_train_span, _ = process_split(
        ds["train"], split_name="train", relation_vocab=train_rel_vocab
    )
    valid_classic, valid_span, _ = process_split(
        ds["validation"], split_name="valid", relation_vocab=train_rel_vocab
    )

    test_sentence_ids = choose_test_sentence_ids(
        classic_rows=full_train_classic, test_frac=args.test_frac, seed=args.seed
    )
    train_classic, test_classic = split_rows_by_sentence_id(
        full_train_classic, test_sentence_ids
    )
    train_span, test_span = split_rows_by_sentence_id(full_train_span, test_sentence_ids)

    write_jsonl(train_classic, os.path.join(args.out_dir, "classic_train.jsonl"))
    write_jsonl(valid_classic, os.path.join(args.out_dir, "classic_valid.jsonl"))
    write_jsonl(test_classic, os.path.join(args.out_dir, "classic_test.jsonl"))
    write_jsonl(train_span, os.path.join(args.out_dir, "span_train.jsonl"))
    write_jsonl(valid_span, os.path.join(args.out_dir, "span_valid.jsonl"))
    write_jsonl(test_span, os.path.join(args.out_dir, "span_test.jsonl"))

    metadata = {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "directionality": "ignored (undirected pair matching)",
        "split_strategy": "train split by sentence_id into train/test; validation kept as valid",
        "test_frac": args.test_frac,
        "seed": args.seed,
        "test_sentence_ids": len(test_sentence_ids),
        "classic_train_stats": classic_stats(train_classic),
        "classic_valid_stats": classic_stats(valid_classic),
        "classic_test_stats": classic_stats(test_classic),
        "span_train_stats": span_stats(train_span),
        "span_valid_stats": span_stats(valid_span),
        "span_test_stats": span_stats(test_span),
        "relation_vocab_from_train": sorted(train_rel_vocab),
        "files": [
            "classic_train.jsonl",
            "classic_valid.jsonl",
            "classic_test.jsonl",
            "span_train.jsonl",
            "span_valid.jsonl",
            "span_test.jsonl",
        ],
    }
    with open(os.path.join(args.out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"train classic: {len(train_classic)}")
    print(f"valid classic: {len(valid_classic)}")
    print(f"test classic:  {len(test_classic)}")
    print(f"train span:    {len(train_span)}")
    print(f"valid span:    {len(valid_span)}")
    print(f"test span:     {len(test_span)}")
    print(f"relations:     {len(train_rel_vocab)}")
    print("done")


if __name__ == "__main__":
    main()
