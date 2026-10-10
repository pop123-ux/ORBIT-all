from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def split_bucket(text: str, buckets: int = 10_000) -> int:
    digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
    return int.from_bytes(digest[:8], "big") % buckets


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a deterministic local token corpus")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--subset", default="sample-10BT")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--tokenizer", default="gpt2")
    parser.add_argument("--train-tokens", type=int, default=50_000_000)
    parser.add_argument("--val-tokens", type=int, default=2_000_000)
    parser.add_argument("--val-permille", type=int, default=20)
    parser.add_argument(
        "--paper-mode",
        action="store_true",
        help="Refuse an unpinned dataset revision. Use an immutable commit/revision for evidence runs.",
    )
    args = parser.parse_args()
    if not (1 <= args.val_permille < 10_000):
        raise SystemExit("--val-permille must be between 1 and 9999")
    if args.paper_mode and not args.revision:
        raise SystemExit("--paper-mode requires --revision with an immutable dataset revision")

    from datasets import load_dataset
    from transformers import AutoTokenizer

    work = Path(args.work_dir)
    data_dir = work / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_tmp = data_dir / "train.bin.tmp"
    val_tmp = data_dir / "val.bin.tmp"
    train_final = data_dir / "train.bin"
    val_final = data_dir / "val.bin"

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    eos = tokenizer.eos_token_id
    stream = load_dataset(
        args.dataset,
        args.subset,
        split="train",
        streaming=True,
        revision=args.revision,
    )

    train_count = 0
    val_count = 0
    docs = 0
    with train_tmp.open("wb") as train_f, val_tmp.open("wb") as val_f:
        for row in stream:
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            if eos is not None:
                ids.append(int(eos))
            if not ids:
                continue
            array = np.asarray(ids, dtype=np.int32)
            is_val = split_bucket(text) < int(args.val_permille)
            if is_val and val_count < args.val_tokens:
                remaining = args.val_tokens - val_count
                chunk = array[:remaining]
                chunk.tofile(val_f)
                val_count += int(len(chunk))
            elif (not is_val) and train_count < args.train_tokens:
                remaining = args.train_tokens - train_count
                chunk = array[:remaining]
                chunk.tofile(train_f)
                train_count += int(len(chunk))
            docs += 1
            if docs % 1000 == 0:
                print(
                    f"documents={docs:,} train={train_count:,}/{args.train_tokens:,} "
                    f"val={val_count:,}/{args.val_tokens:,}",
                    flush=True,
                )
            if train_count >= args.train_tokens and val_count >= args.val_tokens:
                break

    if train_count < args.train_tokens or val_count < args.val_tokens:
        raise RuntimeError(
            f"stream ended early: train={train_count}, val={val_count}; temporary files retained"
        )
    train_tmp.replace(train_final)
    val_tmp.replace(val_final)
    manifest = {
        "dataset": args.dataset,
        "subset": args.subset,
        "revision": args.revision,
        "tokenizer": args.tokenizer,
        "train_tokens": train_count,
        "val_tokens": val_count,
        "val_permille": args.val_permille,
        "documents_scanned": docs,
        "split": "sha256(text) modulo 10000",
        "dtype": "int32",
        "paper_mode": bool(args.paper_mode),
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
