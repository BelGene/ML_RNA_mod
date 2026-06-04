#!/usr/bin/env python3
"""Generate mean-pooled ESM-C embeddings for weak-POC FASTA sequences."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# User parameters
# ---------------------------------------------------------------------------
DEFAULT_FASTA = "data/processed/poc_weak/weak_trna_mod_sequences.faa"
DEFAULT_OUTPUT_DIR = "data/processed/poc_weak/embeddings/esmc_600m"
DEFAULT_MODEL_NAME = "biohub/ESMC-600M"
DEFAULT_MAX_TOKENS_PER_BATCH = 8192
DEFAULT_MAX_LENGTH = 2048


@dataclass(frozen=True)
class FastaRecord:
    accession: str
    header: str
    sequence: str


def find_repo_root(start: Path) -> Path:
    for candidate in start.parents:
        if (candidate / "pyproject.toml").exists() and (candidate / "src").is_dir():
            return candidate
    raise RuntimeError("Could not locate repository root from script path.")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
os.chdir(REPO_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", default=DEFAULT_FASTA)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-tokens-per-batch", type=int, default=DEFAULT_MAX_TOKENS_PER_BATCH)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--device-map",
        choices=("none", "auto"),
        default="none",
        help="Use Hugging Face Accelerate device placement. Useful for large models such as ESMC-6B.",
    )
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--long-sequence-policy", choices=("truncate", "skip"), default="truncate")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Embed only the first N records for testing.")
    return parser.parse_args()


def iter_fasta(path: str | Path) -> Iterable[FastaRecord]:
    header: str | None = None
    chunks: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    sequence = "".join(chunks).upper()
                    yield FastaRecord(accession=header.split("|", 1)[0], header=header, sequence=sequence)
                header = line[1:]
                chunks = []
            else:
                chunks.append(line)
        if header is not None:
            sequence = "".join(chunks).upper()
            yield FastaRecord(accession=header.split("|", 1)[0], header=header, sequence=sequence)


def choose_device(device_arg: str):
    import torch

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def choose_dtype(dtype_arg: str, device):
    import torch

    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32


def load_model(model_name: str, dtype, device, device_map: str):
    from transformers import AutoModel

    kwargs = {"trust_remote_code": True}
    if device_map != "none":
        kwargs["device_map"] = device_map

    try:
        model = AutoModel.from_pretrained(model_name, dtype=dtype, **kwargs)
    except TypeError:
        model = AutoModel.from_pretrained(model_name, torch_dtype=dtype, **kwargs)

    model = model.eval()
    if device_map == "none":
        model = model.to(device)
    return model


def get_input_device(model, fallback):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return fallback


def make_batches(records: list[FastaRecord], max_tokens: int, max_length: int) -> Iterable[list[FastaRecord]]:
    batch: list[FastaRecord] = []
    batch_tokens = 0
    for record in sorted(records, key=lambda item: len(item.sequence)):
        token_estimate = min(len(record.sequence) + 2, max_length)
        if batch and batch_tokens + token_estimate > max_tokens:
            yield batch
            batch = []
            batch_tokens = 0
        batch.append(record)
        batch_tokens += token_estimate
    if batch:
        yield batch


def mean_pool_embeddings(outputs, encoded) -> np.ndarray:
    import torch

    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is None:
        hidden = outputs[0]
    attention_mask = encoded["attention_mask"].bool()
    special_tokens_mask = encoded.get("special_tokens_mask")
    if special_tokens_mask is not None:
        valid = attention_mask & ~special_tokens_mask.bool()
    else:
        valid = attention_mask
    valid = valid.unsqueeze(-1)
    summed = (hidden * valid).sum(dim=1)
    counts = valid.sum(dim=1).clamp(min=1)
    pooled = summed / counts
    return pooled.detach().to(torch.float32).cpu().numpy()


def main() -> None:
    args = parse_args()
    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "This script requires torch and transformers. Install them in the Bridges environment before running."
        ) from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "embedding_manifest.tsv"
    records = list(iter_fasta(args.fasta))
    if args.limit:
        records = records[: args.limit]

    device = choose_device(args.device)
    dtype = choose_dtype(args.dtype, device)
    print(f"loading {args.model_name} on device={device} device_map={args.device_map} dtype={dtype}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = load_model(args.model_name, dtype=dtype, device=device, device_map=args.device_map)
    input_device = get_input_device(model, device)

    rows: list[dict[str, object]] = []
    completed = 0
    with torch.inference_mode():
        for batch in make_batches(records, args.max_tokens_per_batch, args.max_length):
            pending = []
            for record in batch:
                out_path = output_dir / f"{record.accession}.npy"
                truncated = len(record.sequence) + 2 > args.max_length
                if truncated and args.long_sequence_policy == "skip":
                    rows.append(
                        {
                            "accession": record.accession,
                            "sequence_length": len(record.sequence),
                            "embedding_path": "",
                            "status": "skipped_long_sequence",
                            "truncated": False,
                        }
                    )
                    continue
                if out_path.exists() and not args.overwrite:
                    rows.append(
                        {
                            "accession": record.accession,
                            "sequence_length": len(record.sequence),
                            "embedding_path": str(out_path),
                            "status": "exists",
                            "truncated": truncated,
                        }
                    )
                    continue
                pending.append(record)
            if not pending:
                continue

            encoded = tokenizer(
                [record.sequence for record in pending],
                return_tensors="pt",
                padding=True,
                truncation=args.long_sequence_policy == "truncate",
                max_length=args.max_length,
                return_special_tokens_mask=True,
            )
            encoded = {key: value.to(input_device) for key, value in encoded.items()}
            outputs = model(**{key: value for key, value in encoded.items() if key != "special_tokens_mask"})
            pooled = mean_pool_embeddings(outputs, encoded)
            for record, vector in zip(pending, pooled):
                out_path = output_dir / f"{record.accession}.npy"
                np.save(out_path, vector.astype(np.float32))
                truncated = len(record.sequence) + 2 > args.max_length
                rows.append(
                    {
                        "accession": record.accession,
                        "sequence_length": len(record.sequence),
                        "embedding_path": str(out_path),
                        "status": "written",
                        "truncated": truncated,
                    }
                )
                completed += 1
            print(f"embedded {completed}/{len(records)} newly written records", flush=True)

    with manifest_path.open("w", encoding="utf-8") as handle:
        handle.write("accession\tsequence_length\tembedding_path\tstatus\ttruncated\n")
        for row in rows:
            handle.write(
                f"{row['accession']}\t{row['sequence_length']}\t{row['embedding_path']}\t{row['status']}\t{row['truncated']}\n"
            )
    config = {
        "fasta": args.fasta,
        "output_dir": str(output_dir),
        "model_name": args.model_name,
        "max_tokens_per_batch": args.max_tokens_per_batch,
        "max_length": args.max_length,
        "device": str(device),
        "device_map": args.device_map,
        "dtype": str(dtype),
        "long_sequence_policy": args.long_sequence_policy,
    }
    (output_dir / "embedding_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
