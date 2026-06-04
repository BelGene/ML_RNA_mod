from __future__ import annotations

import hashlib
from pathlib import Path


def file_md5(path: str | Path) -> str:
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return ""
    digest = hashlib.md5()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return ""
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_md5(sequence: str | None) -> str:
    cleaned = (sequence or "").replace(" ", "").replace("\n", "").upper()
    if not cleaned:
        return ""
    return hashlib.md5(cleaned.encode("ascii")).hexdigest()


def sequence_sha256(sequence: str | None) -> str:
    cleaned = (sequence or "").replace(" ", "").replace("\n", "").upper()
    if not cleaned:
        return ""
    return hashlib.sha256(cleaned.encode("ascii")).hexdigest()

