from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def discover_class_files(root: Path, extensions: Iterable[str]) -> list[tuple[str, Path, list[Path]]]:
    """Return every non-empty media directory as (class, directory, files)."""
    root = Path(root)
    allowed = {ext.lower() for ext in extensions}
    if not root.is_dir():
        return []
    found: list[tuple[str, Path, list[Path]]] = []
    for directory in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: p.as_posix().lower()):
        files = sorted(
            (p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in allowed),
            key=lambda p: p.name.lower(),
        )
        if files:
            found.append((directory.name, directory, files))
    return found


def write_class_metadata(models_dir: Path, task: str, class_names: list[str]) -> None:
    """Persist task-specific labels and a combined class_names.json audit file."""
    models_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = models_dir / "class_names.json"
    metadata: dict[str, object] = {}
    if metadata_path.exists():
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                metadata = loaded
        except (OSError, json.JSONDecodeError):
            metadata = {}
    metadata[task] = class_names
    metadata["updated_task"] = task
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)


def count_classes(root: Path, extensions: Iterable[str]) -> int:
    return len(discover_class_files(root, extensions))