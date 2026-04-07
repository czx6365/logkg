from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import yaml


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility in lightweight experiments."""
    random.seed(seed)
    np.random.seed(seed)



def ensure_dir(path: Path) -> None:
    """Create parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)



def load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML config file."""
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict), f"YAML root must be dict: {path}"
    return data



def resolve_path(path_str: str, base_dir: Path) -> Path:
    """Resolve a possibly relative path against a config directory."""
    p = Path(path_str)
    return p if p.is_absolute() else (base_dir / p).resolve()



def save_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    """Save records to JSONL file."""
    ensure_dir(path)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")



def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load records from JSONL file."""
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out



def simple_tokenize(text: str) -> List[str]:
    """Simple lowercase tokenization for FOLS and retrieval."""
    if not text:
        return []
    return re.findall(r"[A-Za-z0-9_]+", text.lower())
