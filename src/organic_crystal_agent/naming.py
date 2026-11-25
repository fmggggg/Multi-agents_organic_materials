from __future__ import annotations

import re


_INTERNAL_PREFIX_RE = re.compile(
    r"^(?:HOF[_-]?HC\d+|HOF\d+|HC\d+|HOF)[\s_-]*",
    flags=re.IGNORECASE,
)


def normalize_molecule_label(label: str) -> str:
    text = (label or "").strip()
    if not text:
        return ""
    text = _INTERNAL_PREFIX_RE.sub("", text)
    text = text.replace("_", " ")
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" -_")
    return text or (label or "").strip()
