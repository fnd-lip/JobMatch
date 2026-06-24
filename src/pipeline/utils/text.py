from __future__ import annotations

import re
import unicodedata


def normalize_text(text: object) -> str:
    if not isinstance(text, str):
        return ""

    text = text.strip().lower()
    text = remove_accents(text)
    text = re.sub(r"[^a-z0-9\s+#.-]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def remove_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))
