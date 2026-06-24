from __future__ import annotations

import json
import time
import uuid
from typing import Any


def log_event(event: str, **fields: Any) -> None:
    """Log estruturado simples para rastrear execução do app."""
    payload = {
        "timestamp": time.time(),
        "trace_id": str(uuid.uuid4()),
        "event": event,
        **fields,
    }

    print(json.dumps(payload, ensure_ascii=False, default=str))
