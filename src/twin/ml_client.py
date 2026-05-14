from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class ExternalModelConfig:
    endpoint_url: str
    api_key: str | None = None
    timeout_seconds: int = 10


def call_external_model(config: ExternalModelConfig, payload: dict[str, Any]) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    response = requests.post(
        config.endpoint_url,
        json=payload,
        headers=headers,
        timeout=config.timeout_seconds,
    )
    response.raise_for_status()
    return response.json()
