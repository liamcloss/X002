"""Trading212 API client for universe ingestion."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class Trading212Response:
    """Container for Trading212 API responses."""

    endpoint: str
    raw_text: str
    json_data: Any


class Trading212Client:
    """Low-level Trading212 API client."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str | None = None,
        timeout: int = 30,
        session: requests.Session | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = (base_url or os.environ.get("T212_API_BASE_URL") or "").rstrip("/")
        if not self.base_url:
            self.base_url = "https://live.trading212.com/api/v0"
        self.timeout = timeout
        self.session = session or requests.Session()
        self.logger = logger or logging.getLogger("trading_bot")

    def fetch_instruments(self) -> Trading212Response:
        """Fetch instruments from Trading212 using the documented endpoint."""

        endpoint = "/instruments"
        response = self._get(endpoint)
        response_text = response.text
        json_data = response.json()
        return Trading212Response(endpoint=endpoint, raw_text=response_text, json_data=json_data)

    def _get(self, path: str) -> requests.Response:
        url = f"{self.base_url}/{path.lstrip('/')}"
        response = self.session.get(url, headers=self._headers(), timeout=self.timeout)
        response.raise_for_status()
        return response

    def _headers(self) -> dict[str, str]:
        return {
            "Accept": "application/json",
            "X-API-KEY": self.api_key,
            "X-API-SECRET": self.api_secret,
        }

