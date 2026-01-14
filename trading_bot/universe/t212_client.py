"""Trading212 API client for universe ingestion."""

from __future__ import annotations

import base64
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
        """Fetch instruments from Trading212 using endpoint discovery."""

        endpoint, response_text, json_data = self._discover_instruments_endpoint()
        return Trading212Response(endpoint=endpoint, raw_text=response_text, json_data=json_data)

    def _discover_instruments_endpoint(self) -> tuple[str, str, Any]:
        candidates = [
            "/equity/metadata/instruments",
        ]
        last_error: Exception | None = None
        last_status: int | None = None
        for path in candidates:
            try:
                response = self._get(path)
                last_status = response.status_code
                response_text = response.text
                json_data = response.json()
            except Exception as exc:  # noqa: BLE001 - surface discovery errors
                last_error = exc
                if isinstance(exc, requests.HTTPError) and exc.response is not None:
                    last_status = exc.response.status_code
                    self.logger.debug(
                        "Endpoint discovery failed for %s: %s (status=%s)",
                        path,
                        exc,
                        exc.response.status_code,
                    )
                else:
                    self.logger.debug("Endpoint discovery failed for %s: %s", path, exc)
                continue

            if _looks_like_instrument_payload(json_data):
                self.logger.info("Discovered Trading212 instruments endpoint: %s", path)
                return path, response_text, json_data

            self.logger.info(
                "Endpoint %s did not resemble instruments payload; keys=%s",
                path,
                _summarize_keys(json_data),
            )

        message = (
            "Unable to discover Trading212 instruments endpoint. "
            f"Last status={last_status}"
        )
        if last_error:
            raise RuntimeError(message) from last_error
        raise RuntimeError(message)

    def _get(self, path: str) -> requests.Response:
        url = f"{self.base_url}/{path.lstrip('/')}"
        response = self.session.get(url, headers=self._headers(), timeout=self.timeout)
        response.raise_for_status()
        return response

    def _headers(self) -> dict[str, str]:
        credentials = f"{self.api_key}:{self.api_secret}".encode("utf-8")
        auth_value = base64.b64encode(credentials).decode("utf-8")
        return {
            "Accept": "application/json",
            "Authorization": f"Basic {auth_value}",
            "X-API-KEY": self.api_key,
            "X-API-SECRET": self.api_secret,
        }


def _looks_like_instrument_payload(payload: Any) -> bool:
    if isinstance(payload, list):
        return bool(payload) and all(isinstance(item, dict) for item in payload[:5])
    if isinstance(payload, dict):
        if "items" in payload and isinstance(payload["items"], list):
            items = payload["items"]
            return bool(items) and all(isinstance(item, dict) for item in items[:5])
        for value in payload.values():
            if isinstance(value, list) and value and all(isinstance(item, dict) for item in value[:5]):
                return True
    return False


def _summarize_keys(payload: Any) -> list[str]:
    if isinstance(payload, dict):
        return sorted(payload.keys())
    if isinstance(payload, list):
        return ["list"]
    return [type(payload).__name__]
