"""Typed-enough HTTP client for the independent ASPIRE broker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import httpx


class BrokerInfrastructureError(RuntimeError):
    pass


@dataclass(frozen=True)
class BrokerArtifact:
    data: bytes
    media_type: str
    kind: str


class AspireToolClient:
    def __init__(
        self,
        *,
        http_client: httpx.AsyncClient,
        broker_url: str,
        admin_token: str,
        protocol_version: str,
        request_timeout_seconds: float = 18000.0,
    ) -> None:
        self.http_client = http_client
        self.broker_url = broker_url.rstrip("/")
        self.admin_token = admin_token
        self.protocol_version = protocol_version
        self.request_timeout_seconds = request_timeout_seconds
        self.episode_id: str | None = None
        self.episode_token: str | None = None
        self.state_version = 0

    async def _request(
        self,
        method: str,
        path: str,
        *,
        token: str,
        json: dict[str, Any] | None = None,
    ) -> httpx.Response:
        try:
            response = await self.http_client.request(
                method,
                f"{self.broker_url}{path}",
                headers={"Authorization": f"Bearer {token}"},
                json=json,
                timeout=self.request_timeout_seconds,
            )
        except httpx.HTTPError as exc:
            raise BrokerInfrastructureError(f"ASPIRE broker request failed: {exc}") from exc
        if response.status_code >= 500:
            raise BrokerInfrastructureError(
                f"ASPIRE broker returned {response.status_code}: {response.text[:1000]}"
            )
        if response.status_code in {401, 403, 404, 409}:
            raise BrokerInfrastructureError(
                f"ASPIRE broker contract error {response.status_code}: {response.text[:1000]}"
            )
        return response

    async def create_episode(self, task_uid: str) -> dict[str, Any]:
        response = await self._request(
            "POST",
            "/v1/episodes",
            token=self.admin_token,
            json={
                "protocol_version": self.protocol_version,
                "task_uid": task_uid,
                "client_episode_id": str(uuid4()),
            },
        )
        if response.status_code != 200:
            raise BrokerInfrastructureError(
                f"episode creation failed ({response.status_code}): {response.text[:1000]}"
            )
        result = response.json()
        self.episode_id = result["episode_id"]
        self.episode_token = result["scoped_token"]
        self.state_version = int(result["state_version"])
        return result

    def _require_episode(self) -> tuple[str, str]:
        if self.episode_id is None or self.episode_token is None:
            raise BrokerInfrastructureError("episode has not been created")
        return self.episode_id, self.episode_token

    def require_episode_id(self) -> str:
        episode_id, _ = self._require_episode()
        return episode_id

    async def action(
        self,
        action: dict[str, Any],
        *,
        model_turn: int | None = None,
        cumulative_policy_output_tokens: int | None = None,
    ) -> dict[str, Any]:
        episode_id, token = self._require_episode()
        response = await self._request(
            "POST",
            f"/v1/episodes/{episode_id}/actions",
            token=token,
            json={
                "request_id": str(uuid4()),
                "expected_state_version": self.state_version,
                "model_turn": model_turn,
                "cumulative_policy_output_tokens": cumulative_policy_output_tokens,
                "action": action,
            },
        )
        if response.status_code == 422:
            return {
                "ok": False,
                "state_version": self.state_version,
                "error": {
                    "error_class": "policy",
                    "code": "invalid_tool_arguments",
                    "message": response.text[:4000],
                    "retryable": False,
                },
            }
        if response.status_code != 200:
            raise BrokerInfrastructureError(
                f"tool action failed ({response.status_code}): {response.text[:1000]}"
            )
        result = response.json()
        self.state_version = int(result["state_version"])
        return result

    async def artifact(self, artifact_id: str) -> BrokerArtifact:
        episode_id, token = self._require_episode()
        response = await self._request(
            "GET",
            f"/v1/episodes/{episode_id}/artifacts/{artifact_id}",
            token=token,
        )
        if response.status_code != 200:
            raise BrokerInfrastructureError(
                f"artifact fetch failed ({response.status_code}): {response.text[:1000]}"
            )
        return BrokerArtifact(
            data=response.content,
            media_type=response.headers.get("content-type", "application/octet-stream").split(
                ";", 1
            )[0],
            kind=response.headers.get("x-artifact-kind", "artifact"),
        )

    async def save_findings(self, findings: str) -> None:
        episode_id, token = self._require_episode()
        response = await self._request(
            "POST",
            f"/v1/episodes/{episode_id}/findings",
            token=token,
            json={"findings": findings},
        )
        if response.status_code != 200:
            raise BrokerInfrastructureError(
                f"findings archive failed ({response.status_code}): {response.text[:1000]}"
            )
        self.state_version = int(response.json()["state_version"])

    async def terminal_score(
        self, *, auto_freeze: bool, fallback_findings: str | None = None
    ) -> dict[str, Any]:
        episode_id, token = self._require_episode()
        response = await self._request(
            "POST",
            f"/v1/episodes/{episode_id}/terminal-score",
            token=token,
            json={
                "auto_freeze": auto_freeze,
                "fallback_findings": fallback_findings,
            },
        )
        if response.status_code != 200:
            raise BrokerInfrastructureError(
                f"terminal scoring failed ({response.status_code}): {response.text[:1000]}"
            )
        return response.json()

    async def close(self) -> None:
        if self.episode_id is None or self.episode_token is None:
            return
        await self._request(
            "POST",
            f"/v1/episodes/{self.episode_id}/close",
            token=self.episode_token,
            json={},
        )

    async def frozen_code(self) -> tuple[str, str]:
        episode_id, _ = self._require_episode()
        response = await self._request(
            "GET",
            f"/v1/admin/episodes/{episode_id}/frozen-code",
            token=self.admin_token,
        )
        if response.status_code != 200:
            raise BrokerInfrastructureError(
                f"frozen code export failed ({response.status_code}): "
                f"{response.text[:1000]}"
            )
        return response.headers["x-code-sha256"], response.text
