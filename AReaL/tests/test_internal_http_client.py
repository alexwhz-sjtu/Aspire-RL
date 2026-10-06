"""Regression tests for node-internal HTTP transport isolation."""

from unittest.mock import MagicMock, patch

from areal.infra.utils.http import create_httpx_client, create_httpx_sync_client


def test_async_internal_client_disables_environment_proxy_resolution() -> None:
    transport = MagicMock()
    with (
        patch(
            "areal.infra.utils.http.httpx.AsyncHTTPTransport",
            return_value=transport,
        ),
        patch("areal.infra.utils.http.httpx.AsyncClient") as client,
    ):
        create_httpx_client(timeout=17.0)

    client.assert_called_once_with(
        timeout=17.0,
        transport=transport,
        trust_env=False,
    )


def test_sync_internal_client_disables_environment_proxy_resolution() -> None:
    transport = MagicMock()
    with (
        patch(
            "areal.infra.utils.http.httpx.HTTPTransport",
            return_value=transport,
        ),
        patch("areal.infra.utils.http.httpx.Client") as client,
    ):
        create_httpx_sync_client(timeout=23.0)

    client.assert_called_once_with(
        timeout=23.0,
        transport=transport,
        trust_env=False,
    )
