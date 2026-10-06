"""Tests for the AReaL-owned SGLang launcher boundary."""

import os

from areal.v2.inference_service.sglang.launch_server import (
    _ensure_internal_hosts_bypass_proxy,
)


def test_sglang_self_warmup_host_is_added_to_both_no_proxy_variants(
    monkeypatch,
) -> None:
    monkeypatch.setenv("NO_PROXY", "existing.internal,127.0.0.1")
    monkeypatch.setenv("no_proxy", "another.internal")

    _ensure_internal_hosts_bypass_proxy("10.204.86.124")

    upper = os.environ["NO_PROXY"].split(",")
    lower = os.environ["no_proxy"].split(",")
    for entries in (upper, lower):
        assert "10.204.86.124" in entries
        assert "127.0.0.1" in entries
        assert "localhost" in entries
        assert "::1" in entries
        assert len(entries) == len(set(entries))


def test_sglang_unspecified_bind_does_not_add_wildcard(monkeypatch) -> None:
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)

    _ensure_internal_hosts_bypass_proxy("0.0.0.0")

    assert "0.0.0.0" not in os.environ["NO_PROXY"].split(",")
    assert os.environ["NO_PROXY"] == os.environ["no_proxy"]
