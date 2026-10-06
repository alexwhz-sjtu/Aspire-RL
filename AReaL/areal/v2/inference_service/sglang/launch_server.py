# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# Adapted from sglang.srt.entrypoints.http_server.launch_server
# (SGLang commit pinned in this repo).
#
# AReaL additions are between # ---- BEGIN AREAL ---- / # ---- END AREAL ----
# markers. Everything else mirrors the upstream launch_server flow.
# ---------------------------------------------------------------------------

from __future__ import annotations

import os
import sys


def _ensure_internal_hosts_bypass_proxy(host: str) -> None:
    """Keep SGLang's self-warmup request off outbound HTTP proxies.

    Upstream SGLang performs warmup by issuing HTTP requests to its own listen
    address.  On clusters with global ``HTTP_PROXY`` settings, a routable node
    IP is not necessarily present in ``NO_PROXY`` and the self-request can be
    sent to the outbound proxy instead.  Update both conventional spellings
    before SGLang creates subprocesses or HTTP clients.
    """
    required = ["127.0.0.1", "localhost", "::1"]
    normalized_host = host.strip("[]")
    if normalized_host not in {"", "0.0.0.0", "::"}:
        required.append(normalized_host)

    for variable in ("NO_PROXY", "no_proxy"):
        current = [item.strip() for item in os.environ.get(variable, "").split(",")]
        entries = [item for item in current if item]
        seen = set(entries)
        for item in required:
            if item not in seen:
                entries.append(item)
                seen.add(item)
        os.environ[variable] = ",".join(entries)


def areal_launch_server(server_args) -> None:
    # SGLang creates scheduler subprocesses and its warmup client below, so the
    # proxy boundary must be established before either is initialized.
    _ensure_internal_hosts_bypass_proxy(server_args.host)

    from sglang.srt.entrypoints.engine import Engine, init_tokenizer_manager
    from sglang.srt.entrypoints.http_server import (
        _execute_server_warmup,
        _setup_and_run_http_server,
        app,
    )
    from sglang.srt.managers.detokenizer_manager import run_detokenizer_process

    # ---- BEGIN AREAL ----
    from areal.v2.inference_service.sglang.awex import (
        register_awex_endpoints,
    )
    from areal.v2.inference_service.sglang.rpc_proxy import RpcProxy
    from areal.v2.inference_service.sglang.scheduler import (
        areal_run_scheduler_process,
        create_result_ipc,
    )
    # ---- END AREAL ----

    # ---- BEGIN AREAL ----
    result_ipc = create_result_ipc()
    # ---- END AREAL ----

    (
        tokenizer_manager,
        template_manager,
        port_args,
        scheduler_init_result,
        subprocess_watchdog,
    ) = Engine._launch_subprocesses(
        server_args=server_args,
        init_tokenizer_manager_func=init_tokenizer_manager,
        # ---- BEGIN AREAL ----
        run_scheduler_process_func=areal_run_scheduler_process,
        # ---- END AREAL ----
        run_detokenizer_process_func=run_detokenizer_process,
    )

    # ---- BEGIN AREAL ----
    if tokenizer_manager is None:
        return
    # ---- END AREAL ----

    # ---- BEGIN AREAL ----
    rpc_proxy = RpcProxy(port_args, result_ipc)
    register_awex_endpoints(app, rpc_proxy)
    # ---- END AREAL ----

    try:
        _setup_and_run_http_server(
            server_args,
            tokenizer_manager,
            template_manager,
            port_args,
            scheduler_init_result.scheduler_infos,
            subprocess_watchdog,
            execute_warmup_func=_execute_server_warmup,
        )
    finally:
        # ---- BEGIN AREAL ----
        rpc_proxy.close()
        # ---- END AREAL ----


if __name__ == "__main__":
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree
    from sglang.srt.utils.common import suppress_noisy_warnings

    suppress_noisy_warnings()

    server_args = prepare_server_args(sys.argv[1:])

    try:
        areal_launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
