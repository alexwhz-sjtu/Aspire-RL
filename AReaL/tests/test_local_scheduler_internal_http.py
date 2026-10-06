# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import requests

from areal.api import Worker
from areal.infra.scheduler.local import LocalScheduler, WorkerInfo


def _worker_info() -> WorkerInfo:
    return WorkerInfo(
        worker=Worker(
            id="actor-guard/0",
            ip="10.0.0.8",
            worker_ports=["12345"],
        ),
        process=None,
        role="actor-guard",
        gpu_devices=[0],
        created_at=0.0,
        log_file="unused.log",
    )


def test_local_worker_health_check_bypasses_host_proxy() -> None:
    response = Mock(status_code=200)
    scheduler = object.__new__(LocalScheduler)

    with patch.object(requests, "get", return_value=response) as get:
        assert scheduler._is_worker_ready(_worker_info())

    get.assert_called_once_with(
        "http://10.0.0.8:12345/health",
        timeout=2.0,
        proxies={"http": None, "https": None},
    )


def test_local_worker_health_check_treats_direct_connection_error_as_not_ready() -> None:
    scheduler = object.__new__(LocalScheduler)
    with patch.object(
        requests,
        "get",
        side_effect=requests.exceptions.ConnectionError("not listening"),
    ):
        assert not scheduler._is_worker_ready(_worker_info())


def test_local_worker_configuration_bypasses_host_proxy() -> None:
    response = Mock(status_code=200)
    scheduler = object.__new__(LocalScheduler)
    scheduler.exp_config = {"test": True}

    with (
        patch.object(scheduler, "_is_worker_ready", return_value=True),
        patch.object(requests, "post", return_value=response) as post,
    ):
        scheduler._configure_worker(_worker_info(), worker_rank=0)

    assert post.call_args.kwargs["proxies"] == {"http": None, "https": None}
