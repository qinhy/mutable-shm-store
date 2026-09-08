from __future__ import annotations

import os
from pathlib import Path

import pytest

from mstore.server import MStoreServer


@pytest.fixture
def server(tmp_path: Path):
    if os.name == "nt":
        endpoint = "tcp://127.0.0.1:0"
    else:
        endpoint = f"unix://{tmp_path / 'mstore.sock'}"
    srv = MStoreServer(endpoint)
    thread = srv.start_in_thread()
    # For TCP port 0, endpoint is updated after bind.
    for _ in range(200):
        try:
            kind = "tcp" if srv.endpoint.startswith("tcp://") else "unix"
            if kind == "tcp" and not srv.endpoint.endswith(":0"):
                break
            if kind == "unix" and Path(srv.endpoint[len("unix://"):]).exists():
                break
        except OSError:
            pass
        thread.join(0.005)
    yield srv
    srv.shutdown()
    thread.join(timeout=3)
