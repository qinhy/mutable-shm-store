from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import pytest

from mstore.server import MStoreServer


@pytest.fixture
def server(tmp_path: Path):
    if os.name == "nt":
        endpoint = f"pipe://mstore-test-{uuid4().hex}"
    else:
        endpoint = f"unix://{tmp_path / 'mstore.sock'}"
    srv = MStoreServer(endpoint)
    thread = srv.start_in_thread()
    # start_in_thread waits until either listener flavor has been published.
    for _ in range(200):
        try:
            if srv.endpoint.startswith("pipe://"):
                break
            if Path(srv.endpoint[len("unix://") :]).exists():
                break
        except OSError:
            pass
        thread.join(0.005)
    yield srv
    srv.shutdown()
    thread.join(timeout=3)
