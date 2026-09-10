from __future__ import annotations

import os
import sys
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import pytest

from mstore.server import MStoreServer


@pytest.fixture
def server():
    if sys.platform != "darwin" and os.name != "nt" and not hasattr(os, "memfd_create"):
        pytest.skip("storage backend requires Linux, macOS, or Windows")
    with ExitStack() as stack:
        if os.name == "nt":
            endpoint = f"pipe://mstore-test-{uuid4().hex}"
        else:
            # pytest's per-test directories can exceed the AF_UNIX path limit.
            directory = stack.enter_context(TemporaryDirectory(prefix="mstore-", dir="/tmp"))
            endpoint = f"unix://{Path(directory) / 'store.sock'}"
        srv = MStoreServer(endpoint)
        thread = srv.start_in_thread()
        try:
            yield srv
        finally:
            srv.shutdown()
            thread.join(timeout=3)
