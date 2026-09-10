from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
import time

import numpy as np
import pytest

import mstore
from mstore.errors import ObjectNotFound, PermissionDenied, TokenExpired, TokenRevoked
from mstore.server import MStoreServer


def test_mutable_zero_copy_views(server):
    store = mstore.connect(server.endpoint)
    owner = store.create(shape=(64, 64, 3), dtype=np.uint8, metadata={"kind": "image"})
    a = owner.numpy()
    a.fill(7)

    write_token = owner.issue("write")
    writer = store.open(owner.object_id, write_token, mode="write")
    b = writer.numpy()
    b[0, 0] = [10, 20, 30]

    assert a[0, 0].tolist() == [10, 20, 30]


def test_read_token_maps_read_only(server):
    store = mstore.connect(server.endpoint)
    owner = store.create(shape=(16,), dtype=np.uint8)
    owner.numpy()[:] = np.arange(16, dtype=np.uint8)
    read_token = owner.issue("read")

    reader = store.open(owner.object_id, read_token, mode="read")
    arr = reader.numpy()
    assert arr.flags.writeable is False
    assert arr.tolist() == list(range(16))
    with pytest.raises(ValueError):
        arr[0] = 99
    with pytest.raises(PermissionDenied):
        store.open(owner.object_id, read_token, mode="write")


def test_revoke_blocks_future_open_but_existing_mapping_survives(server):
    store = mstore.connect(server.endpoint)
    owner = store.create(shape=(8,), dtype=np.uint8)
    token = owner.issue("write")
    existing = store.open(owner.object_id, token, mode="write")
    arr = existing.numpy()
    arr[0] = 1

    owner.revoke(token)
    with pytest.raises(TokenRevoked):
        store.open(owner.object_id, token, mode="write")

    arr[0] = 2
    assert owner.numpy()[0] == 2


def test_expiring_token(server):
    store = mstore.connect(server.endpoint)
    owner = store.create(shape=(4,), dtype=np.uint8)
    token = owner.issue("read", expires_in=0.03)
    time.sleep(0.05)
    with pytest.raises(TokenExpired):
        store.open(owner.object_id, token, mode="read")


def test_delete_blocks_new_opens_existing_mapping_survives(server):
    store = mstore.connect(server.endpoint)
    owner = store.create(shape=(8,), dtype=np.uint8)
    read_token = owner.issue("read")
    existing = store.open(owner.object_id, read_token, mode="read")
    existing_arr = existing.numpy()
    owner.numpy()[0] = 55
    assert existing_arr[0] == 55

    owner.delete()
    with pytest.raises(ObjectNotFound):
        store.open(owner.object_id, read_token, mode="read")
    assert existing_arr[0] == 55


def test_raw_buffer(server):
    store = mstore.connect(server.endpoint)
    owner = store.create(size=128)
    view = owner.buffer()
    view[:4] = b"MSTR"
    token = owner.issue("read")
    reader = store.open(owner.object_id, token)
    assert bytes(reader.buffer()[:4]) == b"MSTR"


def test_independent_python_process_mutates_same_pages(server):
    import os
    import subprocess
    import sys

    store = mstore.connect(server.endpoint)
    owner = store.create(shape=(1024,), dtype=np.uint8)
    owner.numpy().fill(3)
    token = owner.issue("write")

    code = r'''
import sys
import numpy as np
import mstore
endpoint, object_id, token = sys.argv[1:4]
obj = mstore.connect(endpoint).open(object_id, token, mode="write")
arr = obj.numpy()
np.add(arr, 4, out=arr, casting="unsafe")
'''
    env = os.environ.copy()
    subprocess.run(
        [sys.executable, "-c", code, server.endpoint, owner.object_id, token],
        check=True,
        env=env,
        timeout=10,
    )
    assert np.all(owner.numpy() == 7)


def test_client_reuses_one_control_connection_across_requests(server):
    store = mstore.connect(server.endpoint)

    assert store.ping()["pong"] is True
    connection = store._conn
    assert connection is not None

    assert store.ping()["pong"] is True
    assert store._conn is connection


def test_client_serializes_persistent_connection_across_threads(server):
    store = mstore.connect(server.endpoint)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: store.ping(), range(32)))

    assert all(result["pong"] is True for result in results)
    assert len(server._connections) == 1


def test_cached_open_reuses_mapping_without_control_request(server):
    store = mstore.connect(server.endpoint)
    owner = store.create(shape=(8,), dtype=np.uint8)
    read_token = owner.issue("read")

    first = store.open(owner.object_id, read_token, cache=True)
    state = first._state
    first.close()
    second = store.open(owner.object_id, read_token, cache=True)

    assert second._state is state
    assert second.numpy().shape == (8,)


def test_cached_mapping_is_an_established_capability_lease(server):
    store = mstore.connect(server.endpoint)
    owner = store.create(shape=(8,), dtype=np.uint8)
    read_token = owner.issue("read")

    cached = store.open(owner.object_id, read_token, cache=True)
    state = cached._state
    cached.close()
    owner.revoke(read_token)

    reused = store.open(owner.object_id, read_token, cache=True)
    assert reused._state is state
    reused.close()

    store.clear_cache(owner.object_id)
    assert state.mapping.closed
    with pytest.raises(TokenRevoked):
        store.open(owner.object_id, read_token, cache=True)


def test_cache_eviction_waits_for_live_shared_object(server):
    store = mstore.connect(server.endpoint, cache_size=1)
    first_owner = store.create(size=8)
    second_owner = store.create(size=8)
    first_token = first_owner.issue("read")
    second_token = second_owner.issue("read")

    first = store.open(first_owner.object_id, first_token, cache=True)
    first_state = first._state
    second = store.open(second_owner.object_id, second_token, cache=True)

    assert not first_state.mapping.closed
    first.close()
    assert first_state.mapping.closed

    second_state = second._state
    second.close()
    store.close()
    assert second_state.mapping.closed


def test_client_context_manager_closes_control_session(server):
    with mstore.connect(server.endpoint) as store:
        assert store.ping()["pong"] is True
        assert store._conn is not None

    assert store._conn is None
    with pytest.raises(RuntimeError, match="client is closed"):
        store.ping()


@pytest.mark.skipif(os.name != "nt", reason="Windows-only TCP fallback")
def test_windows_tcp_control_fallback():
    srv = MStoreServer("tcp://127.0.0.1:0")
    thread = srv.start_in_thread()
    try:
        with mstore.connect(srv.endpoint) as store:
            assert store.ping()["pong"] is True
    finally:
        srv.shutdown()
        thread.join(timeout=3)
