from __future__ import annotations

import time

import numpy as np
import pytest

import mstore
from mstore.errors import ObjectNotFound, PermissionDenied, TokenExpired, TokenRevoked


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
