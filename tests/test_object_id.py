from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import mstore
from mstore import InvalidRequest, ObjectExists


def test_fixed_object_id_with_public_access(server):
    store = mstore.connect(server.endpoint)
    owner = store.create(shape=(4,), dtype=np.uint8, object_id="camera-frame")
    assert owner.object_id == "camera-frame"
    owner.numpy()[:] = [1, 2, 3, 4]
    owner.make_public("read")

    other = mstore.connect(server.endpoint)  # e.g. another process
    assert other.open("camera-frame").numpy().tolist() == [1, 2, 3, 4]


def test_duplicate_id_is_rejected(server):
    store = mstore.connect(server.endpoint)
    first = store.create(size=16, object_id="dup")
    with pytest.raises(ObjectExists):
        store.create(size=16, object_id="dup")
    assert first.info()["object"]["object_id"] == "dup"  # the original is untouched


def test_id_can_be_reused_after_delete(server):
    store = mstore.connect(server.endpoint)
    old = store.create(size=16, object_id="reuse")
    old_token = old.token
    old.delete()
    new = store.create(size=32, object_id="reuse")
    assert new.size == 32
    with pytest.raises(mstore.AuthenticationError):
        store.open("reuse", old_token)       # old object's tokens don't carry over


@pytest.mark.parametrize("bad", ["", "-leading", "has space", "a/b", "x" * 65, "é"])
def test_invalid_ids_are_rejected(server, bad):
    store = mstore.connect(server.endpoint)
    with pytest.raises(InvalidRequest):
        store.create(size=16, object_id=bad)


def test_concurrent_claims_have_exactly_one_winner(server):
    def claim(_):
        with mstore.connect(server.endpoint) as client:
            try:
                client.create(size=4096, object_id="contended").close()
                return "won"
            except ObjectExists:
                return "lost"

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(claim, range(16)))
    assert outcomes.count("won") == 1
    assert outcomes.count("lost") == 15


def test_random_ids_still_default(server):
    store = mstore.connect(server.endpoint)
    assert len(store.create(size=16).object_id) == 32
