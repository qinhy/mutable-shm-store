from __future__ import annotations

import numpy as np
import pytest

import mstore
from mstore import AuthenticationError, InvalidRequest, PermissionDenied
from mstore.permissions import (
    ALL_PERMISSIONS, LEVEL_ORDER, PERMISSION_LEVELS, has_level, level_of, normalize_permissions,
)


# ---- level definitions (no server needed) ----------------------------------

def test_levels_are_ordered_and_cumulative():
    assert LEVEL_ORDER == ("none", "info", "read", "write", "share", "admin")
    for lower, higher in zip(LEVEL_ORDER, LEVEL_ORDER[1:]):
        assert PERMISSION_LEVELS[lower] < PERMISSION_LEVELS[higher]
    assert PERMISSION_LEVELS["admin"] == ALL_PERMISSIONS


@pytest.mark.parametrize("level", LEVEL_ORDER)
def test_level_names_round_trip(level):
    assert level_of(normalize_permissions(level)) == level


def test_permission_sets_and_custom_levels():
    assert normalize_permissions(["write"]) == PERMISSION_LEVELS["write"]  # write implies read+info
    assert level_of(normalize_permissions(["read", "delete"])) == "custom"
    assert has_level(PERMISSION_LEVELS["share"], "write")
    assert not has_level(PERMISSION_LEVELS["read"], "write")
    with pytest.raises(InvalidRequest, match="levels are"):
        normalize_permissions("superuser")


# ---- levels on the wire ----------------------------------------------------

def test_token_info_reports_level(server):
    store = mstore.connect(server.endpoint)
    owner = store.create(size=16)
    assert owner.info()["token"]["level"] == "admin"
    for level in ("info", "read", "write", "share"):
        token = owner.issue(level)
        assert store.info(owner.object_id, token)["token"]["level"] == level


def test_share_level_can_delegate_but_not_delete(server):
    store = mstore.connect(server.endpoint)
    owner = store.create(size=16)
    sharer = store.open(owner.object_id, owner.issue("share"), mode="write")
    reader_token = sharer.issue("read")
    assert store.info(owner.object_id, reader_token)["token"]["level"] == "read"
    with pytest.raises(PermissionDenied):
        sharer.issue("admin")                     # cannot delegate above its own level
    with pytest.raises(PermissionDenied):
        sharer.delete()


def test_info_level_cannot_map(server):
    store = mstore.connect(server.endpoint)
    owner = store.create(size=16)
    token = owner.issue("info")
    assert store.info(owner.object_id, token)["object"]["size"] == 16
    with pytest.raises(PermissionDenied):
        store.open(owner.object_id, token)


# ---- public (token-less) access --------------------------------------------

def test_public_read(server):
    store = mstore.connect(server.endpoint)
    owner = store.create(shape=(4,), dtype=np.uint8)
    owner.numpy()[:] = [1, 2, 3, 4]
    with pytest.raises(AuthenticationError):
        store.open(owner.object_id)               # private by default
    owner.make_public("read")
    anon = store.open(owner.object_id)            # no token at all
    assert anon.public_level == "read"
    assert anon.numpy().tolist() == [1, 2, 3, 4]
    assert not anon.numpy().flags.writeable
    assert store.info(owner.object_id)["token"] is None
    with pytest.raises(AuthenticationError):
        store.open(owner.object_id, mode="write")  # write was not made public
    with pytest.raises(AuthenticationError):
        store.open(owner.object_id, "wrong-token")  # a bad token is still rejected
    owner.make_private()
    with pytest.raises(AuthenticationError):
        store.open(owner.object_id)
    assert anon.numpy()[0] == 1                   # existing mapping keeps working


def test_public_info_only(server):
    store = mstore.connect(server.endpoint)
    owner = store.create(size=16, metadata={"kind": "demo"})
    owner.make_public("info")
    assert store.info(owner.object_id)["object"]["public_level"] == "info"
    with pytest.raises(AuthenticationError):
        store.open(owner.object_id)


def test_public_write_and_limits(server):
    store = mstore.connect(server.endpoint)
    owner = store.create(shape=(4,), dtype=np.uint8)
    owner.make_public("write")
    writer = store.open(owner.object_id, mode="write")
    view = writer.numpy()
    view[0] = 9
    del view
    writer.close()
    assert owner.numpy()[0] == 9
    for too_high in ("share", "admin", ["delete"]):
        with pytest.raises(InvalidRequest, match="public access is limited"):
            owner.make_public(too_high)
    reader = store.open(owner.object_id, owner.issue("read"))
    with pytest.raises(PermissionDenied):
        reader.make_public("read")                # needs the grant permission


# ---- regression: closing a handle under a live NumPy view ------------------

def test_numpy_view_outlives_its_handle(server):
    store = mstore.connect(server.endpoint)
    owner = store.create(shape=(1024,), dtype=np.uint8)
    token = owner.issue("write")
    store.open(owner.object_id, token, mode="write").numpy()[0] = 9  # handle dies at once
    arr = store.open(owner.object_id, token, mode="write").numpy()
    arr[:] = 5                                     # used to segfault: memory was unmapped
    assert owner.numpy()[0] == 5


@pytest.mark.parametrize("order", ["C", "F"])
def test_numpy_view_shape_and_order(server, order):
    store = mstore.connect(server.endpoint)
    owner = store.create(shape=(3, 4), dtype=np.int32, order=order)
    arr = owner.numpy()
    arr[:] = np.arange(12, dtype=np.int32).reshape(3, 4)
    assert arr.flags.f_contiguous if order == "F" else arr.flags.c_contiguous
    reader = store.open(owner.object_id, owner.issue("read")).numpy()
    assert reader.tolist() == arr.tolist()
