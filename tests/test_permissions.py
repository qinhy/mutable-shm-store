from __future__ import annotations

import numpy as np
import pytest
from typing import get_args

import mstore
from mstore import AuthenticationError, InvalidRequest, PermissionDenied
from mstore.permissions import LEVEL_ORDER, PermissionLevel, has_level, validate_level


# ---- level definitions (no server needed) ----------------------------------

def test_levels_are_ordered_and_cumulative():
    assert LEVEL_ORDER == ("none", "info", "read", "write", "share", "admin")
    assert get_args(PermissionLevel) == LEVEL_ORDER
    for available_index, available in enumerate(LEVEL_ORDER):
        for required_index, required in enumerate(LEVEL_ORDER):
            assert has_level(available, required) is (available_index >= required_index)


@pytest.mark.parametrize("level", LEVEL_ORDER)
def test_level_names_validate(level):
    assert validate_level(level) == level


@pytest.mark.parametrize(
    "invalid",
    (["write"], ["read", "delete"], {"write"}, ("read",),
     "delete", "grant", "READ", "superuser", "", None, 0),
)
def test_arbitrary_permissions_and_invalid_levels_are_rejected(invalid):
    with pytest.raises(InvalidRequest, match="level must be one of"):
        validate_level(invalid)
    with pytest.raises(InvalidRequest, match="level must be one of"):
        has_level(invalid, "read")
    with pytest.raises(InvalidRequest, match="level must be one of"):
        has_level("admin", invalid)


def test_legacy_permission_set_helpers_are_removed():
    import mstore.permissions as permissions

    for name in ("ALL_PERMISSIONS", "PERMISSION_LEVELS", "normalize_permissions", "level_of"):
        assert not hasattr(permissions, name)
        assert not hasattr(mstore, name)


# ---- levels on the wire ----------------------------------------------------

def test_token_info_reports_level(server):
    store = mstore.connect(server.endpoint)
    owner = store.create(size=16)
    assert owner.info()["token"]["level"] == "admin"
    assert owner.info()["object"]["public_level"] == "none"
    for level in ("info", "read", "write", "share"):
        token = owner.issue(level=level)
        metadata = store.info(owner.object_id, token)
        assert metadata["token"]["level"] == level
        assert "permissions" not in metadata["token"]
        assert "public" not in metadata["object"]


def test_share_level_can_delegate_but_not_delete(server):
    store = mstore.connect(server.endpoint)
    owner = store.create(size=16)
    sharer = store.open(owner.object_id, owner.issue(level="share"), mode="write")
    reader_token = sharer.issue(level="read")
    assert store.info(owner.object_id, reader_token)["token"]["level"] == "read"
    with pytest.raises(PermissionDenied):
        sharer.issue(level="admin")                     # cannot delegate above its own level
    with pytest.raises(PermissionDenied):
        sharer.delete()


def test_info_level_cannot_map(server):
    store = mstore.connect(server.endpoint)
    owner = store.create(size=16)
    token = owner.issue(level="info")
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
    owner.make_public(level="read")
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
    owner.make_public(level="info")
    assert store.info(owner.object_id)["object"]["public_level"] == "info"
    with pytest.raises(AuthenticationError):
        store.open(owner.object_id)


def test_public_write_and_limits(server):
    store = mstore.connect(server.endpoint)
    owner = store.create(shape=(4,), dtype=np.uint8)
    owner.make_public(level="write")
    writer = store.open(owner.object_id, mode="write")
    view = writer.numpy()
    view[0] = 9
    del view
    writer.close()
    assert owner.numpy()[0] == 9
    reader = store.open(owner.object_id, owner.issue(level="read"))
    with pytest.raises(PermissionDenied):
        reader.make_public(level="read")                # needs the grant permission
    sharer = store.open(owner.object_id, owner.issue(level="share"), mode="write")
    with pytest.raises(PermissionDenied):
        sharer.make_public(level="admin")               # cannot publish more than it holds


def test_public_admin_lets_anyone_delete_and_grant(server):
    owner_client = mstore.connect(server.endpoint)
    owner = owner_client.create(shape=(4,), dtype=np.uint8)
    owner.make_public(level="admin")

    anyone = mstore.connect(server.endpoint)      # no token anywhere below
    assert anyone.info(owner.object_id)["object"]["public_level"] == "admin"
    handle = anyone.open(owner.object_id, mode="write")
    token = handle.issue(level="read")                  # token-less grant works
    assert anyone.info(owner.object_id, token)["token"]["level"] == "read"
    handle.make_public(level="read")                    # ... and can lower public access
    with pytest.raises(AuthenticationError):
        handle.delete()                           # which now applies to itself too
    owner.make_public(level="admin")
    handle.delete()                               # token-less delete
    with pytest.raises(mstore.ObjectNotFound):
        anyone.info(owner.object_id)


# ---- regression: closing a handle under a live NumPy view ------------------

def test_numpy_view_outlives_its_handle(server):
    store = mstore.connect(server.endpoint)
    owner = store.create(shape=(1024,), dtype=np.uint8)
    token = owner.issue(level="write")
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
    reader = store.open(owner.object_id, owner.issue(level="read")).numpy()
    assert reader.tolist() == arr.tolist()


# ---- caller-chosen token values --------------------------------------------

def test_custom_token_value(server):
    store = mstore.connect(server.endpoint)
    owner = store.create(size=16)
    assert owner.issue(level="read", token="123") == "123"
    assert store.open(owner.object_id, "123").size == 16
    other = store.create(size=16)
    with pytest.raises(InvalidRequest, match="already in use"):
        other.issue(level="read", token="123")          # tokens are global: no duplicates
    with pytest.raises(InvalidRequest):
        owner.issue(level="read", token="")
    owner.delete()
    assert other.issue(level="read", token="123") == "123"   # free again after delete
