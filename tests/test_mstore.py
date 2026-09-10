from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from mstore import Client, MStoreServer, PermissionDenied, TokenRevoked


@pytest.fixture
def server():
    if os.name == "nt":
        pytest.skip("Linux CI test uses AF_UNIX; Windows AF_PIPE requires Windows")
    path = tempfile.mktemp(prefix="mstore-test-", suffix=".sock")
    srv = MStoreServer(f"unix://{path}")
    thread = srv.start_in_thread()
    yield srv
    srv.shutdown()
    thread.join(timeout=2)


def test_persistent_create_open_mutate_and_readonly(server):
    with Client(server.endpoint) as client:
        owner = client.create(shape=(32,), dtype=np.int64)
        arr = owner.numpy()
        arr[:] = np.arange(32)
        token = owner.issue("write")

        with client.open(owner.object_id, token, mode="write") as other:
            other.numpy()[0] = 1234

        read_token = owner.issue("read")
        with client.open(owner.object_id, read_token, mode="read") as ro:
            view = ro.numpy()
            assert view[0] == 1234
            assert not view.flags.writeable
            with pytest.raises(ValueError):
                view[0] = 9

        with pytest.raises(PermissionDenied):
            client.open(owner.object_id, read_token, mode="write")

        del arr
        owner.delete()
        owner.close()


def test_cache_hit_bypasses_control_plane_and_mapping_attach(server):
    with Client(server.endpoint, cache_size=4) as client:
        owner = client.create(shape=(8,), dtype=np.uint8)
        token = owner.issue("write")

        first = client.open(owner.object_id, token, mode="write", cache=True)
        assert not first.cache_hit
        state = first._state
        first.close()

        # If a cache hit accidentally touches _request(), fail immediately. Since
        # _open_mapping() is only reached after _request(), this also proves there is no
        # second OS mapping attachment on the hit path.
        original_request = client._request

        def forbidden_request(*args, **kwargs):
            raise AssertionError("cache hit contacted the mstore control plane")

        client._request = forbidden_request  # type: ignore[method-assign]
        second = client.open(owner.object_id, token, mode="write", cache=True)
        assert second.cache_hit
        assert second._state is state
        second.numpy()[0] = 77
        second.close()
        client._request = original_request  # type: ignore[method-assign]

        info = client.cache_info()
        assert info["hits"] == 1
        assert info["misses"] == 1
        assert info["size"] == 1

        assert client.clear_cache(owner.object_id) == 1
        owner.delete()
        owner.close()


def test_revocation_clears_same_client_cached_capability(server):
    with Client(server.endpoint, cache_size=4) as client:
        owner = client.create(shape=(8,), dtype=np.uint8)
        token = owner.issue("read")

        cached = client.open(owner.object_id, token, mode="read", cache=True)
        cached.close()
        assert client.cache_info()["size"] == 1

        owner.revoke(token)
        assert client.cache_info()["size"] == 0

        with pytest.raises(TokenRevoked):
            client.open(owner.object_id, token, mode="read", cache=True)

        owner.delete()
        owner.close()
