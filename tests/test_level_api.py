"""Integration tests for the level-only permission API."""
from __future__ import annotations

import inspect
import pytest

from mstore import Client
from mstore.errors import AuthenticationError, InvalidRequest, PermissionDenied, TokenRevoked
from mstore.permissions import LEVEL_ORDER, has_level, validate_level


@pytest.fixture
def live(server):
    with Client(server.endpoint) as owner, Client(server.endpoint) as guest:
        yield owner, guest


def test_level_helpers():
    assert LEVEL_ORDER == ("none", "info", "read", "write", "share", "admin")
    for i, available in enumerate(LEVEL_ORDER):
        for j, required in enumerate(LEVEL_ORDER):
            assert has_level(available, required) == (i >= j)
    for invalid in (["read"], ["read", "share"], {"read"}, ("read",), "grant", "delete", None, 0, "READ"):
        with pytest.raises(InvalidRequest):
            validate_level(invalid)


def test_no_permissions_parameter_in_api():
    from mstore import SharedObject
    from mstore.registry import Registry
    for method in (SharedObject.issue, SharedObject.issue_url, SharedObject.make_public,
                   Client.issue_token, Client.set_public, Registry.issue_token, Registry.set_public):
        assert "permissions" not in inspect.signature(method).parameters
        assert "level" in inspect.signature(method).parameters


def test_read_write_share_grants_and_revocation(live):
    owner, guest = live
    with owner.create(size=16) as obj:
        assert obj.info()["token"]["level"] == "admin"
        assert obj.info()["object"]["public_level"] == "none"
        read_token = obj.issue(level="read")
        with guest.open(obj.object_id, read_token, mode="read") as view:
            assert len(view.buffer()) == 16
        with pytest.raises(PermissionDenied):
            guest.open(obj.object_id, read_token, mode="write")
        with pytest.raises(PermissionDenied):
            guest.issue_token(obj.object_id, read_token, level="read")

        writer_token = obj.issue(level="write")
        with guest.open(obj.object_id, writer_token, mode="write"):
            pass
        share_token = obj.issue(level="share")
        child = guest.issue_token(obj.object_id, share_token, level="write")
        with guest.open(obj.object_id, child, mode="write"):
            pass
        with pytest.raises(PermissionDenied):
            guest.issue_token(obj.object_id, share_token, level="admin")
        guest.revoke_token(obj.object_id, share_token, child)
        with pytest.raises(TokenRevoked):
            guest.open(obj.object_id, child)
        with pytest.raises(InvalidRequest):
            obj.issue(level=["read"])
        obj.delete()


def test_removed_permissions_keywords_are_rejected(live):
    owner, guest = live
    with owner.create(size=16) as obj:
        with pytest.raises(TypeError):
            obj.issue(permissions="read")
        with pytest.raises(TypeError):
            obj.issue_url(permissions="read")
        with pytest.raises(TypeError):
            obj.make_public(permissions="read")
        with pytest.raises(TypeError):
            guest.issue_token(obj.object_id, obj.token, permissions="read")
        with pytest.raises(TypeError):
            guest.set_public(obj.object_id, obj.token, permissions="read")
        assert obj.info()["object"]["public_level"] == "none"
        obj.delete()


def test_public_access(live):
    owner, guest = live
    with owner.create(size=16) as obj:
        assert obj.make_public(level="info") == "info"
        info = guest.info(obj.object_id)
        assert info["token"] is None
        with pytest.raises(AuthenticationError):
            guest.open(obj.object_id)
        assert obj.make_public(level="read") == "read"
        with guest.open(obj.object_id):
            pass
        with pytest.raises(AuthenticationError):
            guest.open(obj.object_id, mode="write")
        obj.make_private()
        with pytest.raises(AuthenticationError):
            guest.info(obj.object_id)
        obj.delete()


def test_public_level_ceiling_and_delegation(live, monkeypatch):
    import mstore.registry as reg
    owner, guest = live
    with owner.create(size=16) as obj:
        monkeypatch.setattr(reg, "MAX_PUBLIC_LEVEL", "write")
        with pytest.raises(InvalidRequest):
            obj.make_public(level="share")
        assert obj.make_public(level="write") == "write"
        with pytest.raises(AuthenticationError):
            guest.issue_token(obj.object_id, "", level="info")
        obj.delete()


def test_raw_protocol_rejects_legacy_field_and_invalid_levels(live):
    owner, guest = live
    with owner.create(size=16) as obj:
        for op in ("grant", "set_public"):
            for legacy in ("read", ["read"], ["read", "write"]):
                with pytest.raises(InvalidRequest):
                    guest._request(op, object_id=obj.object_id, token=obj.token,
                                   permissions=legacy)
            with pytest.raises(InvalidRequest):
                guest._request(op, object_id=obj.object_id, token=obj.token,
                               level=["read"])
        obj.delete()


def test_custom_token_and_urls(live):
    owner, guest = live
    with owner.create(size=16) as obj:
        assert obj.issue(level="read", token="my-fixed-token") == "my-fixed-token"
        with guest.open(obj.object_id, "my-fixed-token"):
            pass
        with pytest.raises(InvalidRequest):
            obj.issue(level="read", token="my-fixed-token")
        for level in ("read", "write", "share", "admin"):
            expected_mode = "read" if level == "read" else "write"
            url = obj.issue_url(level=level)
            assert f"mode={expected_mode}" in url
            with guest.open_url(url) as view:
                assert view.mode == expected_mode
                assert view.info()["token"]["level"] == level
        for level in ("none", "info"):
            with pytest.raises(ValueError, match="at least 'read'"):
                obj.issue_url(level=level)
        obj.delete()
