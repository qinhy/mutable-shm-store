from __future__ import annotations

import hashlib
import secrets
import threading
import time
import uuid
from typing import Any, Iterable

from .backend import create_region
from .errors import (
    AuthenticationError,
    InvalidRequest,
    ObjectNotFound,
    PermissionDenied,
    TokenExpired,
    TokenRevoked,
)
from .models import ObjectRecord, TokenRecord

READ = "read"
WRITE = "write"
GRANT = "grant"
DELETE = "delete"
INFO = "info"
ALL_PERMISSIONS = frozenset({READ, WRITE, GRANT, DELETE, INFO})


def normalize_permissions(value: str | Iterable[str]) -> frozenset[str]:
    if isinstance(value, str):
        if value == "read":
            return frozenset({READ, INFO})
        if value == "write":
            return frozenset({READ, WRITE, INFO})
        if value == "admin":
            return ALL_PERMISSIONS
        value = [value]
    result = set(value)
    unknown = result - ALL_PERMISSIONS
    if unknown:
        raise InvalidRequest(f"unknown permissions: {sorted(unknown)}")
    if WRITE in result:
        result.add(READ)
    if result:
        result.add(INFO)
    return frozenset(result)


def _hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class Registry:
    def __init__(self) -> None:
        self._objects: dict[str, ObjectRecord] = {}
        self._tokens: dict[str, TokenRecord] = {}
        self._lock = threading.RLock()

    def create_object(
        self,
        *,
        size: int,
        shape: list[int] | None,
        dtype: str | None,
        order: str,
        metadata: dict[str, Any],
    ) -> tuple[ObjectRecord, str]:
        if size <= 0:
            raise InvalidRequest("size must be > 0")
        object_id = uuid.uuid4().hex
        region = create_region(size, object_id)
        now = time.time()
        obj = ObjectRecord(
            object_id=object_id,
            size=size,
            region=region,
            shape=shape,
            dtype=dtype,
            order=order,
            metadata=metadata,
            created_at=now,
        )
        with self._lock:
            self._objects[object_id] = obj
            token = self._issue_locked(obj, ALL_PERMISSIONS, None)
        return obj, token

    def _issue_locked(
        self, obj: ObjectRecord, permissions: frozenset[str], expires_at: float | None
    ) -> str:
        raw = secrets.token_urlsafe(32)
        now = time.time()
        record = TokenRecord(
            token_hash=_hash_token(raw),
            object_id=obj.object_id,
            permissions=permissions,
            generation=obj.generation,
            created_at=now,
            expires_at=expires_at,
        )
        self._tokens[record.token_hash] = record
        return raw

    def validate(
        self,
        object_id: str,
        raw_token: str,
        required: str | Iterable[str],
    ) -> tuple[ObjectRecord, TokenRecord]:
        if not raw_token:
            raise AuthenticationError("missing token")
        required_set = normalize_permissions(required)
        # INFO is automatically inserted by normalize_permissions; it is not required
        # merely because the requested permission includes another operation.
        if isinstance(required, str) and required != INFO:
            required_set = frozenset({required})
        with self._lock:
            obj = self._objects.get(object_id)
            if obj is None:
                raise ObjectNotFound(f"object {object_id!r} does not exist")
            token = self._tokens.get(_hash_token(raw_token))
            if token is None or token.object_id != object_id:
                raise AuthenticationError("invalid token")
            if token.revoked:
                raise TokenRevoked("token has been revoked")
            now = time.time()
            if token.expires_at is not None and now >= token.expires_at:
                raise TokenExpired("token has expired")
            if token.generation != obj.generation:
                raise AuthenticationError("token belongs to a stale object generation")
            missing = required_set - token.permissions
            if missing:
                raise PermissionDenied(f"token lacks permission(s): {sorted(missing)}")
            return obj, token

    def issue_token(
        self,
        object_id: str,
        issuer_raw: str,
        permissions: str | Iterable[str],
        expires_in: float | None,
    ) -> str:
        requested = normalize_permissions(permissions)
        with self._lock:
            obj, issuer = self.validate(object_id, issuer_raw, GRANT)
            # A grant-capable token cannot delegate permissions it doesn't itself possess.
            if not requested.issubset(issuer.permissions):
                raise PermissionDenied("cannot delegate permissions the issuer does not possess")
            expires_at = None
            if expires_in is not None:
                if expires_in <= 0:
                    raise InvalidRequest("expires_in must be > 0")
                expires_at = time.time() + float(expires_in)
                if issuer.expires_at is not None:
                    expires_at = min(expires_at, issuer.expires_at)
            elif issuer.expires_at is not None:
                expires_at = issuer.expires_at
            return self._issue_locked(obj, requested, expires_at)

    def revoke_token(self, object_id: str, issuer_raw: str, target_raw: str) -> None:
        with self._lock:
            self.validate(object_id, issuer_raw, GRANT)
            target = self._tokens.get(_hash_token(target_raw))
            if target is None or target.object_id != object_id:
                raise AuthenticationError("target token is invalid")
            if target.permissions == ALL_PERMISSIONS and target_raw == issuer_raw:
                raise InvalidRequest("refusing to revoke the token currently authorizing this request")
            target.revoked = True

    def delete_object(self, object_id: str, raw_token: str) -> None:
        with self._lock:
            obj, _ = self.validate(object_id, raw_token, DELETE)
            self._objects.pop(object_id, None)
            for token_hash in [
                key for key, token in self._tokens.items() if token.object_id == object_id
            ]:
                self._tokens.pop(token_hash, None)
        # Close outside the registry lock; existing client mappings remain valid.
        obj.region.close()

    def close(self) -> None:
        with self._lock:
            objects = list(self._objects.values())
            self._objects.clear()
            self._tokens.clear()
        for obj in objects:
            try:
                obj.region.close()
            except Exception:
                pass
