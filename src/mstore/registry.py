from __future__ import annotations

import hashlib
import re
import secrets
import threading
import time
import uuid
from typing import Any

from .backend import create_region
from .errors import (
    AuthenticationError,
    InvalidRequest,
    ObjectExists,
    ObjectNotFound,
    PermissionDenied,
    TokenExpired,
    TokenRevoked,
)
from .models import ObjectRecord, TokenRecord
from .permissions import PermissionLevel, MAX_PUBLIC_LEVEL, has_level, validate_level


# Caller-chosen ids end up in OS object names (memfd name, Windows mapping name),
# so keep them short and to a portable character set.
OBJECT_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")


def validate_object_id(object_id: str) -> str:
    if not isinstance(object_id, str) or not OBJECT_ID_PATTERN.fullmatch(object_id):
        raise InvalidRequest(
            "object_id must be 1-64 characters of letters, digits, '_', '.' or '-', "
            "starting with a letter or digit"
        )
    return object_id


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
        object_id: str | None = None,
    ) -> tuple[ObjectRecord, str]:
        if size <= 0:
            raise InvalidRequest("size must be > 0")
        if object_id is None:
            object_id = uuid.uuid4().hex
        else:
            validate_object_id(object_id)
            with self._lock:  # fail fast before allocating shared memory
                if object_id in self._objects:
                    raise ObjectExists(f"object {object_id!r} already exists")
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
            # Re-check: another client may have taken the id while the region was made.
            if object_id in self._objects:
                taken = True
            else:
                taken = False
                self._objects[object_id] = obj
                token = self._issue_locked(obj, "admin", None)
        if taken:
            region.close()
            raise ObjectExists(f"object {object_id!r} already exists")
        return obj, token

    def _issue_locked(
        self, obj: ObjectRecord, level: PermissionLevel, expires_at: float | None,
        raw: str | None = None,
    ) -> str:
        if raw is None:
            raw = secrets.token_urlsafe(32)
        elif not isinstance(raw, str) or not raw:
            raise InvalidRequest("custom token must be a non-empty string")
        elif _hash_token(raw) in self._tokens:
            # Tokens are looked up globally; a duplicate would hijack another object's token.
            raise InvalidRequest("token is already in use")
        now = time.time()
        record = TokenRecord(
            token_hash=_hash_token(raw),
            object_id=obj.object_id,
            level=level,
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
        required: PermissionLevel,
    ) -> tuple[ObjectRecord, TokenRecord | None]:
        validate_level(required)
        with self._lock:
            obj = self._objects.get(object_id)
            if obj is None:
                raise ObjectNotFound(f"object {object_id!r} does not exist")
            if not raw_token:
                if has_level(obj.public_level, required):
                    return obj, None
                raise AuthenticationError("missing token")
            token = self._tokens.get(_hash_token(raw_token))
            if token is None or token.object_id != object_id:
                raise AuthenticationError("invalid token")
            if token.revoked:
                raise TokenRevoked("token has been revoked")
            if token.expires_at is not None and time.time() >= token.expires_at:
                raise TokenExpired("token has expired")
            if token.generation != obj.generation:
                raise AuthenticationError("token belongs to a stale object generation")
            if not has_level(token.level, required):
                raise PermissionDenied(f"requires {required!r} level; token has {token.level!r}")
            return obj, token

    def issue_token(
        self,
        object_id: str,
        issuer_raw: str,
        level: PermissionLevel,
        expires_in: float | None,
        token: str | None = None,
    ) -> str:
        validate_level(level)
        with self._lock:
            obj, issuer = self.validate(object_id, issuer_raw, "share")
            issuer_level = issuer.level if issuer else obj.public_level
            issuer_expires_at = issuer.expires_at if issuer else None
            if not has_level(issuer_level, level):
                raise PermissionDenied("cannot delegate a level higher than the issuer's")
            expires_at = None
            if expires_in is not None:
                if expires_in <= 0:
                    raise InvalidRequest("expires_in must be > 0")
                expires_at = time.time() + float(expires_in)
                if issuer_expires_at is not None:
                    expires_at = min(expires_at, issuer_expires_at)
            elif issuer_expires_at is not None:
                expires_at = issuer_expires_at
            return self._issue_locked(obj, level, expires_at, token)

    def set_public(
        self, object_id: str, issuer_raw: str, level: PermissionLevel
    ) -> PermissionLevel:
        validate_level(level)
        if not has_level(MAX_PUBLIC_LEVEL, level):
            raise InvalidRequest(f"public access is limited to {MAX_PUBLIC_LEVEL!r}")
        with self._lock:
            obj, issuer = self.validate(object_id, issuer_raw, "share")
            issuer_level = issuer.level if issuer else obj.public_level
            if not has_level(issuer_level, level):
                raise PermissionDenied("cannot make public a level higher than the issuer's")
            obj.public_level = level
            return level

    def revoke_token(self, object_id: str, issuer_raw: str, target_raw: str) -> None:
        with self._lock:
            self.validate(object_id, issuer_raw, "share")
            target = self._tokens.get(_hash_token(target_raw))
            if target is None or target.object_id != object_id:
                raise AuthenticationError("target token is invalid")
            if target.level == "admin" and target_raw == issuer_raw:
                raise InvalidRequest("refusing to revoke the token currently authorizing this request")
            target.revoked = True

    def delete_object(self, object_id: str, raw_token: str) -> None:
        with self._lock:
            obj, _ = self.validate(object_id, raw_token, "admin")
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
