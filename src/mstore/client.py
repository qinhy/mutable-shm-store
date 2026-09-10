from __future__ import annotations

import mmap
import os
import threading
from collections import OrderedDict
from typing import Any, Iterable

import numpy as np

from .errors import ERROR_TYPES, MStoreError, ProtocolError
from .transport import ControlConnection, connect_control, default_endpoint


class _MappingState:
    """Own one OS mapping; cache ownership and SharedObject leases are independent."""

    def __init__(self, mapping: mmap.mmap, info: dict[str, Any]) -> None:
        self.mapping = mapping
        self.info = info
        self._leases = 0
        self._cached = False
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            if self.mapping.closed:
                raise RuntimeError("shared-memory mapping is closed")
            self._leases += 1

    def release(self) -> None:
        with self._lock:
            if self._leases > 0:
                self._leases -= 1
            self._maybe_close_locked()

    def set_cached(self, cached: bool) -> None:
        with self._lock:
            self._cached = cached
            self._maybe_close_locked()

    def _maybe_close_locked(self) -> None:
        if self._cached or self._leases != 0 or self.mapping.closed:
            return
        try:
            self.mapping.close()
        except BufferError:
            # Exported NumPy/memoryview objects still own the mmap buffer. Leaving the
            # mmap alive is safe; Python will release it when those exports disappear.
            pass


class SharedObject:
    def __init__(
        self,
        client: "Client",
        info: dict[str, Any],
        token: str,
        mode: str,
        state: _MappingState,
        *,
        cache_hit: bool = False,
    ) -> None:
        self._client = client
        self._info = info
        self.token = token
        self.mode = mode
        self._state = state
        self._cache_hit = cache_hit
        self._released = False
        self._state.acquire()

    @property
    def object_id(self) -> str:
        return self._info["object_id"]

    @property
    def size(self) -> int:
        return int(self._info["size"])

    @property
    def shape(self) -> tuple[int, ...] | None:
        value = self._info.get("shape")
        return tuple(value) if value is not None else None

    @property
    def dtype(self) -> np.dtype[Any] | None:
        value = self._info.get("dtype")
        return np.dtype(value) if value is not None else None

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._info.get("metadata") or {})

    @property
    def generation(self) -> int:
        return int(self._info["generation"])

    @property
    def closed(self) -> bool:
        return self._released or self._state.mapping.closed

    @property
    def cache_hit(self) -> bool:
        """True when this lease came entirely from the process-local mapping cache."""
        return self._cache_hit

    def _require_open(self) -> mmap.mmap:
        if self._released or self._state.mapping.closed:
            raise ValueError("shared object is closed")
        return self._state.mapping

    def numpy(self) -> np.ndarray:
        if self.shape is None or self.dtype is None:
            raise TypeError("this object has no NumPy shape/dtype metadata")
        arr = np.ndarray(
            self.shape,
            dtype=self.dtype,
            buffer=self._require_open(),
            order=self._info.get("order", "C"),
        )
        if self.mode == "read":
            arr.flags.writeable = False
        return arr

    def buffer(self) -> memoryview:
        view = memoryview(self._require_open())
        if self.mode == "read" and not view.readonly:
            view = view.toreadonly()
        return view

    def issue(
        self,
        permissions: str | Iterable[str] = "read",
        *,
        expires_in: float | None = None,
    ) -> str:
        return self._client.issue_token(
            self.object_id,
            self.token,
            permissions,
            expires_in=expires_in,
        )

    def revoke(self, target_token: str) -> None:
        self._client.revoke_token(self.object_id, self.token, target_token)

    def info(self) -> dict[str, Any]:
        return self._client.info(self.object_id, self.token)

    def delete(self) -> None:
        self._client.delete(self.object_id, self.token)

    def close(self) -> None:
        if self._released:
            return
        self._released = True
        self._state.release()

    def __enter__(self) -> "SharedObject":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class Client:
    """Persistent mstore control client with optional LRU mapping cache.

    ``open(..., cache=True)`` keeps the OS mapping attached after the returned
    SharedObject is closed, allowing later opens with the same object/token/mode to
    avoid both control IPC and mmap attachment. This intentionally extends an already
    opened capability lease; token revocation/expiry only prevents *future* mappings.
    Call ``clear_cache()`` when that lease should be dropped.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        *,
        timeout: float = 10.0,
        cache_size: int = 64,
    ) -> None:
        if cache_size < 0:
            raise ValueError("cache_size must be >= 0")
        self.endpoint = endpoint or default_endpoint()
        self.timeout = timeout
        self.cache_size = cache_size
        self._conn: ControlConnection | None = None
        self._request_lock = threading.RLock()
        self._cache_lock = threading.RLock()
        # The raw capability token is intentionally used only as an in-process dict key.
        # Python's string dict lookup is enough here and avoids computing SHA-256 on a
        # cache hit. The daemon still stores only SHA-256 token hashes.
        self._cache: OrderedDict[tuple[str, str, str], _MappingState] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_evictions = 0
        self._pid = os.getpid()
        self._closed = False

    def _after_fork_if_needed(self) -> None:
        pid = os.getpid()
        if pid == self._pid:
            return
        # Never share a request/response stream across fork boundaries. Closing the
        # child's inherited descriptor does not affect the parent's descriptor.
        inherited = self._conn
        self._conn = None
        if inherited is not None:
            inherited.close()
        for state in self._cache.values():
            state.set_cached(False)
        self._request_lock = threading.RLock()
        self._cache_lock = threading.RLock()
        self._cache = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_evictions = 0
        self._pid = pid
        self._closed = False

    def _ensure_connection(self) -> ControlConnection:
        self._after_fork_if_needed()
        if self._closed:
            raise RuntimeError("client is closed")
        if self._conn is None:
            self._conn = connect_control(self.endpoint, self.timeout)
        return self._conn

    def _drop_connection(self) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:
            conn.close()

    def _request(self, op: str, **args: Any) -> tuple[dict[str, Any], int | None]:
        # A single persistent connection is a strict request/response stream. Serialize
        # access so one Client can safely be shared by multiple application threads.
        with self._request_lock:
            conn = self._ensure_connection()
            try:
                conn.send({"op": op, "args": args})
                response, fd = conn.recv()
            except (EOFError, OSError, ConnectionError, ProtocolError) as exc:
                self._drop_connection()
                raise ProtocolError(f"mstore control connection failed during {op!r}") from exc

        if not response.get("ok"):
            if fd is not None:
                os.close(fd)
            error = response.get("error") or {}
            kind = error.get("type", "internal_error")
            message = error.get("message", kind)
            cls = ERROR_TYPES.get(kind, MStoreError)
            raise cls(message)
        result = response.get("result")
        if not isinstance(result, dict):
            if fd is not None:
                os.close(fd)
            raise ProtocolError("server response is missing a result object")
        return result, fd

    @staticmethod
    def _open_mapping(mapping: dict[str, Any], fd: int | None, mode: str) -> mmap.mmap:
        backend = mapping.get("backend")
        size = int(mapping["size"])
        access = mmap.ACCESS_READ if mode == "read" else mmap.ACCESS_WRITE
        if backend == "memfd":
            if fd is None:
                raise ProtocolError("memfd response did not include a file descriptor")
            try:
                return mmap.mmap(fd, size, access=access)
            finally:
                os.close(fd)
        if backend == "windows_named":
            if os.name != "nt":
                raise ProtocolError("received a Windows mapping on a non-Windows client")
            name = mapping.get("name")
            if not name:
                raise ProtocolError("Windows mapping response is missing its name")
            return mmap.mmap(-1, size, tagname=name, access=access)
        if fd is not None:
            os.close(fd)
        raise ProtocolError(f"unsupported mapping backend: {backend!r}")

    def _cache_get(self, key: tuple[str, str, str]) -> _MappingState | None:
        with self._cache_lock:
            state = self._cache.get(key)
            if state is None:
                self._cache_misses += 1
                return None
            if state.mapping.closed:
                self._cache.pop(key, None)
                state.set_cached(False)
                self._cache_misses += 1
                return None
            self._cache.move_to_end(key)
            self._cache_hits += 1
            return state

    def _cache_put(self, key: tuple[str, str, str], state: _MappingState) -> None:
        if self.cache_size == 0:
            return
        with self._cache_lock:
            old = self._cache.pop(key, None)
            if old is not None and old is not state:
                old.set_cached(False)
            state.set_cached(True)
            self._cache[key] = state
            while len(self._cache) > self.cache_size:
                _, evicted = self._cache.popitem(last=False)
                self._cache_evictions += 1
                evicted.set_cached(False)

    def clear_cache(
        self,
        object_id: str | None = None,
        *,
        token: str | None = None,
        mode: str | None = None,
    ) -> int:
        """Drop cached mappings matching the supplied filters and return the count."""
        if mode is not None and mode not in {"read", "write"}:
            raise ValueError("mode must be 'read' or 'write'")
        with self._cache_lock:
            keys = [
                key
                for key in self._cache
                if (object_id is None or key[0] == object_id)
                and (token is None or key[1] == token)
                and (mode is None or key[2] == mode)
            ]
            states = [self._cache.pop(key) for key in keys]
            for state in states:
                state.set_cached(False)
            return len(states)

    def cache_info(self) -> dict[str, int]:
        """Return process-local mapping-cache counters."""
        with self._cache_lock:
            return {
                "size": len(self._cache),
                "capacity": self.cache_size,
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "evictions": self._cache_evictions,
            }

    def ping(self) -> dict[str, Any]:
        result, fd = self._request("ping")
        if fd is not None:
            os.close(fd)
        return result

    def create(
        self,
        *,
        size: int | None = None,
        shape: Iterable[int] | None = None,
        dtype: Any | None = None,
        order: str = "C",
        metadata: dict[str, Any] | None = None,
    ) -> SharedObject:
        shape_list: list[int] | None = None
        dtype_string: str | None = None
        if shape is not None or dtype is not None:
            if shape is None or dtype is None:
                raise ValueError("shape and dtype must be provided together")
            shape_list = [int(x) for x in shape]
            if not shape_list or any(x <= 0 for x in shape_list):
                raise ValueError("shape dimensions must be positive")
            dt = np.dtype(dtype)
            dtype_string = dt.str
            computed_size = int(np.prod(shape_list, dtype=np.int64)) * dt.itemsize
            if size is not None and int(size) != computed_size:
                raise ValueError("size does not match shape * dtype.itemsize")
            size = computed_size
        if size is None or int(size) <= 0:
            raise ValueError("size must be > 0")
        if order not in {"C", "F"}:
            raise ValueError("order must be 'C' or 'F'")
        result, fd = self._request(
            "create",
            size=int(size),
            shape=shape_list,
            dtype=dtype_string,
            order=order,
            metadata=metadata or {},
        )
        token = str(result["token"])
        mapping = self._open_mapping(result["mapping"], fd, "write")
        state = _MappingState(mapping, result["object"])
        return SharedObject(self, result["object"], token, "write", state)

    def open(
        self,
        object_id: str,
        token: str,
        *,
        mode: str = "read",
        cache: bool = False,
    ) -> SharedObject:
        if mode not in {"read", "write"}:
            raise ValueError("mode must be 'read' or 'write'")
        # IMPORTANT: the cache lookup happens before any control-plane work.
        # On a hit this path performs no Named Pipe/socket round-trip, no daemon-side
        # SHA-256/registry lock, and no OpenFileMapping/mmap attachment.
        key = (object_id, token, mode)
        if cache:
            state = self._cache_get(key)
            if state is not None:
                return SharedObject(
                    self, state.info, token, mode, state, cache_hit=True
                )

        result, fd = self._request("open", object_id=object_id, token=token, mode=mode)
        mapping = self._open_mapping(result["mapping"], fd, mode)
        state = _MappingState(mapping, result["object"])
        if cache:
            self._cache_put(key, state)
        return SharedObject(
            self, result["object"], token, mode, state, cache_hit=False
        )

    def issue_token(
        self,
        object_id: str,
        issuer_token: str,
        permissions: str | Iterable[str],
        *,
        expires_in: float | None = None,
    ) -> str:
        if not isinstance(permissions, str):
            permissions = list(permissions)
        result, fd = self._request(
            "grant",
            object_id=object_id,
            token=issuer_token,
            permissions=permissions,
            expires_in=expires_in,
        )
        if fd is not None:
            os.close(fd)
        return str(result["token"])

    def revoke_token(self, object_id: str, issuer_token: str, target_token: str) -> None:
        result, fd = self._request(
            "revoke",
            object_id=object_id,
            token=issuer_token,
            target_token=target_token,
        )
        if fd is not None:
            os.close(fd)
        if not result.get("revoked"):
            raise ProtocolError("server did not confirm token revocation")

    def info(self, object_id: str, token: str) -> dict[str, Any]:
        result, fd = self._request("info", object_id=object_id, token=token)
        if fd is not None:
            os.close(fd)
        return result

    def delete(self, object_id: str, token: str) -> None:
        result, fd = self._request("delete", object_id=object_id, token=token)
        if fd is not None:
            os.close(fd)
        if not result.get("deleted"):
            raise ProtocolError("server did not confirm object deletion")
        self.clear_cache(object_id)

    def close(self) -> None:
        if self._closed:
            return
        self.clear_cache()
        with self._request_lock:
            self._drop_connection()
            self._closed = True

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def connect(
    endpoint: str | None = None,
    *,
    timeout: float = 10.0,
    cache_size: int = 64,
) -> Client:
    return Client(endpoint, timeout=timeout, cache_size=cache_size)
