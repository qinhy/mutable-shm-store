from __future__ import annotations

import mmap
import os
import socket
from typing import Any, Iterable

import numpy as np

from .errors import ERROR_TYPES, MStoreError, ProtocolError
from .protocol import recv_frame_with_optional_fd, send_frame
from .server import default_endpoint, parse_endpoint


class SharedObject:
    def __init__(
        self,
        client: "Client",
        info: dict[str, Any],
        token: str,
        mode: str,
        mapping: mmap.mmap,
    ) -> None:
        self._client = client
        self._info = info
        self.token = token
        self.mode = mode
        self._mapping = mapping
        self._close_requested = False

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
        return self._mapping.closed

    def numpy(self) -> np.ndarray:
        if self.shape is None or self.dtype is None:
            raise TypeError("this object has no NumPy shape/dtype metadata")
        arr = np.ndarray(
            self.shape,
            dtype=self.dtype,
            buffer=self._mapping,
            order=self._info.get("order", "C"),
        )
        if self.mode == "read":
            arr.flags.writeable = False
        return arr

    def buffer(self) -> memoryview:
        view = memoryview(self._mapping)
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
        if self._mapping.closed:
            return
        self._close_requested = True
        try:
            self._mapping.close()
        except BufferError:
            # NumPy/memoryview exports may still point at the mapping. Keeping the mmap
            # alive is safer than invalidating live arrays; it will close once views die.
            pass

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
    def __init__(self, endpoint: str | None = None, *, timeout: float = 10.0) -> None:
        self.endpoint = endpoint or default_endpoint()
        self.timeout = timeout

    def _connect(self) -> socket.socket:
        kind, address = parse_endpoint(self.endpoint)
        if kind == "unix":
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        else:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(address)
        return sock

    def _request(self, op: str, **args: Any) -> tuple[dict[str, Any], int | None]:
        sock = self._connect()
        try:
            send_frame(sock, {"op": op, "args": args})
            response, fd = recv_frame_with_optional_fd(sock)
        finally:
            sock.close()
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
        token = result["token"]
        mapping = self._open_mapping(result["mapping"], fd, "write")
        return SharedObject(self, result["object"], token, "write", mapping)

    def open(self, object_id: str, token: str, *, mode: str = "read") -> SharedObject:
        if mode not in {"read", "write"}:
            raise ValueError("mode must be 'read' or 'write'")
        result, fd = self._request(
            "open", object_id=object_id, token=token, mode=mode
        )
        mapping = self._open_mapping(result["mapping"], fd, mode)
        return SharedObject(self, result["object"], token, mode, mapping)

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


def connect(endpoint: str | None = None, *, timeout: float = 10.0) -> Client:
    return Client(endpoint, timeout=timeout)
