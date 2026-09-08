from __future__ import annotations

import os
import socket
import threading
import traceback
from pathlib import Path
from typing import Any

from .errors import (
    AuthenticationError,
    InvalidRequest,
    MStoreError,
    ObjectNotFound,
    PermissionDenied,
    ProtocolError,
    TokenExpired,
    TokenRevoked,
)
from .protocol import recv_frame, send_frame
from .registry import DELETE, INFO, READ, WRITE, Registry

ERROR_CODES = {
    AuthenticationError: "authentication_error",
    PermissionDenied: "permission_denied",
    ObjectNotFound: "object_not_found",
    TokenExpired: "token_expired",
    TokenRevoked: "token_revoked",
    InvalidRequest: "invalid_request",
    ProtocolError: "protocol_error",
}


def default_endpoint() -> str:
    if os.name == "nt":
        return "tcp://127.0.0.1:65432"
    runtime = os.environ.get("XDG_RUNTIME_DIR") or "/tmp"
    return f"unix://{runtime}/mstore-{os.getuid()}.sock"


def parse_endpoint(endpoint: str) -> tuple[str, Any]:
    if endpoint.startswith("unix://"):
        return "unix", endpoint[len("unix://") :]
    if endpoint.startswith("tcp://"):
        hostport = endpoint[len("tcp://") :]
        host, sep, port = hostport.rpartition(":")
        if not sep or not host or not port:
            raise ValueError(f"invalid TCP endpoint: {endpoint}")
        return "tcp", (host, int(port))
    raise ValueError("endpoint must start with unix:// or tcp://")


class MStoreServer:
    def __init__(self, endpoint: str | None = None, *, debug: bool = False) -> None:
        self.endpoint = endpoint or default_endpoint()
        self.debug = debug
        self.registry = Registry()
        self._listener: socket.socket | None = None
        self._stop = threading.Event()
        self._threads: set[threading.Thread] = set()
        self._threads_lock = threading.Lock()

    def _make_listener(self) -> socket.socket:
        kind, address = parse_endpoint(self.endpoint)
        if kind == "unix":
            if os.name == "nt":
                raise RuntimeError("unix:// endpoints are not supported by mstore on Windows")
            path = Path(address)
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.bind(str(path))
            os.chmod(path, 0o600)
        else:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(address)
            # If port 0 was requested, publish the actual bound endpoint.
            host, port = sock.getsockname()[:2]
            self.endpoint = f"tcp://{host}:{port}"
        sock.listen(128)
        sock.settimeout(0.5)
        return sock

    def serve_forever(self) -> None:
        self._listener = self._make_listener()
        try:
            while not self._stop.is_set():
                try:
                    conn, _addr = self._listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self._stop.is_set():
                        break
                    raise
                thread = threading.Thread(target=self._serve_connection, args=(conn,), daemon=True)
                with self._threads_lock:
                    self._threads.add(thread)
                thread.start()
        finally:
            self._cleanup()

    def start_in_thread(self) -> threading.Thread:
        thread = threading.Thread(target=self.serve_forever, name="mstore-server", daemon=True)
        thread.start()
        # Wait until the listener exists without hardcoding a sleep.
        for _ in range(1000):
            if self._listener is not None:
                break
            self._stop.wait(0.001)
        return thread

    def shutdown(self) -> None:
        self._stop.set()
        if self._listener is not None:
            try:
                self._listener.close()
            except OSError:
                pass

    def _cleanup(self) -> None:
        listener = self._listener
        self._listener = None
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass
        self.registry.close()
        kind, address = parse_endpoint(self.endpoint)
        if kind == "unix":
            try:
                Path(address).unlink()
            except FileNotFoundError:
                pass
        current = threading.current_thread()
        with self._threads_lock:
            threads = [t for t in self._threads if t is not current]
        for thread in threads:
            thread.join(timeout=1.0)

    def _serve_connection(self, conn: socket.socket) -> None:
        fd_to_close: int | None = None
        try:
            request = recv_frame(conn)
            result, fd_to_close = self._handle(request)
            send_frame(conn, {"ok": True, "result": result}, fd=fd_to_close)
        except Exception as exc:
            error_type = next(
                (code for cls, code in ERROR_CODES.items() if isinstance(exc, cls)),
                "internal_error",
            )
            payload: dict[str, Any] = {
                "ok": False,
                "error": {"type": error_type, "message": str(exc)},
            }
            if self.debug and not isinstance(exc, MStoreError):
                payload["error"]["traceback"] = traceback.format_exc()
            try:
                send_frame(conn, payload)
            except OSError:
                pass
        finally:
            if fd_to_close is not None:
                try:
                    os.close(fd_to_close)
                except OSError:
                    pass
            try:
                conn.close()
            except OSError:
                pass
            with self._threads_lock:
                self._threads.discard(threading.current_thread())

    def _handle(self, request: dict[str, Any]) -> tuple[dict[str, Any], int | None]:
        op = request.get("op")
        args = request.get("args") or {}
        if not isinstance(args, dict):
            raise InvalidRequest("args must be an object")

        if op == "ping":
            return {"pong": True, "endpoint": self.endpoint}, None

        if op == "create":
            obj, token = self.registry.create_object(
                size=int(args["size"]),
                shape=args.get("shape"),
                dtype=args.get("dtype"),
                order=args.get("order", "C"),
                metadata=args.get("metadata") or {},
            )
            mapping, fd = obj.region.client_mapping("write")
            return {"object": obj.public(), "token": token, "mapping": mapping}, fd

        object_id = str(args.get("object_id") or "")
        token = str(args.get("token") or "")
        if not object_id:
            raise InvalidRequest("object_id is required")

        if op == "open":
            mode = args.get("mode", "read")
            required = WRITE if mode == "write" else READ if mode == "read" else None
            if required is None:
                raise InvalidRequest("mode must be 'read' or 'write'")
            obj, _ = self.registry.validate(object_id, token, required)
            mapping, fd = obj.region.client_mapping(mode)
            return {"object": obj.public(), "mapping": mapping}, fd

        if op == "info":
            obj, tok = self.registry.validate(object_id, token, INFO)
            return {
                "object": obj.public(),
                "token": {
                    "permissions": sorted(tok.permissions),
                    "expires_at": tok.expires_at,
                },
            }, None

        if op == "grant":
            issued = self.registry.issue_token(
                object_id,
                token,
                args.get("permissions", "read"),
                args.get("expires_in"),
            )
            return {"token": issued}, None

        if op == "revoke":
            target = str(args.get("target_token") or "")
            if not target:
                raise InvalidRequest("target_token is required")
            self.registry.revoke_token(object_id, token, target)
            return {"revoked": True}, None

        if op == "delete":
            self.registry.validate(object_id, token, DELETE)
            self.registry.delete_object(object_id, token)
            return {"deleted": True}, None

        raise InvalidRequest(f"unknown operation: {op!r}")
