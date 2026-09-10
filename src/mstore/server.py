from __future__ import annotations

import os
import socket
import threading
import traceback
from multiprocessing.connection import Listener as PipeListener
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
from .registry import DELETE, INFO, READ, WRITE, Registry
from .transport import (
    ControlConnection,
    PipeControlConnection,
    SocketControlConnection,
    connect_control,
    default_endpoint,
    make_pipe_listener,
    make_socket_listener,
    parse_endpoint,
)

ERROR_CODES = {
    AuthenticationError: "authentication_error",
    PermissionDenied: "permission_denied",
    ObjectNotFound: "object_not_found",
    TokenExpired: "token_expired",
    TokenRevoked: "token_revoked",
    InvalidRequest: "invalid_request",
    ProtocolError: "protocol_error",
}


class MStoreServer:
    def __init__(self, endpoint: str | None = None, *, debug: bool = False) -> None:
        self.endpoint = endpoint or default_endpoint()
        kind, _ = parse_endpoint(self.endpoint)
        if kind == "tcp" and os.name != "nt":
            raise ValueError(
                "tcp:// control endpoints cannot carry Linux memfd descriptors; "
                "use unix:// on Linux"
            )
        self.debug = debug
        self.registry = Registry()
        self._socket_listener: socket.socket | None = None
        self._pipe_listener: PipeListener | None = None
        self._stop = threading.Event()
        self._threads: set[threading.Thread] = set()
        self._threads_lock = threading.Lock()
        self._connections: set[ControlConnection] = set()
        self._connections_lock = threading.Lock()

    def _register_connection(self, conn: ControlConnection) -> None:
        with self._connections_lock:
            self._connections.add(conn)

    def _unregister_connection(self, conn: ControlConnection) -> None:
        with self._connections_lock:
            self._connections.discard(conn)

    def _start_worker(self, conn: ControlConnection) -> None:
        self._register_connection(conn)
        thread = threading.Thread(target=self._serve_connection, args=(conn,), daemon=True)
        with self._threads_lock:
            self._threads.add(thread)
        thread.start()

    def serve_forever(self) -> None:
        kind, _address = parse_endpoint(self.endpoint)
        try:
            if kind == "pipe":
                self._serve_pipe_forever()
            else:
                self._serve_socket_forever()
        finally:
            self._cleanup()

    def _serve_socket_forever(self) -> None:
        listener, published = make_socket_listener(self.endpoint)
        self._socket_listener = listener
        self.endpoint = published
        while not self._stop.is_set():
            try:
                sock, _addr = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                if self._stop.is_set():
                    break
                raise
            # Avoid Nagle on localhost TCP. AF_UNIX ignores this branch.
            if sock.family in {socket.AF_INET, getattr(socket, "AF_INET6", -1)}:
                try:
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                except OSError:
                    pass
            self._start_worker(SocketControlConnection(sock))

    def _serve_pipe_forever(self) -> None:
        listener = make_pipe_listener(self.endpoint)
        self._pipe_listener = listener
        while not self._stop.is_set():
            try:
                raw = listener.accept()
            except (OSError, EOFError):
                if self._stop.is_set():
                    break
                raise
            conn = PipeControlConnection(raw)
            if self._stop.is_set():
                conn.close()
                break
            self._start_worker(conn)

    def start_in_thread(self) -> threading.Thread:
        thread = threading.Thread(target=self.serve_forever, name="mstore-server", daemon=True)
        thread.start()
        # Wait for either listener flavor to become live without an arbitrary long sleep.
        for _ in range(2000):
            if self._socket_listener is not None or self._pipe_listener is not None:
                break
            if not thread.is_alive():
                break
            self._stop.wait(0.001)
        return thread

    def shutdown(self) -> None:
        self._stop.set()

        listener = self._socket_listener
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass

        # multiprocessing.connection.Listener.accept() is blocking. A local wake-up
        # connection lets the accept loop observe _stop and exit deterministically.
        pipe_listener = self._pipe_listener
        if pipe_listener is not None:
            try:
                wake = connect_control(self.endpoint, 1.0)
                wake.close()
            except Exception:
                pass
            try:
                # The wake-up may let the server thread reach _cleanup() first,
                # which closes this same Listener and clears its internal handle.
                pipe_listener.close()
            except (AttributeError, OSError):
                pass

        with self._connections_lock:
            connections = list(self._connections)
        for conn in connections:
            conn.close()

    def _cleanup(self) -> None:
        listener, self._socket_listener = self._socket_listener, None
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass
        pipe_listener, self._pipe_listener = self._pipe_listener, None
        if pipe_listener is not None:
            try:
                pipe_listener.close()
            except OSError:
                pass

        with self._connections_lock:
            connections = list(self._connections)
        for conn in connections:
            conn.close()

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

    def _error_payload(self, exc: Exception) -> dict[str, Any]:
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
        return payload

    def _serve_connection(self, conn: ControlConnection) -> None:
        try:
            # Persistent connection: one worker handles many sequential requests from
            # the same Client instead of paying connect/accept/thread creation per op.
            while not self._stop.is_set():
                try:
                    request, request_fd = conn.recv()
                except EOFError:
                    break
                except (OSError, ConnectionError):
                    break
                except ProtocolError as exc:
                    # A malformed stream may be desynchronized (e.g. oversized frame),
                    # so report the error once and close the connection.
                    try:
                        conn.send(self._error_payload(exc))
                    except Exception:
                        pass
                    break

                if request_fd is not None:
                    # Clients never send descriptors to the server in this protocol.
                    try:
                        os.close(request_fd)
                    except OSError:
                        pass
                    try:
                        conn.send(self._error_payload(ProtocolError("unexpected client file descriptor")))
                    except Exception:
                        pass
                    break

                fd_to_close: int | None = None
                try:
                    result, fd_to_close = self._handle(request)
                    conn.send({"ok": True, "result": result}, fd=fd_to_close)
                except Exception as exc:
                    try:
                        conn.send(self._error_payload(exc))
                    except (OSError, EOFError, ConnectionError):
                        break
                finally:
                    if fd_to_close is not None:
                        try:
                            os.close(fd_to_close)
                        except OSError:
                            pass
        finally:
            conn.close()
            self._unregister_connection(conn)
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
            try:
                size = int(args["size"])
            except (KeyError, TypeError, ValueError) as exc:
                raise InvalidRequest("size is required and must be an integer") from exc
            obj, token = self.registry.create_object(
                size=size,
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
            self.registry.delete_object(object_id, token)
            return {"deleted": True}, None

        raise InvalidRequest(f"unknown operation: {op!r}")


__all__ = ["MStoreServer", "default_endpoint", "parse_endpoint"]
