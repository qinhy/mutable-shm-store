from __future__ import annotations

import getpass
import os
import re
import socket
from multiprocessing.connection import Client as PipeClient
from multiprocessing.connection import Connection, Listener as PipeListener
from pathlib import Path
from typing import Any, Protocol

from .errors import ProtocolError
from .protocol import (
    recv_frame_with_optional_fd,
    recv_pipe_frame,
    send_frame,
    send_pipe_frame,
)


class ControlConnection(Protocol):
    def send(self, message: dict[str, Any], fd: int | None = None) -> None: ...
    def recv(self) -> tuple[dict[str, Any], int | None]: ...
    def close(self) -> None: ...


class SocketControlConnection:
    def __init__(self, sock: socket.socket) -> None:
        self.sock = sock

    def send(self, message: dict[str, Any], fd: int | None = None) -> None:
        send_frame(self.sock, message, fd=fd)

    def recv(self) -> tuple[dict[str, Any], int | None]:
        return recv_frame_with_optional_fd(self.sock)

    def close(self) -> None:
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass


class PipeControlConnection:
    def __init__(self, conn: Connection) -> None:
        self.conn = conn

    def send(self, message: dict[str, Any], fd: int | None = None) -> None:
        if fd is not None:
            raise ProtocolError("file descriptors cannot be passed over a Windows named pipe")
        send_pipe_frame(self.conn, message)

    def recv(self) -> tuple[dict[str, Any], int | None]:
        return recv_pipe_frame(self.conn), None

    def close(self) -> None:
        try:
            self.conn.close()
        except OSError:
            pass


def _pipe_name() -> str:
    user = re.sub(r"[^A-Za-z0-9_.-]+", "-", getpass.getuser()).strip("-") or "user"
    return f"mstore-{user}"


def default_endpoint() -> str:
    if os.name == "nt":
        return f"pipe://{_pipe_name()}"
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
    if endpoint.startswith("pipe://"):
        name = endpoint[len("pipe://") :].strip()
        if not name:
            raise ValueError("pipe endpoint requires a name")
        if name.startswith("\\\\.\\pipe\\"):
            address = name
        else:
            address = "\\\\.\\pipe\\" + name
        return "pipe", address
    raise ValueError("endpoint must start with unix://, tcp://, or pipe://")


def connect_control(endpoint: str, timeout: float) -> ControlConnection:
    kind, address = parse_endpoint(endpoint)
    if kind == "pipe":
        if os.name != "nt":
            raise RuntimeError("pipe:// endpoints are supported only on Windows")
        # AF_PIPE uses the Windows named-pipe implementation in the stdlib.
        return PipeControlConnection(PipeClient(address, family="AF_PIPE"))

    if kind == "unix":
        if os.name == "nt":
            raise RuntimeError("unix:// endpoints are not supported on Windows")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    else:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.settimeout(timeout)
    try:
        sock.connect(address)
    except Exception:
        sock.close()
        raise
    # The connection remains persistent; timeout continues to bound each blocking IO.
    return SocketControlConnection(sock)


def make_socket_listener(endpoint: str) -> tuple[socket.socket, str]:
    kind, address = parse_endpoint(endpoint)
    if kind == "pipe":
        raise ValueError("pipe endpoint is not a socket endpoint")
    if kind == "unix":
        if os.name == "nt":
            raise RuntimeError("unix:// endpoints are not supported on Windows")
        path = Path(address)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(str(path))
        os.chmod(path, 0o600)
        published = endpoint
    else:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.bind(address)
        host, port = sock.getsockname()[:2]
        published = f"tcp://{host}:{port}"
    sock.listen(128)
    sock.settimeout(0.5)
    return sock, published


def make_pipe_listener(endpoint: str) -> PipeListener:
    kind, address = parse_endpoint(endpoint)
    if kind != "pipe":
        raise ValueError("endpoint is not a pipe endpoint")
    if os.name != "nt":
        raise RuntimeError("pipe:// endpoints are supported only on Windows")
    return PipeListener(address=address, family="AF_PIPE", backlog=128, authkey=None)
