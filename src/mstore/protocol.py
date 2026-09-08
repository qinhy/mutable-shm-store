from __future__ import annotations

import array
import json
import socket
import struct
from typing import Any

from .errors import ProtocolError

MAX_FRAME = 4 * 1024 * 1024
_HEADER = struct.Struct("!I")


def encode_frame(message: dict[str, Any]) -> bytes:
    payload = json.dumps(message, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if len(payload) > MAX_FRAME:
        raise ProtocolError(f"control frame exceeds {MAX_FRAME} bytes")
    return _HEADER.pack(len(payload)) + payload


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ProtocolError("connection closed while receiving a control frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_frame(sock: socket.socket) -> dict[str, Any]:
    header = recv_exact(sock, _HEADER.size)
    (size,) = _HEADER.unpack(header)
    if size > MAX_FRAME:
        raise ProtocolError(f"control frame exceeds {MAX_FRAME} bytes")
    payload = recv_exact(sock, size)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("invalid JSON control frame") from exc
    if not isinstance(value, dict):
        raise ProtocolError("control frame must be a JSON object")
    return value


def send_frame(sock: socket.socket, message: dict[str, Any], fd: int | None = None) -> None:
    frame = encode_frame(message)
    if fd is None:
        sock.sendall(frame)
        return
    if not hasattr(sock, "sendmsg"):
        raise ProtocolError("file-descriptor passing is unavailable on this platform")
    rights = array.array("i", [fd])
    sent = sock.sendmsg(
        [frame],
        [(socket.SOL_SOCKET, socket.SCM_RIGHTS, rights.tobytes())],
    )
    if sent < len(frame):
        sock.sendall(frame[sent:])


def recv_frame_with_optional_fd(sock: socket.socket) -> tuple[dict[str, Any], int | None]:
    if not hasattr(sock, "recvmsg"):
        return recv_frame(sock), None

    ancbuf = socket.CMSG_SPACE(array.array("i").itemsize)
    first, ancdata, _flags, _addr = sock.recvmsg(64 * 1024, ancbuf)
    if not first:
        raise ProtocolError("connection closed while receiving a control frame")

    fd: int | None = None
    for level, kind, data in ancdata:
        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
            ints = array.array("i")
            usable = len(data) - (len(data) % ints.itemsize)
            ints.frombytes(data[:usable])
            if ints:
                fd = ints[0]
                for extra in ints[1:]:
                    try:
                        import os

                        os.close(extra)
                    except OSError:
                        pass
                break

    buf = bytearray(first)
    while len(buf) < _HEADER.size:
        buf.extend(recv_exact(sock, _HEADER.size - len(buf)))
    (size,) = _HEADER.unpack(buf[: _HEADER.size])
    if size > MAX_FRAME:
        raise ProtocolError(f"control frame exceeds {MAX_FRAME} bytes")
    total = _HEADER.size + size
    if len(buf) < total:
        buf.extend(recv_exact(sock, total - len(buf)))
    payload = bytes(buf[_HEADER.size:total])
    try:
        message = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("invalid JSON control frame") from exc
    if not isinstance(message, dict):
        raise ProtocolError("control frame must be a JSON object")
    return message, fd
