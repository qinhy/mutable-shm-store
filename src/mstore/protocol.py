from __future__ import annotations

import array
import json
import os
import socket
import struct
from multiprocessing.connection import Connection
from typing import Any

from .errors import ProtocolError

MAX_FRAME = 4 * 1024 * 1024
_HEADER = struct.Struct("!I")


def encode_frame(message: dict[str, Any]) -> bytes:
    payload = json.dumps(message, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if len(payload) > MAX_FRAME:
        raise ProtocolError(f"control frame exceeds {MAX_FRAME} bytes")
    return _HEADER.pack(len(payload)) + payload


def decode_frame(frame: bytes) -> dict[str, Any]:
    if len(frame) < _HEADER.size:
        raise ProtocolError("control frame is missing its length header")
    (size,) = _HEADER.unpack(frame[: _HEADER.size])
    if size > MAX_FRAME:
        raise ProtocolError(f"control frame exceeds {MAX_FRAME} bytes")
    if len(frame) != _HEADER.size + size:
        raise ProtocolError("control frame length does not match its header")
    payload = frame[_HEADER.size :]
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("invalid JSON control frame") from exc
    if not isinstance(value, dict):
        raise ProtocolError("control frame must be a JSON object")
    return value


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("connection closed while receiving a control frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_frame(sock: socket.socket) -> dict[str, Any]:
    header = recv_exact(sock, _HEADER.size)
    (size,) = _HEADER.unpack(header)
    if size > MAX_FRAME:
        raise ProtocolError(f"control frame exceeds {MAX_FRAME} bytes")
    return decode_frame(header + recv_exact(sock, size))


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
        raise EOFError("connection closed while receiving a control frame")

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
                        os.close(extra)
                    except OSError:
                        pass
                break

    buf = bytearray(first)
    while len(buf) < _HEADER.size:
        buf.extend(recv_exact(sock, _HEADER.size - len(buf)))
    (size,) = _HEADER.unpack(buf[: _HEADER.size])
    if size > MAX_FRAME:
        if fd is not None:
            os.close(fd)
        raise ProtocolError(f"control frame exceeds {MAX_FRAME} bytes")
    total = _HEADER.size + size
    if len(buf) < total:
        buf.extend(recv_exact(sock, total - len(buf)))
    if len(buf) != total:
        # Synchronous request/response means this should never happen. Refuse to
        # silently discard bytes because that would desynchronize a persistent stream.
        if fd is not None:
            os.close(fd)
        raise ProtocolError("received bytes beyond the end of a control frame")
    try:
        return decode_frame(bytes(buf)), fd
    except Exception:
        if fd is not None:
            os.close(fd)
        raise


def send_pipe_frame(conn: Connection, message: dict[str, Any]) -> None:
    conn.send_bytes(encode_frame(message))


def recv_pipe_frame(conn: Connection) -> dict[str, Any]:
    # recv_bytes enforces the maximum before returning the message. Broken-pipe and
    # oversized-message OSErrors are handled by the transport/session layer.
    frame = conn.recv_bytes(MAX_FRAME + _HEADER.size)
    return decode_frame(frame)
