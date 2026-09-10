from __future__ import annotations

import array
import socket
import struct

import pytest

from mstore.errors import ProtocolError
from mstore.protocol import MAX_FRAME, decode_frame, encode_frame, recv_frame_with_optional_fd
from mstore.transport import parse_endpoint


def test_control_frame_round_trip() -> None:
    message = {"op": "ping", "args": {"label": "日本語"}}

    assert decode_frame(encode_frame(message)) == message


@pytest.mark.parametrize(
    ("frame", "match"),
    [
        (b"", "missing its length header"),
        (struct.pack("!I", 2) + b"{}x", "does not match"),
        (struct.pack("!I", MAX_FRAME + 1), "exceeds"),
        (struct.pack("!I", 1) + b"[", "invalid JSON"),
        (struct.pack("!I", 2) + b"[]", "must be a JSON object"),
    ],
)
def test_decode_frame_rejects_malformed_input(frame: bytes, match: str) -> None:
    with pytest.raises(ProtocolError, match=match):
        decode_frame(frame)


def test_parse_supported_endpoints() -> None:
    assert parse_endpoint("unix:///tmp/mstore.sock") == ("unix", "/tmp/mstore.sock")
    assert parse_endpoint("tcp://127.0.0.1:65432") == ("tcp", ("127.0.0.1", 65432))
    assert parse_endpoint("pipe://example") == ("pipe", r"\\.\pipe\example")


@pytest.mark.parametrize("endpoint", ["http://localhost", "pipe://", "tcp://localhost"])
def test_parse_endpoint_rejects_invalid_values(endpoint: str) -> None:
    with pytest.raises(ValueError):
        parse_endpoint(endpoint)


@pytest.mark.skipif(
    not hasattr(socket, "CMSG_SPACE"),
    reason="SCM_RIGHTS descriptor passing is POSIX-only",
)
def test_received_fd_is_closed_when_rest_of_frame_is_missing(monkeypatch) -> None:
    class PartialFrameSocket:
        def recvmsg(self, _size, _ancbuf):
            rights = array.array("i", [123]).tobytes()
            return b"\x00", [(socket.SOL_SOCKET, socket.SCM_RIGHTS, rights)], 0, None

        def recv(self, _size):
            return b""

    closed: list[int] = []
    monkeypatch.setattr("mstore.protocol.os.close", closed.append)

    with pytest.raises(EOFError):
        recv_frame_with_optional_fd(PartialFrameSocket())

    assert closed == [123]
