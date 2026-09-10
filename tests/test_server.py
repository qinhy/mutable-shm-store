from __future__ import annotations

from mstore.server import MStoreServer


def test_pipe_shutdown_uses_stable_listener_reference(monkeypatch) -> None:
    class Listener:
        closed = False

        def close(self) -> None:
            self.closed = True

    class WakeConnection:
        def close(self) -> None:
            pass

    server = MStoreServer("pipe://shutdown-race-test")
    listener = Listener()
    server._pipe_listener = listener

    def wake_and_simulate_cleanup(_endpoint: str, _timeout: float) -> WakeConnection:
        # The server thread can clear the attribute as soon as accept() is woken.
        server._pipe_listener = None
        return WakeConnection()

    monkeypatch.setattr("mstore.server.connect_control", wake_and_simulate_cleanup)
    try:
        server.shutdown()
    finally:
        server.registry.close()

    assert listener.closed
