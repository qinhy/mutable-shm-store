from __future__ import annotations

import mmap
import os
import secrets
from dataclasses import dataclass
from typing import Any, Protocol


class Region(Protocol):
    size: int

    def client_mapping(self, mode: str) -> tuple[dict[str, Any], int | None]: ...

    def close(self) -> None: ...


@dataclass
class LinuxMemfdRegion:
    fd: int
    size: int
    name: str

    @classmethod
    def create(cls, size: int, object_id: str) -> "LinuxMemfdRegion":
        if not hasattr(os, "memfd_create"):
            raise RuntimeError("Linux memfd_create() is required by the Linux backend")
        flags = getattr(os, "MFD_CLOEXEC", 0)
        fd = os.memfd_create(f"mstore-{object_id}", flags=flags)
        try:
            os.ftruncate(fd, size)
        except Exception:
            os.close(fd)
            raise
        return cls(fd=fd, size=size, name=f"mstore-{object_id}")

    def client_mapping(self, mode: str) -> tuple[dict[str, Any], int]:
        if mode == "read":
            # Re-open through procfs to obtain a genuinely O_RDONLY descriptor.
            # A dup() would retain O_RDWR and a hostile client could remap it writable.
            path = f"/proc/self/fd/{self.fd}"
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        elif mode == "write":
            fd = os.dup(self.fd)
        else:
            raise ValueError(f"unsupported mapping mode: {mode}")
        return {"backend": "memfd", "size": self.size}, fd

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


@dataclass
class WindowsNamedRegion:
    mapping: mmap.mmap
    size: int
    name: str

    @classmethod
    def create(cls, size: int, object_id: str) -> "WindowsNamedRegion":
        # Keep a daemon-owned handle open so the named mapping survives client exits.
        # A random suffix avoids collisions if stale clients keep an old mapping alive.
        name = f"mstore-{object_id}-{secrets.token_hex(8)}"
        mapping = mmap.mmap(-1, size, tagname=name, access=mmap.ACCESS_WRITE)
        return cls(mapping=mapping, size=size, name=name)

    def client_mapping(self, mode: str) -> tuple[dict[str, Any], None]:
        if mode not in {"read", "write"}:
            raise ValueError(f"unsupported mapping mode: {mode}")
        return {
            "backend": "windows_named",
            "size": self.size,
            "name": self.name,
            "mode": mode,
        }, None

    def close(self) -> None:
        self.mapping.close()


def create_region(size: int, object_id: str) -> Region:
    if os.name == "nt":
        return WindowsNamedRegion.create(size, object_id)
    if os.name == "posix" and hasattr(os, "memfd_create"):
        return LinuxMemfdRegion.create(size, object_id)
    raise RuntimeError("mstore v0.1 supports Linux (memfd) and Windows (named mmap)")
