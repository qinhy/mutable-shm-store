from __future__ import annotations

import ctypes
import mmap
import os
import secrets
import sys
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
            # dup() would preserve O_RDWR. Reopening through procfs produces a real
            # O_RDONLY descriptor, so the receiving process cannot remap it writable.
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
class MacOSSharedRegion:
    fd: int
    read_fd: int
    size: int

    @classmethod
    def create(cls, size: int, object_id: str) -> "MacOSSharedRegion":
        libc = ctypes.CDLL(None, use_errno=True)
        # shm_open is variadic; mode is passed as a promoted C int.
        libc.shm_open.argtypes = [ctypes.c_char_p, ctypes.c_int]
        libc.shm_open.restype = ctypes.c_int
        libc.shm_unlink.argtypes = [ctypes.c_char_p]
        libc.shm_unlink.restype = ctypes.c_int
        name = ("/mstore-" + secrets.token_hex(10)).encode("ascii")

        def check(result: int) -> int:
            if result < 0:
                error = ctypes.get_errno()
                raise OSError(error, os.strerror(error))
            return result

        fd = check(libc.shm_open(name, os.O_CREAT | os.O_EXCL | os.O_RDWR, ctypes.c_int(0o600)))
        read_fd = -1
        linked = True
        try:
            os.ftruncate(fd, size)
            # Keep a separate O_RDONLY handle before removing the name: dup of
            # the writable handle would let readers create writable mappings.
            read_fd = check(libc.shm_open(name, os.O_RDONLY, ctypes.c_int(0)))
            check(libc.shm_unlink(name))
            linked = False
            os.set_inheritable(fd, False)
            os.set_inheritable(read_fd, False)
            return cls(fd=fd, read_fd=read_fd, size=size)
        except BaseException:
            os.close(fd)
            if read_fd >= 0:
                os.close(read_fd)
            raise
        finally:
            if linked:
                check(libc.shm_unlink(name))

    def client_mapping(self, mode: str) -> tuple[dict[str, Any], int]:
        if mode not in {"read", "write"}:
            raise ValueError(f"unsupported mapping mode: {mode}")
        fd = os.dup(self.read_fd if mode == "read" else self.fd)
        return {"backend": "posix_shm", "size": self.size}, fd

    def close(self) -> None:
        for attribute in ("fd", "read_fd"):
            fd = getattr(self, attribute)
            if fd >= 0:
                os.close(fd)
                setattr(self, attribute, -1)


@dataclass
class WindowsNamedRegion:
    mapping: mmap.mmap
    size: int
    name: str

    @classmethod
    def create(cls, size: int, object_id: str) -> "WindowsNamedRegion":
        # The daemon keeps this mapping open so its named kernel object survives
        # producer/client exits. Random suffixes prevent collisions with stale views.
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
    if sys.platform == "darwin":
        return MacOSSharedRegion.create(size, object_id)
    raise RuntimeError("mstore supports Linux, macOS, and Windows")
