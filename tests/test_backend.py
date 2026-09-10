from __future__ import annotations

import mmap
import os
import sys

import pytest

from mstore.backend import MacOSSharedRegion, create_region

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="macOS shared memory backend")


def test_read_descriptor_rejects_writable_mapping_and_survives_close():
    region = create_region(4096, "test")
    assert isinstance(region, MacOSSharedRegion)
    _, read_fd = region.client_mapping("read")
    _, write_fd = region.client_mapping("write")
    try:
        with mmap.mmap(write_fd, 4096, access=mmap.ACCESS_WRITE) as writer:
            with mmap.mmap(read_fd, 4096, access=mmap.ACCESS_READ) as reader:
                with pytest.raises(OSError):
                    mmap.mmap(read_fd, 4096, access=mmap.ACCESS_WRITE)
                region.close()
                writer[:4] = b"test"
                assert reader[:4] == b"test"
    finally:
        os.close(read_fd)
        os.close(write_fd)
        region.close()


def test_allocation_failure_closes_descriptor(monkeypatch):
    opened = []

    def fail_truncate(fd, size):
        opened.append(fd)
        raise OSError("allocation failed")

    monkeypatch.setattr("mstore.backend.os.ftruncate", fail_truncate)
    with pytest.raises(OSError, match="allocation failed"):
        create_region(4096, "test")
    assert len(opened) == 1
    with pytest.raises(OSError):
        os.fstat(opened[0])
