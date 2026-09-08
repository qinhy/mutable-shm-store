"""Simple cross-process-ish throughput benchmark against in-place shared memory.

Run mstore-server first, then:
    python benchmarks/pipeline.py --mib 1024 --passes 5
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import mstore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mib", type=int, default=256)
    parser.add_argument("--passes", type=int, default=5)
    args = parser.parse_args()

    nbytes = args.mib * 1024 * 1024
    store = mstore.connect()
    obj = store.create(shape=(nbytes,), dtype=np.uint8)
    arr = obj.numpy()
    arr.fill(1)

    start = time.perf_counter()
    for _ in range(args.passes):
        np.add(arr, 1, out=arr, casting="unsafe")
    elapsed = time.perf_counter() - start
    touched = nbytes * args.passes * 2  # rough read + write bytes
    print(f"object: {args.mib} MiB")
    print(f"passes: {args.passes}")
    print(f"elapsed: {elapsed:.3f}s")
    print(f"approx memory traffic: {touched / elapsed / 1024**3:.2f} GiB/s")


if __name__ == "__main__":
    main()
