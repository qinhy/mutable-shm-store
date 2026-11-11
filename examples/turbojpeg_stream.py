"""Decode JPEG frames straight into mstore shared memory with PyTurboJPEG.

A simulated camera produces RGB frames, they are JPEG-encoded, and each one is
decoded *directly into* a shared object: no intermediate array, no copy. A
second process then opens the object by its fixed id, without a token.

Requirements:
    pip install "PyTurboJPEG<2"        # with libjpeg-turbo 2.x (e.g. Ubuntu 24.04 apt)
    pip install PyTurboJPEG            # only with libjpeg-turbo 3.0 or newer
    apt install libturbojpeg  |  brew install jpeg-turbo  |  Windows: libjpeg-turbo installer

Usage:
    python examples/turbojpeg_stream.py                     # starts a private server
    python examples/turbojpeg_stream.py --endpoint unix:///run/user/1000/mstore-1000.sock
    python examples/turbojpeg_stream.py --frames 100 --width 3840 --height 2160
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
from turbojpeg import TJPF_RGB, TJSAMP_420, TurboJPEG

import mstore

CHANNELS = {TJPF_RGB: 3}


# --------------------------------------------------------------------------- simulation

def simulate_frame(index: int, height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    """Synthetic camera frame: colour gradient + a moving square + sensor noise."""
    y = np.linspace(0, 255, height, dtype=np.float32)[:, None]
    x = np.linspace(0, 255, width, dtype=np.float32)[None, :]
    frame = np.empty((height, width, 3), dtype=np.float32)
    frame[..., 0] = x
    frame[..., 1] = y
    frame[..., 2] = 128 + 64 * np.sin((x + y) / 40 + index / 5)

    side = height // 5
    top = (index * 17) % (height - side)
    left = (index * 31) % (width - side)
    frame[top:top + side, left:left + side] = (255, 255, 255)

    frame += rng.normal(0, 4, size=frame.shape).astype(np.float32)
    return frame.clip(0, 255).astype(np.uint8)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


# --------------------------------------------------------------------------- zero-copy decode

def decode_into(tj: TurboJPEG, jpeg: bytes, dst: np.ndarray, *, pixel_format: int = TJPF_RGB,
                scaling_factor: tuple[int, int] | None = None, fast: bool = False) -> np.ndarray:
    """Decode ``jpeg`` into ``dst`` in place, refusing every case that would copy.

    PyTurboJPEG silently allocates a *new* array when dst does not match exactly,
    and it assumes a C-contiguous layout (pitch 0). Check all of that up front.
    """
    if not dst.flags.writeable:
        raise ValueError("dst is read-only: open the object with mode='write'")
    if not dst.flags.c_contiguous:
        raise ValueError("dst must be C-contiguous: create the object with order='C'")
    width, height, _sub, _cs = tj.decode_header(jpeg)
    if scaling_factor:
        num, den = scaling_factor
        width, height = (width * num + den - 1) // den, (height * num + den - 1) // den
    expected = (height, width, CHANNELS[pixel_format])
    if dst.shape != expected or dst.dtype != np.uint8:
        raise ValueError(f"dst must be uint8 {expected}, got {dst.dtype} {dst.shape}")

    flags = 2048 if fast else 0  # TJFLAG_FASTDCT
    out = tj.decode(jpeg, pixel_format=pixel_format, scaling_factor=scaling_factor,
                    flags=flags, dst=dst)
    if out is not dst:  # belt and braces against the silent-fallback behaviour
        raise RuntimeError("PyTurboJPEG decoded into a new array instead of dst")
    return out


# --------------------------------------------------------------------------- reader process

READER_CODE = r"""
import sys, numpy as np, mstore
endpoint, object_id = sys.argv[1:3]
obj = mstore.connect(endpoint).open(object_id)          # public: no token needed
arr = obj.numpy()
print(f"  reader pid {__import__('os').getpid()}: {object_id} shape={arr.shape} "
      f"mean={arr.mean():.2f} pixel[0,0]={arr[0, 0].tolist()} writeable={arr.flags.writeable}")
"""


# --------------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--endpoint", help="existing mstore server (default: start a private one)")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--quality", type=int, default=90)
    args = parser.parse_args()

    server = None
    endpoint = args.endpoint
    if endpoint is None:
        from mstore.server import MStoreServer
        if os.name == "nt":
            endpoint = f"pipe://mstore-turbojpeg-{os.getpid()}"
        else:
            endpoint = f"unix://{tempfile.mkdtemp(prefix='mstore-', dir='/tmp')}/s.sock"
        server = MStoreServer(endpoint)
        server.start_in_thread()

    tj = TurboJPEG()
    rng = np.random.default_rng(0)
    store = mstore.connect(endpoint)

    # 1. Learn the frame size from the first JPEG header, then allocate once.
    first_jpeg = tj.encode(simulate_frame(0, args.height, args.width, rng),
                           quality=args.quality, pixel_format=TJPF_RGB, jpeg_subsample=TJSAMP_420)
    width, height, _sub, _cs = tj.decode_header(first_jpeg)
    frame_obj = store.create(shape=(height, width, 3), dtype=np.uint8, order="C",
                             object_id="sim-frame", metadata={"format": "RGB"})
    half_obj = store.create(shape=((height + 1) // 2, (width + 1) // 2, 3), dtype=np.uint8,
                            object_id="sim-frame-half", metadata={"format": "RGB", "scale": 0.5})
    frame_obj.make_public("read")
    half_obj.make_public("read")
    frame_dst = frame_obj.numpy()   # writable view of the shared memory
    half_dst = half_obj.numpy()
    print(f"objects: sim-frame {frame_dst.shape}, sim-frame-half {half_dst.shape}  "
          f"({frame_dst.nbytes / 2**20:.1f} MiB + {half_dst.nbytes / 2**20:.1f} MiB shared)\n")

    # 2. Stream simulated frames: encode -> decode straight into shared memory.
    enc_ms, dec_ms, half_ms, quality = [], [], [], []
    for i in range(args.frames):
        original = simulate_frame(i, args.height, args.width, rng)
        t0 = time.perf_counter()
        jpeg = tj.encode(original, quality=args.quality, pixel_format=TJPF_RGB,
                         jpeg_subsample=TJSAMP_420)
        t1 = time.perf_counter()
        decode_into(tj, jpeg, frame_dst)                               # full size, zero-copy
        t2 = time.perf_counter()
        decode_into(tj, jpeg, half_dst, scaling_factor=(1, 2), fast=True)  # DCT-domain downscale
        t3 = time.perf_counter()
        enc_ms.append((t1 - t0) * 1e3)
        dec_ms.append((t2 - t1) * 1e3)
        half_ms.append((t3 - t2) * 1e3)
        quality.append(psnr(original, frame_dst))
        if i < 3 or i == args.frames - 1:
            print(f"frame {i:3d}: jpeg {len(jpeg) / 1024:6.0f} KiB  encode {enc_ms[-1]:5.1f} ms  "
                  f"decode->shm {dec_ms[-1]:5.1f} ms  half->shm {half_ms[-1]:5.1f} ms  "
                  f"PSNR {quality[-1]:.1f} dB")
        elif i == 3:
            print("  ...")

    print(f"\nmedian: encode {np.median(enc_ms):.1f} ms, decode into shm {np.median(dec_ms):.1f} ms, "
          f"half-size decode {np.median(half_ms):.1f} ms, PSNR {np.median(quality):.1f} dB")
    print(f"decoded array is the shared object itself: {np.shares_memory(frame_dst, frame_obj.numpy())}")

    # 3. Another process reads the last frame by its fixed id, without a token.
    print("\nindependent reader processes:")
    for object_id in ("sim-frame", "sim-frame-half"):
        subprocess.run([sys.executable, "-c", READER_CODE, endpoint, object_id], check=True, timeout=30)
    print(f"  writer pid {os.getpid()}:  sim-frame mean={frame_dst.mean():.2f} "
          f"pixel[0,0]={frame_dst[0, 0].tolist()}")

    # 4. Show the guard: a wrongly shaped dst is refused instead of silently copied.
    try:
        decode_into(tj, jpeg, half_dst)
    except ValueError as exc:
        print(f"\nguard works: {exc}")

    del frame_dst, half_dst
    for obj in (frame_obj, half_obj):
        obj.delete()
        obj.close()
    store.close()
    if server is not None:
        server.shutdown()


if __name__ == "__main__":
    main()
