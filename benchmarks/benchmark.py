#!/usr/bin/env python3
"""Detailed latency / throughput benchmark for mstore.

Covers every operation the client exposes, plus the higher-level "exists" and
"get" / "put" patterns built on top of them:

  control       ping, exists (hit / miss / unauthorized), info, grant, revoke
  lifecycle     create (untouched / first-touch fill), delete
  data          open (read / write / cached), get (zero-copy view / copy /
                cached copy / bytes), put (uncached / cached), in-place mutate,
                private-memory memcpy baseline
  concurrency   N threads sharing one Client vs. one Client per thread
  multiprocess  N independent processes hitting the same object

By default a private mstore server is started as a *separate process* on a
temporary endpoint, so every number includes real cross-process IPC. Use
--endpoint to measure an already running server instead.

A verification pass runs first (exists hit/miss, read-only enforcement,
revocation, expiry, cache hits, delete ...) so a broken build never produces
plausible-looking numbers.

Examples:
    python benchmarks/benchmark.py
    python benchmarks/benchmark.py --quick
    python benchmarks/benchmark.py --list
    python benchmarks/benchmark.py --filter 'exists_*,get_*' --sizes 4K,1M,256M
    python benchmarks/benchmark.py --groups control,data --json out.json --raw
    python benchmarks/benchmark.py --endpoint unix:///run/user/1000/mstore-1000.sock
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import gc
import json
import math
import multiprocessing as mp
import os
import platform
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import numpy as np

import mstore
from mstore import Client, SharedObject
from mstore.errors import (
    AuthenticationError,
    MStoreError,
    ObjectNotFound,
    PermissionDenied,
    TokenExpired,
    TokenRevoked,
)

perf_ns = time.perf_counter_ns
KiB, MiB, GiB = 1024, 1024**2, 1024**3

# A syntactically plausible token that the server never issued. Must be
# non-empty: the server rejects an empty token before it looks up the object.
BOGUS_TOKEN = "x" * 43


# ============================================================================
# exists() helper
# ============================================================================


def exists(client: Client, object_id: str, token: str | None = None, *, strict: bool = False) -> bool:
    """Return True if ``object_id`` is live on the server.

    Implemented with the ``info`` op. Registry.validate() looks the object up
    *before* checking the token, so:

      * unknown / deleted object            -> ObjectNotFound      -> False
      * live object, bad/revoked/expired tok -> AuthenticationError etc. -> True
      * live object, valid token             -> success             -> True

    With ``strict=True`` token errors are re-raised instead, i.e. "exists and
    I am allowed to see it".
    """
    try:
        client.info(object_id, token or BOGUS_TOKEN)
    except ObjectNotFound:
        return False
    except (AuthenticationError, PermissionDenied, TokenExpired, TokenRevoked):
        if strict:
            raise
        return True
    return True


# ============================================================================
# Formatting helpers
# ============================================================================

_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([KMG]?)(?:I?B)?\s*$", re.IGNORECASE)


def parse_size(text: str) -> int:
    match = _SIZE_RE.match(text)
    if not match:
        raise argparse.ArgumentTypeError(f"invalid size: {text!r} (use e.g. 4096, 4K, 16M, 1G)")
    number, unit = match.groups()
    value = int(float(number) * {"": 1, "K": KiB, "M": MiB, "G": GiB}[unit.upper()])
    if value <= 0:
        raise argparse.ArgumentTypeError("sizes must be > 0")
    return value


def parse_size_list(text: str) -> list[int]:
    return [parse_size(part) for part in text.split(",") if part.strip()]


def parse_int_list(text: str) -> list[int]:
    values = [int(part) for part in text.split(",") if part.strip()]
    if not values or any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return values


def fmt_size(n: int | None) -> str:
    if n is None:
        return "-"
    for unit, scale in (("G", GiB), ("M", MiB), ("K", KiB)):
        if n >= scale:
            value = n / scale
            return f"{value:.0f}{unit}" if value == int(value) else f"{value:.1f}{unit}"
    return f"{n}B"


def fmt_ns(ns: float) -> str:
    if ns < 1e3:
        return f"{ns:.0f} ns"
    if ns < 1e6:
        return f"{ns / 1e3:.2f} µs"
    if ns < 1e9:
        return f"{ns / 1e6:.2f} ms"
    return f"{ns / 1e9:.2f} s"


def fmt_rate(ops_per_s: float) -> str:
    if ops_per_s >= 1e6:
        return f"{ops_per_s / 1e6:.2f}M"
    if ops_per_s >= 1e3:
        return f"{ops_per_s / 1e3:.1f}k"
    return f"{ops_per_s:.1f}"


def fmt_bw(bytes_per_s: float | None) -> str:
    if not bytes_per_s:
        return "-"
    if bytes_per_s >= GiB:
        return f"{bytes_per_s / GiB:.2f} GiB/s"
    return f"{bytes_per_s / MiB:.1f} MiB/s"


def percentile(sorted_values: list[int], pct: float) -> float:
    """Linear-interpolated percentile of an already sorted list."""
    if not sorted_values:
        return math.nan
    k = (len(sorted_values) - 1) * pct / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return float(sorted_values[lo])
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)


# ============================================================================
# Results and measurement primitives
# ============================================================================


@dataclass
class Result:
    name: str
    group: str
    samples_ns: list[int]
    elapsed_ns: int  # time base for ops/s: busy time (serial) or wall time (parallel)
    ops: int
    size: int | None = None
    bytes_per_op: int = 0
    workers: int = 1
    note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        s = sorted(self.samples_ns)
        n = len(s)
        secs = self.elapsed_ns / 1e9
        ops_per_s = self.ops / secs if secs > 0 else math.inf
        bytes_per_s = self.bytes_per_op * self.ops / secs if self.bytes_per_op and secs > 0 else None
        return {
            "name": self.name,
            "group": self.group,
            "size": self.size,
            "workers": self.workers,
            "samples": n,
            "ops": self.ops,
            "mean_ns": statistics.fmean(s) if n else math.nan,
            "stdev_ns": statistics.pstdev(s) if n > 1 else 0.0,
            "min_ns": s[0] if n else math.nan,
            "p50_ns": percentile(s, 50),
            "p90_ns": percentile(s, 90),
            "p99_ns": percentile(s, 99),
            "p999_ns": percentile(s, 99.9),
            "max_ns": s[-1] if n else math.nan,
            "elapsed_s": secs,
            "ops_per_s": ops_per_s,
            "bytes_per_op": self.bytes_per_op,
            "bytes_per_s": bytes_per_s,
            "note": self.note,
            **self.extra,
        }


def measure(
    name: str,
    group: str,
    fn: Callable[[Any], Any],
    *,
    iterations: int,
    warmup: int,
    setup: Callable[[], Any] | None = None,
    teardown: Callable[[Any, Any], None] | None = None,
    size: int | None = None,
    bytes_per_op: int = 0,
    note: str = "",
) -> Result:
    """Time ``fn(setup())`` per iteration; setup/teardown are excluded from timing."""

    def one() -> int:
        ctx = setup() if setup else None
        t0 = perf_ns()
        out = fn(ctx)
        t1 = perf_ns()
        if teardown:
            teardown(ctx, out)
        return t1 - t0

    for _ in range(warmup):
        one()
    gc.collect()
    gc_was_enabled = gc.isenabled()
    gc.disable()  # keep cyclic-GC pauses out of the samples
    try:
        samples = [one() for _ in range(iterations)]
    finally:
        if gc_was_enabled:
            gc.enable()
    return Result(
        name, group, samples, elapsed_ns=sum(samples), ops=iterations,
        size=size, bytes_per_op=bytes_per_op, note=note,
    )


def measure_threads(
    name: str,
    group: str,
    make_worker: Callable[[int], tuple[Callable[[], Any], Callable[[], None]]],
    *,
    threads: int,
    ops_per_thread: int,
    warmup: int,
    size: int | None = None,
    bytes_per_op: int = 0,
    note: str = "",
) -> Result:
    """Run ``op`` concurrently in N threads; ops/s is based on wall time.

    ``make_worker(tid)`` returns ``(op, cleanup)`` and is called inside the
    worker thread, so per-thread setup (connect, allocate) is not timed.
    """
    barrier = threading.Barrier(threads + 1)
    samples: list[list[int]] = [[] for _ in range(threads)]
    ends = [0] * threads
    errors: list[BaseException] = []

    def run(tid: int) -> None:
        try:
            op, cleanup = make_worker(tid)
        except BaseException as exc:  # noqa: BLE001 - reported to the main thread
            errors.append(exc)
            barrier.abort()
            return
        try:
            for _ in range(warmup):
                op()
            barrier.wait()
            local = samples[tid]
            for _ in range(ops_per_thread):
                t0 = perf_ns()
                op()
                local.append(perf_ns() - t0)
            ends[tid] = perf_ns()
        except threading.BrokenBarrierError:
            pass
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
            barrier.abort()
        finally:
            cleanup()

    pool = [threading.Thread(target=run, args=(i,), daemon=True) for i in range(threads)]
    for t in pool:
        t.start()
    start = 0
    try:
        barrier.wait(timeout=120)
        start = perf_ns()
    except threading.BrokenBarrierError:
        pass
    for t in pool:
        t.join()
    if errors:
        raise errors[0]
    flat = [x for per in samples for x in per]
    return Result(
        name, group, flat, elapsed_ns=max(ends) - start, ops=len(flat),
        size=size, bytes_per_op=bytes_per_op, workers=threads, note=note,
    )


def _process_op(client: Client, op: str, object_id: str, token: str, dest: np.ndarray | None) -> None:
    if op == "info":
        client.info(object_id, token)
        return
    if op == "exists":
        exists(client, object_id, token)
        return
    if op in {"get_copy", "get_copy_cached"}:
        obj = client.open(object_id, token, mode="read", cache=(op == "get_copy_cached"))
        try:
            src = obj.numpy()
            np.copyto(dest, src)
            del src
        finally:
            obj.close()
        return
    raise ValueError(f"unknown process op: {op}")


def _process_worker(endpoint, object_id, token, op, size, iterations, warmup, barrier, queue) -> None:
    """Top-level so it can be pickled by the 'spawn' start method."""
    try:
        client = mstore.connect(endpoint)
        dest = np.empty(size, dtype=np.uint8) if size else None
        if dest is not None:
            dest.fill(0)  # pre-fault destination pages
        for _ in range(warmup):
            _process_op(client, op, object_id, token, dest)
        barrier.wait()
        samples = []
        for _ in range(iterations):
            t0 = time.perf_counter_ns()
            _process_op(client, op, object_id, token, dest)
            samples.append(time.perf_counter_ns() - t0)
        queue.put(("ok", samples))
        client.close()
    except BaseException as exc:  # noqa: BLE001
        queue.put(("error", f"{type(exc).__name__}: {exc}"))
        try:
            barrier.abort()
        except Exception:
            pass


def measure_processes(
    name: str,
    group: str,
    *,
    endpoint: str,
    object_id: str,
    token: str,
    op: str,
    size: int,
    processes: int,
    ops_per_process: int,
    warmup: int,
    bytes_per_op: int = 0,
    note: str = "",
) -> Result:
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(processes + 1)
    queue = ctx.Queue()
    procs = [
        ctx.Process(
            target=_process_worker,
            args=(endpoint, object_id, token, op, size, ops_per_process, warmup, barrier, queue),
            daemon=True,
        )
        for _ in range(processes)
    ]
    for p in procs:
        p.start()
    try:
        try:
            barrier.wait(timeout=120)  # process start-up/import time is excluded
        except threading.BrokenBarrierError:
            status, payload = queue.get(timeout=10)
            raise RuntimeError(f"worker failed before start: {payload}") from None
        t0 = perf_ns()
        samples: list[int] = []
        for _ in range(processes):
            status, payload = queue.get(timeout=600)
            if status != "ok":
                raise RuntimeError(f"worker failed: {payload}")
            samples.extend(payload)
        t1 = perf_ns()
    finally:
        for p in procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
    return Result(
        name, group, samples, elapsed_ns=t1 - t0, ops=len(samples),
        size=size or None, bytes_per_op=bytes_per_op, workers=processes, note=note,
    )


# ============================================================================
# Server management
# ============================================================================


class ServerHandle:
    """Start (or attach to) an mstore server for the duration of the run."""

    def __init__(self, mode: str, endpoint: str | None) -> None:
        if mode == "auto":
            mode = "external" if endpoint else "process"
        self.mode = mode
        self.endpoint = endpoint
        self._tmpdir: str | None = None
        self._proc: subprocess.Popen | None = None
        self._stderr_path: str | None = None
        self._server = None
        self._thread: threading.Thread | None = None

    def _private_endpoint(self) -> str:
        if os.name == "nt":
            return f"pipe://mstore-bench-{uuid.uuid4().hex}"
        # Keep the path short: AF_UNIX paths are limited to ~104-108 bytes.
        base = "/tmp" if os.path.isdir("/tmp") else None
        assert self._tmpdir is not None
        sock_dir = tempfile.mkdtemp(prefix="mstore-bench-", dir=base)
        self._sock_dir = sock_dir
        return f"unix://{sock_dir}/s.sock"

    def __enter__(self) -> "ServerHandle":
        self._tmpdir = tempfile.mkdtemp(prefix="mstore-bench-log-")
        if self.mode == "external":
            self.endpoint = self.endpoint or mstore.default_endpoint()
            self._wait_ready(5.0)
            return self

        self.endpoint = self.endpoint or self._private_endpoint()
        if self.mode == "process":
            env = os.environ.copy()
            # Make sure the child imports the same mstore as this process.
            package_root = os.path.dirname(os.path.dirname(os.path.abspath(mstore.__file__)))
            env["PYTHONPATH"] = os.pathsep.join(filter(None, [package_root, env.get("PYTHONPATH")]))
            self._stderr_path = os.path.join(self._tmpdir, "server.stderr")
            with open(self._stderr_path, "wb") as err:
                self._proc = subprocess.Popen(
                    [sys.executable, "-m", "mstore.cli", "--endpoint", self.endpoint],
                    stdout=subprocess.DEVNULL,
                    stderr=err,
                    env=env,
                )
        elif self.mode == "thread":
            from mstore.server import MStoreServer

            self._server = MStoreServer(self.endpoint)
            self._thread = self._server.start_in_thread()
            self.endpoint = self._server.endpoint
        else:
            raise ValueError(f"unknown server mode: {self.mode}")
        self._wait_ready(15.0)
        return self

    def _wait_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        last: Exception | None = None
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(f"mstore server exited early:\n{self._server_stderr()}")
            try:
                with Client(self.endpoint, timeout=1.0) as probe:
                    if probe.ping().get("pong"):
                        return
            except Exception as exc:  # noqa: BLE001
                last = exc
            time.sleep(0.05)
        raise RuntimeError(f"mstore server at {self.endpoint} not reachable: {last}")

    def _server_stderr(self) -> str:
        if self._stderr_path and os.path.exists(self._stderr_path):
            with open(self._stderr_path, encoding="utf-8", errors="replace") as fh:
                return fh.read()[-4000:]
        return ""

    def __exit__(self, *exc: Any) -> None:
        if self._proc is not None and self._proc.poll() is None:
            try:
                if os.name == "nt":
                    self._proc.terminate()
                else:
                    self._proc.send_signal(signal.SIGINT)  # graceful: KeyboardInterrupt -> cleanup
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=5)
        if self._server is not None:
            self._server.shutdown()
            if self._thread is not None:
                self._thread.join(timeout=3)
        for path in (getattr(self, "_sock_dir", None), self._tmpdir):
            if path:
                shutil.rmtree(path, ignore_errors=True)


# ============================================================================
# Verification pass
# ============================================================================


def _raises(exc_type: type[BaseException], fn: Callable[..., Any], *args: Any, **kwargs: Any) -> bool:
    try:
        fn(*args, **kwargs)
    except exc_type:
        return True
    return False


def run_verification(c: Client) -> bool:
    print("Verification", flush=True)
    checks: list[tuple[str, bool, str]] = []

    def check(label: str, fn: Callable[[], Any]) -> None:
        try:
            ok, detail = bool(fn()), ""
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        checks.append((label, ok, detail))
        print(f"  {'PASS' if ok else 'FAIL'}  {label}{'  -> ' + detail if detail else ''}", flush=True)

    pattern = (np.arange(4096) % 251).astype(np.uint8)
    owner = c.create(shape=(4096,), dtype=np.uint8, metadata={"verify": True})
    oid = owner.object_id
    try:
        view = owner.numpy()
        view[:] = pattern
        del view
        read_tok = owner.issue("read")
        write_tok = owner.issue("write")

        check("ping returns pong", lambda: c.ping().get("pong") is True)
        check("exists: live object, valid token -> True", lambda: exists(c, oid, owner.token) is True)
        check("exists: unknown object id -> False", lambda: exists(c, uuid.uuid4().hex) is False)
        check("exists: live object, foreign token -> True", lambda: exists(c, oid, BOGUS_TOKEN) is True)
        check(
            "exists(strict): foreign token raises AuthenticationError",
            lambda: _raises(AuthenticationError, exists, c, oid, BOGUS_TOKEN, strict=True),
        )
        check(
            "info: metadata/shape round-trip",
            lambda: (
                (info := c.info(oid, read_tok)["object"])["metadata"] == {"verify": True}
                and info["shape"] == [4096]
            ),
        )

        def get_roundtrip() -> bool:
            with c.open(oid, read_tok, mode="read") as r:
                return bool(np.array_equal(r.numpy(), pattern))

        check("get: reader sees producer's bytes (zero-copy)", get_roundtrip)

        def read_only() -> bool:
            with c.open(oid, read_tok, mode="read") as r:
                arr = r.numpy()
                ok = (not arr.flags.writeable) and _raises(ValueError, arr.__setitem__, 0, 1)
                del arr
                return ok

        check("get: read token maps read-only", read_only)
        check(
            "open(write) with read token -> PermissionDenied",
            lambda: _raises(PermissionDenied, c.open, oid, read_tok, mode="write"),
        )

        def put_visible() -> bool:
            with c.open(oid, write_tok, mode="write") as w:
                arr = w.numpy()
                arr[:4] = [9, 8, 7, 6]
                del arr
            with c.open(oid, read_tok) as r:
                return r.numpy()[:4].tolist() == [9, 8, 7, 6]

        check("put: writer's bytes visible to a new reader", put_visible)

        def cache_hit() -> bool:
            c.open(oid, read_tok, cache=True).close()
            again = c.open(oid, read_tok, cache=True)
            hit = again.cache_hit
            again.close()
            c.clear_cache(oid)
            return hit

        check("open(cache=True): second open is a cache hit", cache_hit)

        def revoked() -> bool:
            tok = owner.issue("read")
            owner.revoke(tok)
            return _raises(TokenRevoked, c.open, oid, tok)

        check("revoke: revoked token cannot open", revoked)

        def expired() -> bool:
            tok = owner.issue("read", expires_in=0.05)
            time.sleep(0.1)
            return _raises(TokenExpired, c.open, oid, tok)

        check("grant(expires_in): expired token cannot open", expired)

        def deleted() -> bool:
            tmp = c.create(size=64)
            before = exists(c, tmp.object_id, tmp.token)
            tmp.delete()
            tmp.close()
            return before and not exists(c, tmp.object_id, tmp.token)

        check("delete: exists flips True -> False", deleted)
    finally:
        c.clear_cache(oid)
        try:
            owner.delete()
        except MStoreError:
            pass
        owner.close()

    failed = [label for label, ok, _ in checks if not ok]
    print(f"  {len(checks) - len(failed)}/{len(checks)} checks passed\n", flush=True)
    return not failed


# ============================================================================
# Benchmark suite
# ============================================================================


@dataclass(frozen=True)
class Spec:
    name: str
    group: str
    kind: str  # single | sized | threads | processes
    method: str
    description: str


SPECS: list[Spec] = [
    # --- control plane ------------------------------------------------------
    Spec("ping", "control", "single", "b_ping", "Round trip with no registry work: floor of control IPC"),
    Spec("exists_hit", "control", "single", "b_exists_hit", "exists() on a live object with its own token"),
    Spec("exists_miss", "control", "single", "b_exists_miss", "exists() on an unknown id (ObjectNotFound path)"),
    Spec("exists_unauthorized", "control", "single", "b_exists_unauth", "exists() on a live object with a foreign token"),
    Spec("info", "control", "single", "b_info", "info(): object metadata + token permissions"),
    Spec("grant", "control", "single", "b_grant", "issue a read token"),
    Spec("grant_expiring", "control", "single", "b_grant_expiring", "issue a read token with expires_in"),
    Spec("revoke", "control", "single", "b_revoke", "revoke a freshly issued token (issue is untimed)"),
    # --- lifecycle ----------------------------------------------------------
    Spec("create", "lifecycle", "sized", "b_create", "create(): allocate region, pass fd, mmap; pages untouched"),
    Spec("create_fill", "lifecycle", "sized", "b_create_fill", "create() + first-touch fill (includes page faults)"),
    Spec("delete", "lifecycle", "sized", "b_delete", "delete() (create is untimed)"),
    # --- data plane ---------------------------------------------------------
    Spec("open_read", "data", "sized", "b_open_read", "uncached open(mode=read): IPC + fd passing + mmap"),
    Spec("open_write", "data", "sized", "b_open_write", "uncached open(mode=write)"),
    Spec("open_cached", "data", "sized", "b_open_cached", "open(cache=True) hit: no IPC, no mmap"),
    Spec("get_view", "data", "sized", "b_get_view", "cached open + numpy() view + close: zero-copy get"),
    Spec("get_copy", "data", "sized", "b_get_copy", "uncached open + copy whole object into a private buffer"),
    Spec("get_copy_cached", "data", "sized", "b_get_copy_cached", "cached open + copy into a private buffer"),
    Spec("get_bytes", "data", "sized", "b_get_bytes", "cached open + bytes(buffer()): copy into a new bytes object"),
    Spec("put", "data", "sized", "b_put", "uncached open(write) + copy a private buffer in"),
    Spec("put_cached", "data", "sized", "b_put_cached", "cached open(write) + copy a private buffer in"),
    Spec("mutate_inplace", "data", "sized", "b_mutate", "np.add in place on the shared mapping (read+write traffic)"),
    Spec("memcpy_baseline", "data", "sized", "b_memcpy", "private-memory np.copyto: upper bound for get/put"),
    # --- concurrency (threads) ----------------------------------------------
    Spec("threads_shared_client_ping", "concurrency", "threads", "b_t_shared_ping", "N threads share one Client (serialized stream)"),
    Spec("threads_ping", "concurrency", "threads", "b_t_ping", "N threads, one Client/connection each"),
    Spec("threads_exists", "concurrency", "threads", "b_t_exists", "N threads, exists() hit, one Client each"),
    Spec("threads_open_read", "concurrency", "threads", "b_t_open_read", "N threads, uncached open+close, one Client each"),
    Spec("threads_get_copy", "concurrency", "threads", "b_t_get_copy", "N threads, uncached get_copy of --concurrency-size"),
    # --- multiprocess -------------------------------------------------------
    Spec("processes_exists", "multiprocess", "processes", "b_p_exists", "N processes, exists() hit"),
    Spec("processes_get_copy", "multiprocess", "processes", "b_p_get_copy", "N processes, uncached get_copy"),
    Spec("processes_get_copy_cached", "multiprocess", "processes", "b_p_get_copy_cached", "N processes, cached get_copy"),
]
GROUP_ORDER = ["control", "lifecycle", "data", "concurrency", "multiprocess"]


def _noop() -> None:
    pass


class Suite:
    def __init__(self, client: Client, args: argparse.Namespace, endpoint: str) -> None:
        self.c = client
        self.a = args
        self.endpoint = endpoint

    # ---- helpers -----------------------------------------------------------

    def _data_iters(self, size: int, *, touches_data: bool = True, workers: int = 1) -> int:
        """Scale iterations so large objects stay within --bytes-budget."""
        if not touches_data:
            return self.a.data_iterations
        budget_iters = self.a.bytes_budget // (size * workers)
        return max(self.a.min_iterations, min(self.a.data_iterations, budget_iters))

    def _warm(self, iterations: int) -> int:
        return min(self.a.warmup, max(1, iterations // 4))

    @contextmanager
    def _object(self, size: int, *, fill: bool = True) -> Iterator[SharedObject]:
        obj = self.c.create(shape=(size,), dtype=np.uint8, metadata={"benchmark": True})
        try:
            if fill:  # pre-fault every page so reads don't hit the zero page
                view = obj.numpy()
                view.fill(0xA5)
                del view
            yield obj
        finally:
            self.c.clear_cache(obj.object_id)
            try:
                obj.delete()
            except MStoreError:
                pass
            obj.close()

    def _control(self, name: str, fn: Callable[[Any], Any], **kw: Any) -> Result:
        return measure(name, "control", fn, iterations=self.a.iterations, warmup=self.a.warmup, **kw)

    # ---- control -----------------------------------------------------------

    def b_ping(self) -> Result:
        return self._control("ping", lambda _: self.c.ping())

    def b_exists_hit(self) -> Result:
        with self._object(64) as o:
            return self._control("exists_hit", lambda _: exists(self.c, o.object_id, o.token))

    def b_exists_miss(self) -> Result:
        missing = uuid.uuid4().hex
        return self._control("exists_miss", lambda _: exists(self.c, missing), note="error-response path")

    def b_exists_unauth(self) -> Result:
        with self._object(64) as o:
            return self._control(
                "exists_unauthorized", lambda _: exists(self.c, o.object_id, BOGUS_TOKEN),
                note="token rejected after lookup",
            )

    def b_info(self) -> Result:
        with self._object(64) as o:
            return self._control("info", lambda _: self.c.info(o.object_id, o.token))

    def b_grant(self) -> Result:
        with self._object(64) as o:
            return self._control("grant", lambda _: self.c.issue_token(o.object_id, o.token, "read"))

    def b_grant_expiring(self) -> Result:
        with self._object(64) as o:
            return self._control(
                "grant_expiring",
                lambda _: self.c.issue_token(o.object_id, o.token, "read", expires_in=3600.0),
            )

    def b_revoke(self) -> Result:
        with self._object(64) as o:
            return self._control(
                "revoke",
                lambda tok: self.c.revoke_token(o.object_id, o.token, tok),
                setup=lambda: self.c.issue_token(o.object_id, o.token, "read"),
            )

    # ---- lifecycle ---------------------------------------------------------

    def _close_deleted(self, _ctx: Any, obj: SharedObject) -> None:
        obj.delete()
        obj.close()

    def b_create(self, size: int) -> Result:
        n = self._data_iters(size, touches_data=False)
        return measure(
            "create", "lifecycle", lambda _: self.c.create(shape=(size,), dtype=np.uint8),
            teardown=self._close_deleted, iterations=n, warmup=self._warm(n), size=size,
        )

    def b_create_fill(self, size: int) -> Result:
        def fn(_: Any) -> SharedObject:
            obj = self.c.create(shape=(size,), dtype=np.uint8)
            view = obj.numpy()
            view.fill(1)
            del view
            return obj

        n = self._data_iters(size)
        return measure(
            "create_fill", "lifecycle", fn, teardown=self._close_deleted,
            iterations=n, warmup=self._warm(n), size=size, bytes_per_op=size,
        )

    def b_delete(self, size: int) -> Result:
        def fn(obj: SharedObject) -> SharedObject:
            obj.delete()
            return obj

        n = self._data_iters(size, touches_data=False)
        return measure(
            "delete", "lifecycle", fn,
            setup=lambda: self.c.create(shape=(size,), dtype=np.uint8),
            teardown=lambda _ctx, obj: obj.close(),
            iterations=n, warmup=self._warm(n), size=size,
        )

    # ---- data plane --------------------------------------------------------

    def _open_bench(self, name: str, size: int, mode: str, cache: bool) -> Result:
        with self._object(size) as o:
            tok = o.issue(mode)
            if cache:
                self.c.open(o.object_id, tok, mode=mode, cache=True).close()  # prime
            before = self.c.cache_info()
            n = self._data_iters(size, touches_data=False)
            result = measure(
                name, "data", lambda _: self.c.open(o.object_id, tok, mode=mode, cache=cache),
                teardown=lambda _ctx, obj: obj.close(),
                iterations=n, warmup=self._warm(n), size=size,
            )
            if cache:
                after = self.c.cache_info()
                result.extra["cache_hits"] = after["hits"] - before["hits"]
                result.extra["cache_misses"] = after["misses"] - before["misses"]
            return result

    def b_open_read(self, size: int) -> Result:
        return self._open_bench("open_read", size, "read", cache=False)

    def b_open_write(self, size: int) -> Result:
        return self._open_bench("open_write", size, "write", cache=False)

    def b_open_cached(self, size: int) -> Result:
        return self._open_bench("open_cached", size, "read", cache=True)

    def b_get_view(self, size: int) -> Result:
        with self._object(size) as o:
            tok = o.issue("read")
            self.c.open(o.object_id, tok, cache=True).close()

            def fn(_: Any) -> int:
                with self.c.open(o.object_id, tok, cache=True) as obj:
                    arr = obj.numpy()
                    first = int(arr[0])  # touch one element so the view is really used
                    del arr
                return first

            n = self._data_iters(size, touches_data=False)
            return measure(
                "get_view", "data", fn, iterations=n, warmup=self._warm(n), size=size,
                note="zero-copy: should be flat across sizes",
            )

    def _get_copy(self, name: str, size: int, cache: bool) -> Result:
        with self._object(size) as o:
            tok = o.issue("read")
            dest = np.zeros(size, dtype=np.uint8)  # pre-faulted destination

            def fn(_: Any) -> None:
                obj = self.c.open(o.object_id, tok, mode="read", cache=cache)
                try:
                    src = obj.numpy()
                    np.copyto(dest, src)
                    del src
                finally:
                    obj.close()

            n = self._data_iters(size)
            return measure(
                name, "data", fn, iterations=n, warmup=self._warm(n),
                size=size, bytes_per_op=size,
            )

    def b_get_copy(self, size: int) -> Result:
        return self._get_copy("get_copy", size, cache=False)

    def b_get_copy_cached(self, size: int) -> Result:
        return self._get_copy("get_copy_cached", size, cache=True)

    def b_get_bytes(self, size: int) -> Result:
        with self._object(size) as o:
            tok = o.issue("read")
            self.c.open(o.object_id, tok, cache=True).close()

            def fn(_: Any) -> bytes:
                with self.c.open(o.object_id, tok, cache=True) as obj:
                    view = obj.buffer()
                    try:
                        return bytes(view)
                    finally:
                        view.release()

            n = self._data_iters(size)
            return measure(
                "get_bytes", "data", fn, iterations=n, warmup=self._warm(n),
                size=size, bytes_per_op=size, note="includes allocating the bytes object",
            )

    def _put(self, name: str, size: int, cache: bool) -> Result:
        with self._object(size) as o:
            tok = o.issue("write")
            src = np.full(size, 0x5A, dtype=np.uint8)

            def fn(_: Any) -> None:
                obj = self.c.open(o.object_id, tok, mode="write", cache=cache)
                try:
                    dst = obj.numpy()
                    np.copyto(dst, src)
                    del dst
                finally:
                    obj.close()

            n = self._data_iters(size)
            return measure(
                name, "data", fn, iterations=n, warmup=self._warm(n),
                size=size, bytes_per_op=size,
            )

    def b_put(self, size: int) -> Result:
        return self._put("put", size, cache=False)

    def b_put_cached(self, size: int) -> Result:
        return self._put("put_cached", size, cache=True)

    def b_mutate(self, size: int) -> Result:
        with self._object(size) as o:
            views = [o.numpy()]  # held in a list so it can be dropped before delete()

            def fn(_: Any) -> None:
                arr = views[0]
                np.add(arr, 1, out=arr, casting="unsafe")

            n = self._data_iters(size)
            result = measure(
                "mutate_inplace", "data", fn,
                iterations=n, warmup=self._warm(n), size=size, bytes_per_op=2 * size,
                note="throughput counts read + write",
            )
            views.clear()
            return result

    def b_memcpy(self, size: int) -> Result:
        src = np.full(size, 0x5A, dtype=np.uint8)
        dst = np.zeros(size, dtype=np.uint8)
        n = self._data_iters(size)
        return measure(
            "memcpy_baseline", "data", lambda _: np.copyto(dst, src),
            iterations=n, warmup=self._warm(n), size=size, bytes_per_op=size,
            note="no mstore involved",
        )

    # ---- concurrency -------------------------------------------------------

    def _threads(self, name: str, make_worker, workers: int, *, size: int | None = None,
                 bytes_per_op: int = 0, ops: int | None = None, note: str = "") -> Result:
        ops = ops or self.a.concurrency_ops
        return measure_threads(
            name, "concurrency", make_worker, threads=workers, ops_per_thread=ops,
            warmup=self._warm(ops), size=size, bytes_per_op=bytes_per_op, note=note,
        )

    def _own_client(self) -> Client:
        return Client(self.endpoint, timeout=self.a.timeout, cache_size=self.a.cache_size)

    def b_t_shared_ping(self, workers: int) -> Result:
        return self._threads(
            "threads_shared_client_ping", lambda _tid: (self.c.ping, _noop), workers,
            note="one connection, request lock",
        )

    def b_t_ping(self, workers: int) -> Result:
        def make(_tid: int):
            client = self._own_client()
            return client.ping, client.close

        return self._threads("threads_ping", make, workers)

    def b_t_exists(self, workers: int) -> Result:
        with self._object(64) as o:
            def make(_tid: int):
                client = self._own_client()
                return (lambda: exists(client, o.object_id, o.token)), client.close

            return self._threads("threads_exists", make, workers)

    def b_t_open_read(self, workers: int) -> Result:
        with self._object(64) as o:
            tok = o.issue("read")

            def make(_tid: int):
                client = self._own_client()
                return (lambda: client.open(o.object_id, tok).close()), client.close

            return self._threads("threads_open_read", make, workers)

    def b_t_get_copy(self, workers: int) -> Result:
        size = self.a.concurrency_size
        with self._object(size) as o:
            tok = o.issue("read")

            def make(_tid: int):
                client = self._own_client()
                dest = np.zeros(size, dtype=np.uint8)

                def op() -> None:
                    obj = client.open(o.object_id, tok)
                    try:
                        src = obj.numpy()
                        np.copyto(dest, src)
                        del src
                    finally:
                        obj.close()

                return op, client.close

            ops = min(self.a.concurrency_ops, self._data_iters(size, workers=workers))
            return self._threads(
                "threads_get_copy", make, workers, size=size, bytes_per_op=size, ops=ops,
            )

    # ---- multiprocess ------------------------------------------------------

    def _procs(self, name: str, op: str, workers: int, size: int | None) -> Result:
        obj_size = size or 64
        with self._object(obj_size) as o:
            tok = o.issue("read")
            ops = self.a.concurrency_ops
            if size:
                ops = min(ops, self._data_iters(size, workers=workers))
            return measure_processes(
                name, "multiprocess", endpoint=self.endpoint, object_id=o.object_id,
                token=tok, op=op, size=size or 0, processes=workers, ops_per_process=ops,
                warmup=self._warm(ops), bytes_per_op=size or 0,
            )

    def b_p_exists(self, workers: int) -> Result:
        return self._procs("processes_exists", "exists", workers, None)

    def b_p_get_copy(self, workers: int) -> Result:
        return self._procs("processes_get_copy", "get_copy", workers, self.a.concurrency_size)

    def b_p_get_copy_cached(self, workers: int) -> Result:
        return self._procs("processes_get_copy_cached", "get_copy_cached", workers, self.a.concurrency_size)

    # ---- driver ------------------------------------------------------------

    def run(self, specs: list[Spec]) -> tuple[list[Result], list[str]]:
        jobs: list[tuple[Spec, int | None]] = []
        for spec in specs:
            if spec.kind == "single":
                jobs.append((spec, None))
            elif spec.kind == "sized":
                jobs.extend((spec, s) for s in self.a.sizes)
            elif spec.kind == "threads":
                jobs.extend((spec, n) for n in self.a.threads)
            elif spec.kind == "processes":
                jobs.extend((spec, n) for n in self.a.processes)

        results: list[Result] = []
        failures: list[str] = []
        width = len(str(len(jobs)))
        for index, (spec, param) in enumerate(jobs, 1):
            label = spec.name
            if spec.kind == "sized":
                label += f" [{fmt_size(param)}]"
            elif spec.kind in {"threads", "processes"}:
                label += f" [x{param}]"
            print(f"[{index:>{width}}/{len(jobs)}] {label:<44}", end="", flush=True, file=sys.stderr)
            started = time.perf_counter()
            try:
                method = getattr(self, spec.method)
                result = method() if param is None else method(param)
            except KeyboardInterrupt:
                print(" interrupted", file=sys.stderr)
                failures.append(f"{label}: interrupted (remaining benchmarks skipped)")
                break
            except Exception as exc:  # noqa: BLE001
                print(f" FAILED ({type(exc).__name__}: {exc})", file=sys.stderr)
                failures.append(f"{label}: {type(exc).__name__}: {exc}")
                continue
            summary = result.summary()
            print(
                f" {time.perf_counter() - started:6.2f}s  p50 {fmt_ns(summary['p50_ns'])}",
                file=sys.stderr, flush=True,
            )
            results.append(result)
        return results, failures


# ============================================================================
# Reporting
# ============================================================================


def environment(args: argparse.Namespace, server: ServerHandle) -> dict[str, Any]:
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": f"{platform.python_implementation()} {platform.python_version()}",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "mstore": getattr(mstore, "__version__", "unknown"),
        "server_mode": server.mode,
        "endpoint": server.endpoint,
        "iterations": args.iterations,
        "data_iterations": args.data_iterations,
        "warmup": args.warmup,
        "bytes_budget": args.bytes_budget,
        "sizes": args.sizes,
        "threads": args.threads,
        "processes": args.processes,
    }


def print_environment(env: dict[str, Any]) -> None:
    print("mstore benchmark")
    print(f"  python    {env['python']}  numpy {env['numpy']}  mstore {env['mstore']}")
    print(f"  platform  {env['platform']} ({env['machine']}, {env['cpu_count']} CPUs)")
    print(f"  server    {env['server_mode']} @ {env['endpoint']}")
    if env["server_mode"] == "thread":
        print("  note      thread-mode server shares this process's GIL; latencies are pessimistic")
    print(flush=True)


def print_report(results: list[Result]) -> None:
    headers = ["benchmark", "size", "N", "samples", "mean", "p50", "p90", "p99", "max", "ops/s", "throughput"]
    by_group: dict[str, list[Result]] = {}
    for r in results:
        by_group.setdefault(r.group, []).append(r)

    for group in GROUP_ORDER + sorted(set(by_group) - set(GROUP_ORDER)):
        rows = by_group.get(group)
        if not rows:
            continue
        table = []
        for r in rows:
            s = r.summary()
            table.append([
                r.name, fmt_size(r.size), str(r.workers), str(s["samples"]),
                fmt_ns(s["mean_ns"]), fmt_ns(s["p50_ns"]), fmt_ns(s["p90_ns"]),
                fmt_ns(s["p99_ns"]), fmt_ns(s["max_ns"]), fmt_rate(s["ops_per_s"]),
                fmt_bw(s["bytes_per_s"]),
            ])
        widths = [max(len(h), *(len(row[i]) for row in table)) for i, h in enumerate(headers)]
        line = "  ".join(
            h.ljust(widths[i]) if i == 0 else h.rjust(widths[i]) for i, h in enumerate(headers)
        )
        print(f"\n== {group} " + "=" * max(0, len(line) - len(group) - 4))
        print(line)
        print("-" * len(line))
        for row in table:
            print("  ".join(
                cell.ljust(widths[i]) if i == 0 else cell.rjust(widths[i]) for i, cell in enumerate(row)
            ))
        notes = {r.name: r.note for r in rows if r.note}
        for name, note in notes.items():
            print(f"  * {name}: {note}")
    print(
        "\nN = threads/processes. ops/s uses busy time for serial runs and wall time for "
        "concurrent runs.\nThroughput = bytes moved per op x ops/s "
        "(mutate_inplace counts read + write)."
    )


def write_json(path: str, env: dict[str, Any], results: list[Result], failures: list[str],
               cache: dict[str, int], raw: bool) -> None:
    payload = {
        "environment": env,
        "client_cache": cache,
        "failures": failures,
        "results": [
            {**r.summary(), **({"samples_ns": r.samples_ns} if raw else {})} for r in results
        ],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)


def write_csv(path: str, results: list[Result]) -> None:
    rows = [r.summary() for r in results]
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================================
# CLI
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Detailed mstore benchmark (control plane, data plane, concurrency).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Benchmark names (for --filter) are shown by --list.",
    )
    srv = p.add_argument_group("server")
    srv.add_argument("--endpoint", help="benchmark an existing server at this endpoint")
    srv.add_argument(
        "--server", choices=["auto", "process", "thread", "external"], default="auto",
        help="auto = external if --endpoint is given, else spawn a private server process",
    )
    srv.add_argument("--timeout", type=float, default=30.0, help="client socket timeout (s)")

    sel = p.add_argument_group("selection")
    sel.add_argument("--list", action="store_true", help="list benchmarks and exit")
    sel.add_argument("--filter", default="*", help="comma-separated name globs, e.g. 'exists_*,get_*'")
    sel.add_argument("--groups", default=",".join(GROUP_ORDER), help="comma-separated groups to run")
    sel.add_argument("--quick", action="store_true", help="small iteration counts for a fast smoke run")
    sel.add_argument("--no-verify", action="store_true", help="skip the correctness pass")
    sel.add_argument("--keep-going", action="store_true", help="benchmark even if verification fails")

    load = p.add_argument_group("workload")
    load.add_argument("--sizes", type=parse_size_list, default=None,
                      help="object sizes for sized benchmarks (default 4K,64K,1M,16M,64M)")
    load.add_argument("--iterations", type=int, default=None, help="iterations for control ops (default 2000)")
    load.add_argument("--data-iterations", type=int, default=None, help="max iterations per sized run (default 200)")
    load.add_argument("--min-iterations", type=int, default=5, help="floor for sized runs (default 5)")
    load.add_argument("--warmup", type=int, default=None, help="warm-up iterations (default 50)")
    load.add_argument("--bytes-budget", type=parse_size, default=None,
                      help="bytes moved per sized run before iterations are capped (default 2G)")
    load.add_argument("--threads", type=parse_int_list, default=None, help="thread counts (default 1,2,4,8)")
    load.add_argument("--processes", type=parse_int_list, default=None, help="process counts (default 1,2,4)")
    load.add_argument("--concurrency-ops", type=int, default=None, help="ops per worker (default 1000)")
    load.add_argument("--concurrency-size", type=parse_size, default=parse_size("1M"),
                      help="object size for concurrent get benchmarks (default 1M)")
    load.add_argument("--cache-size", type=int, default=64, help="client mapping cache capacity")

    out = p.add_argument_group("output")
    out.add_argument("--json", metavar="PATH", help="write summary (and environment) as JSON")
    out.add_argument("--raw", action="store_true", help="include every latency sample in the JSON")
    out.add_argument("--csv", metavar="PATH", help="write summary rows as CSV")
    return p


def resolve_defaults(args: argparse.Namespace) -> None:
    quick = args.quick
    defaults = {
        "sizes": [4 * KiB, 1 * MiB, 16 * MiB] if quick else [4 * KiB, 64 * KiB, 1 * MiB, 16 * MiB, 64 * MiB],
        "iterations": 200 if quick else 2000,
        "data_iterations": 30 if quick else 200,
        "warmup": 10 if quick else 50,
        "bytes_budget": 256 * MiB if quick else 2 * GiB,
        "threads": [1, 4] if quick else [1, 2, 4, 8],
        "processes": [1, 2] if quick else [1, 2, 4],
        "concurrency_ops": 100 if quick else 1000,
    }
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)


def select_specs(args: argparse.Namespace) -> list[Spec]:
    patterns = [p.strip() for p in args.filter.split(",") if p.strip()] or ["*"]
    groups = {g.strip() for g in args.groups.split(",") if g.strip()}
    unknown = groups - set(GROUP_ORDER)
    if unknown:
        raise SystemExit(f"unknown group(s): {sorted(unknown)}; choose from {GROUP_ORDER}")
    return [
        s for s in SPECS
        if s.group in groups and any(fnmatch.fnmatchcase(s.name, pat) for pat in patterns)
    ]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    resolve_defaults(args)

    if args.list:
        for group in GROUP_ORDER:
            print(f"{group}:")
            for s in SPECS:
                if s.group == group:
                    print(f"  {s.name:<28} {s.description}")
        return 0

    specs = select_specs(args)
    if not specs:
        print("no benchmarks match --filter/--groups", file=sys.stderr)
        return 1

    with ServerHandle(args.server, args.endpoint) as server:
        env = environment(args, server)
        print_environment(env)
        with Client(server.endpoint, timeout=args.timeout, cache_size=args.cache_size) as client:
            if not args.no_verify and not run_verification(client) and not args.keep_going:
                print("verification failed; not benchmarking (use --keep-going to override)", file=sys.stderr)
                return 2
            results, failures = Suite(client, args, server.endpoint).run(specs)
            cache = client.cache_info()

    print_report(results)
    if failures:
        print("\nFailures:")
        for failure in failures:
            print(f"  - {failure}")
    if args.json:
        write_json(args.json, env, results, failures, cache, args.raw)
        print(f"\nJSON written to {args.json}")
    if args.csv:
        write_csv(args.csv, results)
        print(f"CSV written to {args.csv}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
