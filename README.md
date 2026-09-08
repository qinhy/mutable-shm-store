# mutable-shm-store (`mstore`)

A small, token-capability **mutable shared-memory object server** for large NumPy arrays and byte buffers.
It is designed for pipelines where copying immutable multi-GB objects between stages is the bottleneck.

**v0.1 targets Linux and Windows.** Linux uses anonymous `memfd` + Unix-domain-socket FD passing. Windows
uses named kernel `mmap` mappings + a loopback control socket. The public Python API is the same.

> Status: alpha / reference-quality v0.1. The core zero-copy path, token model, tests, packaging, examples,
> and CI are included. Read `SECURITY.md` before production use.

## Why

Instead of:

```text
stage A -> copy/version -> stage B -> copy/version -> stage C
```

mstore keeps one mutable allocation:

```text
                     mstore daemon
                    metadata + tokens
                           |
        +------------------+------------------+
        |                  |                  |
     process A          process B          process C
        |                  |                  |
        +------------------+------------------+
                           |
                 SAME shared-memory pages
```

The control plane sends object metadata and capabilities. The bulk data never passes through the server.

## Features

- Mutable zero-copy shared memory between independent local processes
- NumPy arrays directly backed by shared pages
- Raw byte-buffer objects
- JSON-serializable user metadata attached to objects
- Opaque capability tokens: `read`, `write`, `admin`
- Token delegation, expiry, and revocation
- Read-only client mappings for read tokens
- Server-owned object lifetime; client crashes do not automatically destroy objects
- Linux: `memfd_create()` + `SCM_RIGHTS`
- Windows: named `mmap` kernel mappings
- No mandatory ownership-transfer protocol and no mandatory writer lock

## Install

```bash
python -m pip install -e .
```

For development:

```bash
python -m pip install -e ".[dev]"
pytest -q
```

## Start the server

Linux:

```bash
mstore-server
# default: unix:///tmp/mstore-<uid>.sock (or XDG_RUNTIME_DIR)
```

Windows:

```powershell
mstore-server
# default: tcp://127.0.0.1:65432
```

Choose an endpoint explicitly:

```bash
mstore-server --endpoint unix:///tmp/my-mstore.sock
```

```powershell
mstore-server --endpoint tcp://127.0.0.1:65432
```

## Python API

Create an image:

```python
import numpy as np
import mstore

store = mstore.connect()
image = store.create(shape=(2160, 3840, 3), dtype=np.uint8)

arr = image.numpy()
arr.fill(10)                 # writes directly into shared memory

read_token = image.issue("read")
write_token = image.issue("write")
print(image.object_id, read_token, write_token)
```

Another independent process can mutate the same pages:

```python
import numpy as np
import mstore

store = mstore.connect()
obj = store.open(OBJECT_ID, WRITE_TOKEN, mode="write")
arr = obj.numpy()
np.add(arr, 5, out=arr, casting="unsafe")
```

A reader receives a read-only mapping:

```python
obj = store.open(OBJECT_ID, READ_TOKEN, mode="read")
arr = obj.numpy()
assert not arr.flags.writeable
```

Raw bytes:

```python
obj = store.create(size=1024 * 1024)
obj.buffer()[:4] = b"MSTR"
```

Expiring capability:

```python
token = image.issue("read", expires_in=60)
```

Revocation blocks future opens:

```python
image.revoke(token)
```

Already-established mappings intentionally remain usable after revocation. This is a property of local
shared memory, not a bug; see `SECURITY.md`.

## Token semantics

| Public token type | Effective permissions |
|---|---|
| `read` | read + metadata |
| `write` | read + write + metadata |
| `admin` | read + write + metadata + grant + delete |

Tokens are opaque 256-bit secrets. The daemon stores only SHA-256 token hashes. A grant-capable token cannot
delegate permissions it does not already possess.

mstore intentionally allows multiple write tokens. If two writers touch the same bytes concurrently,
synchronization is your responsibility. A later version can layer optional leases/region locks on top of
this token model without changing the data plane.

## Architecture

```text
                   +-----------------------+
                   |     mstore daemon     |
                   |-----------------------|
                   | object registry       |
                   | token registry        |
                   | lifetime management   |
                   +-----------+-----------+
                               |
                 tiny control messages only
                               |
             +-----------------+-----------------+
             |                                   |
          Linux                              Windows
   Unix socket + FD pass               loopback TCP + name
             |                                   |
          memfd                               named mmap
             +-----------------+-----------------+
                               |
                         NumPy / mmap
```

### Linux

The daemon creates an anonymous `memfd`, keeps its descriptor open, and sends a duplicated descriptor to
authorized clients via `SCM_RIGHTS`. Read clients receive an `O_RDONLY` descriptor; write clients receive
an RW descriptor.

### Windows

The daemon owns a named kernel `mmap` mapping. After token validation the client receives its random mapping
name and opens it with `ACCESS_READ` or `ACCESS_WRITE`. The daemon's open handle keeps the mapping alive.

## Object lifecycle

```text
create -> open/read/write -> optional grant/revoke -> delete
```

`delete()` prevents new opens and drops the daemon's backing-memory reference. Existing client mappings
continue until those clients close them. This makes deletion safe for live NumPy views.

## Performance guidance

mstore avoids the *handoff copy*. Your algorithm can still allocate temporaries. Prefer in-place NumPy APIs:

```python
np.add(arr, 5, out=arr, casting="unsafe")
```

rather than expressions that allocate a whole new array:

```python
arr = arr + 5
```

For image/tensor pipelines, measure algorithmic temporary allocations separately from IPC copies.

## Run tests

```bash
pytest -q
```

The GitHub Actions matrix runs on Ubuntu and Windows using Python 3.10 and 3.13.

## Current scope / non-goals

v0.1 intentionally does not provide:

- multi-host/network object access
- persistence across daemon restarts
- arbitrary Python object serialization
- mandatory write locking or ownership transfer
- hard revocation of already-established mappings
- region-scoped capabilities
- Windows custom ACL management

Those can be layered on after the core mutable zero-copy path is stable.

## Roadmap

1. Region-scoped read/write tokens for tiled images and tensor partitions
2. Optional exclusive leases / reader-writer synchronization
3. Persistent metadata journal and daemon restart recovery
4. C/C++ client library using the same wire protocol and mappings
5. Metrics, quotas, pressure-aware eviction, and observability
6. Hardened Windows ACLs / peer identity checks
7. Optional RDMA-aware multi-node data plane as a separate subsystem

## License

MIT License
