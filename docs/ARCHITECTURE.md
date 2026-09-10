# Architecture

## Invariants

1. Bulk data never travels through the control channel.
2. The daemon owns backing-memory lifetime.
3. Tokens are capabilities, not ownership records.
4. Multiple writers are legal; synchronization is opt-in and outside v0.2.
5. Revocation affects future opens, not mappings already established.
6. NumPy is an adapter over shared bytes, not the storage layer.

## Control plane

Each client lazily establishes one local control connection and reuses it for strict, sequential
request/response exchanges. Client-side locking makes a session safe to share across application threads.
Connections are not shared across a process fork: the child drops the inherited connection and opens its
own session on demand.

Socket transports carry length-prefixed JSON frames. Windows named pipes preserve message boundaries but
carry the same framed JSON payload, keeping protocol validation and future non-Python implementations
consistent. A malformed stream is answered once when possible and then closed rather than reused.

Operations in v0.2:

- `ping`
- `create`
- `open`
- `info`
- `grant`
- `revoke`
- `delete`

The wire protocol is deliberately private in v0.2; compatibility is promised at the Python API level only.

## Data plane: Linux

The daemon creates an anonymous `memfd`, sizes it with `ftruncate`, and retains the owning descriptor.
Authorized clients receive a descriptor over an AF_UNIX socket using `SCM_RIGHTS`.

For read opens the daemon reopens `/proc/self/fd/<fd>` with `O_RDONLY`, then passes that descriptor. For
write opens it duplicates the RW descriptor. The client maps with `mmap.ACCESS_READ` or `ACCESS_WRITE`.

## Data plane: Windows

The daemon creates a named anonymous mapping with `mmap(..., tagname=...)` and holds it open. After token
validation, clients receive the random tag name and map it with `ACCESS_READ` or `ACCESS_WRITE`.

The control plane uses a Windows named pipe by default. Loopback TCP is retained as an explicit fallback;
the mapping name, rather than an OS descriptor, is sufficient to attach the data plane on Windows. Custom
Windows ACL hardening remains future work.

## Client mapping cache

`open(..., cache=True)` stores the attached mapping in a bounded LRU cache keyed by object ID, SHA-256
token digest, and access mode. A `SharedObject` lease and cache ownership are tracked independently:

- closing a `SharedObject` releases its lease but leaves a cached mapping attached;
- eviction or `clear_cache()` closes an idle mapping;
- a live `SharedObject` keeps an evicted mapping valid until its lease ends;
- `Client.close()` clears the cache and closes the persistent control connection.

Reusing a cache entry performs no control request. It therefore has the same revocation and deletion
semantics as any other already-established mapping.

## Token model

The server returns random 256-bit opaque tokens and stores only their SHA-256 hashes.

- `read`: read + metadata
- `write`: read + write + metadata
- `admin`: all permissions, including grant and delete

Granting is monotonic: a token cannot delegate permissions it does not itself possess. Expiring child
tokens cannot outlive an expiring issuer.

## Lifecycle

Deleting an object removes it from the registry and closes the daemon-owned backing handle. Existing
client mappings continue to reference the same pages until those clients close their mappings. This is
intentional and mirrors revocation semantics.
