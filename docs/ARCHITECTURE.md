# Architecture

## Invariants

1. Bulk data never travels through the control socket.
2. The daemon owns backing-memory lifetime.
3. Tokens are capabilities, not ownership records.
4. Multiple writers are legal; synchronization is opt-in and outside v0.1.
5. Revocation affects future opens, not mappings already established.
6. NumPy is an adapter over shared bytes, not the storage layer.

## Control plane

Each client request uses one short-lived local connection and one length-prefixed JSON message. This keeps
failure recovery simple and makes the protocol easy to reimplement in C/C++ later.

Operations in v0.1:

- `ping`
- `create`
- `open`
- `info`
- `grant`
- `revoke`
- `delete`

The wire protocol is deliberately private in v0.1; compatibility is promised at the Python API level only.

## Data plane: Linux

The daemon creates an anonymous `memfd`, sizes it with `ftruncate`, and retains the owning descriptor.
Authorized clients receive a descriptor over an AF_UNIX socket using `SCM_RIGHTS`.

For read opens the daemon reopens `/proc/self/fd/<fd>` with `O_RDONLY`, then passes that descriptor. For
write opens it duplicates the RW descriptor. The client maps with `mmap.ACCESS_READ` or `ACCESS_WRITE`.

## Data plane: Windows

The daemon creates a named anonymous mapping with `mmap(..., tagname=...)` and holds it open. After token
validation, clients receive the random tag name and map it with `ACCESS_READ` or `ACCESS_WRITE`.

The control plane uses loopback TCP because Python's SCM_RIGHTS helpers are Unix-only. Named pipes and
Windows ACL hardening are future work.

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
