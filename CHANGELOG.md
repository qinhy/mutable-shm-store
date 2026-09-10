# Changelog

## 0.2.0

- Reuse one persistent, thread-safe control connection per client.
- Use Windows named pipes as the default control transport, with loopback TCP as a fallback.
- Add an opt-in LRU cache for shared-memory mappings via `open(..., cache=True)`.
- Add explicit `Client.close()` and client context-manager support.
- Reset inherited client connections and cached mappings after a process fork.
- Harden control-frame validation and file-descriptor cleanup.

## 0.1.0

- Initial Linux `memfd` + SCM_RIGHTS backend.
- Initial Windows named-mmap backend.
- Mutable NumPy and raw-buffer views.
- Read/write/admin capability tokens.
- Delegation, expiry, and revocation.
- Server-owned object lifetime and delete semantics.
- Cross-platform GitHub Actions matrix.
