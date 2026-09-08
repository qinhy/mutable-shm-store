# Security model

mstore v0.1 is a **same-host, same-trust-domain** shared-memory service.

- Tokens are 256-bit opaque capability secrets; the daemon stores SHA-256 hashes, not raw tokens.
- Linux uses anonymous `memfd` objects. Clients cannot attach by name; the daemon must pass an FD.
- Linux read opens receive a descriptor reopened `O_RDONLY` and are mapped `ACCESS_READ`.
- Windows uses named kernel mappings because Unix FD passing is unavailable. The random mapping name is
  disclosed only after token validation, but v0.1 does not install custom Windows ACLs. A process in the
  same login/session that learns the mapping name may bypass the mstore control plane.
- Revocation prevents **future** opens. It cannot invalidate a mapping/FD already handed to a client.
- Multiple write tokens may coexist. mstore does not serialize writers in v0.1; application-level data
  races are the caller's responsibility.
- Do not expose the TCP endpoint beyond loopback. Network/multi-host access is out of scope for v0.1.

For hostile multi-user environments, add OS identity checks/ACLs before production deployment.
