"""Permission names and the named levels built from them.

A token, and an object's public (token-less) setting, each hold a *set* of
permissions. Levels are named presets ordered from least to most access, and
every level contains all permissions of the levels below it:

    level   info  read  write  grant  delete   typical use
    none     -     -     -      -      -       make_private()
    info     x     -     -      -      -       check existence / metadata only
    read     x     x     -      -      -       consumers (read-only mapping)
    write    x     x     x      -      -       workers that mutate in place
    share    x     x     x      x      -       can hand out / revoke tokens
    admin    x     x     x      x      x       owner (token returned by create)

Anywhere a permission argument is accepted you can pass a level name
("read"), a single permission ("delete"), or an iterable of permissions
(["read", "grant"]).
"""

from __future__ import annotations

from typing import Iterable

from .errors import InvalidRequest

INFO = "info"  # see metadata: shape, dtype, metadata, existence
READ = "read"  # map the object read-only
WRITE = "write"  # map the object read-write
GRANT = "grant"  # issue/revoke tokens and change public access
DELETE = "delete"  # delete the object

ALL_PERMISSIONS = frozenset({INFO, READ, WRITE, GRANT, DELETE})

# Ordered least -> most access. Edit this table to change what the names mean.
PERMISSION_LEVELS: dict[str, frozenset[str]] = {
    "none": frozenset(),
    "info": frozenset({INFO}),
    "read": frozenset({INFO, READ}),
    "write": frozenset({INFO, READ, WRITE}),
    "share": frozenset({INFO, READ, WRITE, GRANT}),
    "admin": ALL_PERMISSIONS,
}
LEVEL_ORDER: tuple[str, ...] = tuple(PERMISSION_LEVELS)

# The highest level that may be given to everyone (anonymous, token-less access).
# "admin" means anyone who can reach the server may also delete the object and
# issue tokens for it: right for shared caches, wrong for data you want to keep.
# Lower this (e.g. to "write") to forbid that server-wide.
MAX_PUBLIC_LEVEL = "admin"
PUBLIC_ALLOWED = PERMISSION_LEVELS[MAX_PUBLIC_LEVEL]


def _check_levels() -> None:
    previous: frozenset[str] = frozenset()
    for name, perms in PERMISSION_LEVELS.items():
        if perms - ALL_PERMISSIONS:
            raise AssertionError(f"level {name!r} uses unknown permissions {sorted(perms - ALL_PERMISSIONS)}")
        if not previous <= perms:
            raise AssertionError(f"level {name!r} must include every permission of the level below it")
        previous = perms


_check_levels()


def normalize_permissions(value: str | Iterable[str]) -> frozenset[str]:
    """Turn a level name, permission name, or iterable of permissions into a set.

    WRITE implies READ, and any non-empty set implies INFO.
    """
    if isinstance(value, str):
        if value in PERMISSION_LEVELS:
            return PERMISSION_LEVELS[value]
        value = [value]
    result = set(value)
    unknown = result - ALL_PERMISSIONS
    if unknown:
        raise InvalidRequest(
            f"unknown permissions: {sorted(unknown)}; "
            f"permissions are {sorted(ALL_PERMISSIONS)}, levels are {list(LEVEL_ORDER)}"
        )
    if WRITE in result:
        result.add(READ)
    if result:
        result.add(INFO)
    return frozenset(result)


def level_of(permissions: Iterable[str]) -> str:
    """Name of the level that exactly matches ``permissions``, or ``"custom"``."""
    perms = frozenset(permissions)
    for name, level in PERMISSION_LEVELS.items():
        if perms == level:
            return name
    return "custom"


def has_level(permissions: Iterable[str], level: str) -> bool:
    """True if ``permissions`` include everything the named level grants."""
    if level not in PERMISSION_LEVELS:
        raise InvalidRequest(f"unknown level {level!r}; levels are {list(LEVEL_ORDER)}")
    return PERMISSION_LEVELS[level] <= frozenset(permissions)


__all__ = [
    "ALL_PERMISSIONS",
    "DELETE",
    "GRANT",
    "INFO",
    "LEVEL_ORDER",
    "MAX_PUBLIC_LEVEL",
    "PERMISSION_LEVELS",
    "PUBLIC_ALLOWED",
    "READ",
    "WRITE",
    "has_level",
    "level_of",
    "normalize_permissions",
]
