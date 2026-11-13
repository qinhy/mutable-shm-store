"""One cumulative access level per token or public object.

none < info < read < write < share < admin

No permission sets or arbitrary permission combinations are supported.
"""

from __future__ import annotations

from typing import Literal

from .errors import InvalidRequest

PermissionLevel = Literal["none", "info", "read", "write", "share", "admin"]
LEVEL_ORDER: tuple[str, ...] = ("none", "info", "read", "write", "share", "admin")
_LEVEL_INDEX = {level: index for index, level in enumerate(LEVEL_ORDER)}

# Upper bound on anonymous access; lower to "write" or "read" if desired.
MAX_PUBLIC_LEVEL: PermissionLevel = "admin"


def validate_level(level: PermissionLevel) -> PermissionLevel:
    """Require exactly one named level (never a list, set, or custom combination)."""
    if not isinstance(level, str) or level not in _LEVEL_INDEX:
        raise InvalidRequest(f"level must be one of {LEVEL_ORDER}; got {level!r}")
    return level


def has_level(available: PermissionLevel, required: PermissionLevel) -> bool:
    """Whether ``available`` includes the ``required`` access level."""
    return _LEVEL_INDEX[validate_level(available)] >= _LEVEL_INDEX[validate_level(required)]


__all__ = ["LEVEL_ORDER", "MAX_PUBLIC_LEVEL", "PermissionLevel", "has_level", "validate_level"]
