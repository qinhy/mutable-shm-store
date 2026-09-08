from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .backend import Region


@dataclass
class ObjectRecord:
    object_id: str
    size: int
    region: Region
    generation: int = 1
    shape: list[int] | None = None
    dtype: str | None = None
    order: str = "C"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0

    def public(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "size": self.size,
            "generation": self.generation,
            "shape": self.shape,
            "dtype": self.dtype,
            "order": self.order,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


@dataclass
class TokenRecord:
    token_hash: str
    object_id: str
    permissions: frozenset[str]
    generation: int
    created_at: float
    expires_at: float | None = None
    revoked: bool = False
