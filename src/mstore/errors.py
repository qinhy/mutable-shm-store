class MStoreError(Exception):
    """Base error raised by mstore."""


class ProtocolError(MStoreError):
    pass


class AuthenticationError(MStoreError):
    pass


class PermissionDenied(MStoreError):
    pass


class ObjectNotFound(MStoreError):
    pass


class TokenExpired(MStoreError):
    pass


class TokenRevoked(MStoreError):
    pass


class InvalidRequest(MStoreError):
    pass


ERROR_TYPES = {
    "authentication_error": AuthenticationError,
    "permission_denied": PermissionDenied,
    "object_not_found": ObjectNotFound,
    "token_expired": TokenExpired,
    "token_revoked": TokenRevoked,
    "invalid_request": InvalidRequest,
    "protocol_error": ProtocolError,
}
