from __future__ import annotations

from typing import ClassVar

from core.exceptions.base import NexusError
from core.models.execution import ErrorCategory


class ServerError(NexusError):
    category: ClassVar[ErrorCategory] = ErrorCategory.CONNECTION


class ServerNotFoundError(ServerError):
    error_code = "SERVER_NOT_FOUND"
    category = ErrorCategory.VALIDATION


class ServerAlreadyRegisteredError(ServerError):
    error_code = "SERVER_ALREADY_REGISTERED"
    category = ErrorCategory.VALIDATION


class ServerDisabledError(ServerError):
    error_code = "SERVER_DISABLED"
    category = ErrorCategory.VALIDATION


class RegistryPersistenceError(ServerError):
    """The persistent server configuration store failed."""

    error_code = "REGISTRY_PERSISTENCE_FAILED"
    category = ErrorCategory.INTERNAL
    default_retryable = True


class ServerConnectionError(ServerError):
    error_code = "SERVER_CONNECTION_FAILED"
    default_retryable = True
