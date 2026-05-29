"""Session runtime abstractions.

This package holds worker-internal runtime objects.  These objects sit between
the worker transport protocol and concrete inference backends, so transport
code does not need to know backend bookkeeping details such as duplex finalize.
"""

from runtime.backend_client import RemoteBackendSession
from runtime.session import BackendRuntimeSession

__all__ = [
    "BackendRuntimeSession",
    "RemoteBackendSession",
]
