"""Compatibility layer for legacy imports.

This module re-exports the storage Database so existing imports like
`from src.core.database import Database` keep working while the project
is migrated to the `src.storage` namespace.
"""

from src.storage.database import Database

__all__ = ["Database"]