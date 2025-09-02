"""Test configuration and fixtures for CEFI MCP Server tests"""

import pytest


@pytest.fixture(scope='session')
def event_loop_policy():
    """Use asyncio event loop for async tests"""
    import asyncio  # noqa: PLC0415

    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear LRU caches before each test to ensure isolation"""
    from cefi_mcp.tools import _cached_open_dataset  # noqa: PLC0415

    # Clear the cache before each test
    _cached_open_dataset.cache_clear()

    yield

    # Optionally clear after test too
    _cached_open_dataset.cache_clear()
