"""Category modules for the local-image-processor MCP server.

Each module registers its tools onto the shared MCPServer instance at
import time via the ``@mcp.tool()`` decorator. Importing this package is
not sufficient for registration — server.py explicitly imports the
submodules below.

Shared utilities (no tool registration) live in ``tools.io_utils`` and are
imported directly by tool modules as needed.
"""

__all__ = ["geometry", "tonal", "color", "detail", "composite", "encoding", "info"]
