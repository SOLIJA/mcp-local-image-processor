"""local-image-processor — an offline MCP server for local image processing.

Exposes 34 image-processing tools (geometry, tonal, color, detail,
compositing, encoding/metadata, inspection) built on Pillow and OpenCV. All processing
is strictly offline; no external APIs or cloud services are used.

Run with:
    python server.py            # stdio transport (default)

The mcp 2.x SDK was used: ``mcp.server.mcpserver.MCPServer``.

.. note::
   The MCP SDK has evolved toward ``mcp.server.fastmcp.FastMCP`` as the primary
   server class.  This server uses the lower-level ``MCPServer`` for fine-grained
   control over tool registration.  If you upgrade the ``mcp`` package and find that
   ``MCPServer`` is no longer available, migrate to ``FastMCP`` and replace
   ``@mcp.tool()`` with ``@mcp.tool`` (the decorator signature differs slightly).
"""

from __future__ import annotations

import sys
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parent
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from mcp.server.mcpserver import MCPServer

# Shared server instance. Tool modules below import this object and
# register their tools onto it at import time via the ``@mcp.tool()`` decorator.
# Importing this package is not sufficient for registration — server.py explicitly
# imports the submodules below.
mcp = MCPServer(
    "local-image-processor",
    description=(
        "Offline image-processing server: resize, crop, rotate, tonal "
        "adjustments, color operations, detail filters, compositing, and "
        "format/metadata tools. All processing runs locally."
    ),
    instructions=(
        "Use the tools to transform images stored on the local filesystem. "
        "Every tool takes an input_path (and usually an output_path) as a "
        "string. Parameters are documented with their valid ranges in each "
        "tool's description. Errors are returned with is_error=True and a "
        "human-readable message."
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# sys.modules alias — why it's needed
# ---------------------------------------------------------------------------
# When launched as ``python server.py``, this module runs as ``__main__``.  The
# tool submodules do ``from server import mcp``.  Without the alias below, Python
# would treat ``server`` and ``__main__`` as *two* distinct module objects, each
# with its own ``mcp`` instance: tools would register on the ``server`` copy while
# ``mcp.run()`` below would serve the ``__main__`` copy — resulting in a server
# with zero registered tools.  By inserting this module under the name ``server``
# in ``sys.modules``, both import paths resolve to the *same* module object and
# therefore the same ``mcp`` instance.
if __name__ == "__main__":
    sys.modules["server"] = sys.modules[__name__]

# Register all tools. Each submodule applies @mcp.tool() to its functions
# at import time. This must come after the sys.modules alias above so the
# submodules' ``from server import mcp`` resolves to this very instance.
from tools import (  # noqa: F401
    color,
    composite,
    detail,
    encoding,
    geometry,
    info,
    tonal,
)


def _main() -> None:
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    _main()
