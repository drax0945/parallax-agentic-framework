"""A local web console for Parallax.

Served from the machine Parallax runs on, because it drives the real agent and
streams real tool calls — nothing here is a mock-up.
"""

from .server import serve

__all__ = ["serve"]
