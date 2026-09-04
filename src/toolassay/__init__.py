"""toolassay: an evaluation harness for MCP tool contracts.

Given an MCP server's tool definitions and a file of natural-language cases, toolassay
asks a model to complete each case, records which tools it picked and what the run cost,
and lets you diff two runs so a tool-contract change can be proven cheaper without being
proven broken.
"""

__version__ = "0.1.0"
