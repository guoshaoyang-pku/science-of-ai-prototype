#!/usr/bin/env python3
"""Stdio MCP server that forwards every tool call to an HTTP handler in the parent loop process.

usage: kb_mcp_bridge.py <tools_spec.json> <handler_url>
tools_spec.json is a list of OpenAI-style function specs (same as kb_science_loop.tools_spec()).
"""
import asyncio
import json
import sys
import urllib.request

import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

SPEC = json.load(open(sys.argv[1]))
URL = sys.argv[2]
srv = Server("kb")


@srv.list_tools()
async def list_tools() -> list[types.Tool]:
    return [types.Tool(name=t["function"]["name"], description=t["function"]["description"],
                       inputSchema=t["function"]["parameters"]) for t in SPEC]


@srv.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    body = json.dumps({"name": name, "arguments": arguments or {}}).encode()

    def go() -> str:
        req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as r:
            return r.read().decode()

    return [types.TextContent(type="text", text=await asyncio.to_thread(go))]


async def main() -> None:
    async with stdio_server() as (r, w):
        await srv.run(r, w, srv.create_initialization_options())


asyncio.run(main())
