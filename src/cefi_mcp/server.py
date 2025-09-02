"""Server creation with FastMCP"""

import argparse
import asyncio

from fastmcp import FastMCP
from loguru import logger
from starlette.requests import Request
from starlette.responses import PlainTextResponse

from . import catalog, tools


async def create_server() -> FastMCP:
    """Create and configure the MCP server and register tools"""
    mcp = FastMCP('Server for Changing Ecosystems and Fisheries Data')

    # Register tools here instead of with a decorator
    # get_dataset_from_catalog returns a lot of data and is not usually
    # used, so it is disabled for now.
    # mcp.tool()(catalog.get_dataset_from_catalog)
    mcp.tool()(catalog.list_variables)
    mcp.tool()(catalog.filter_by_frequency)
    mcp.tool()(catalog.filter_by_variable)
    mcp.tool()(tools.query_variable_metadata)
    mcp.tool()(tools.get_variable_point)
    mcp.tool()(tools.get_variable_climatology_point)
    mcp.tool()(tools.geocode_ocean_place)

    # Add health check endpoint, mainly for Docker purposes
    @mcp.custom_route('/health', methods=['GET'])
    async def health_check(request: Request) -> PlainTextResponse:
        return PlainTextResponse('OK')

    return mcp


async def async_main(transport: str, host: str, port: int):
    # Disable logging for stdio transport to avoid interfering with MCP protocol
    if transport == 'stdio':
        logger.remove()
        logger.add(lambda _: None)

    server = await create_server()
    logger.info('Server created')
    if transport == 'stdio':
        await server.run_async(transport='stdio')
    elif transport in ['http', 'sse']:
        await server.run_async(transport=transport, host=host, port=port)


def main():
    parser = argparse.ArgumentParser(
        description='MCP Server for Changing Ecosystems and Fisheries Data'
    )
    parser.add_argument(
        '--transport',
        choices=['stdio', 'http', 'sse'],
        default='stdio',
        help='Transport protocol to use (default: stdio)',
    )
    parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='Host to bind to for http/sse transport (default: 127.0.0.1)',
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port to bind to for http/sse transport (default: 8000)',
    )

    args = parser.parse_args()

    # Limit what host can be
    allowed_hosts = ['127.0.0.1', 'localhost', '0.0.0.0']
    if args.host not in allowed_hosts:
        raise ValueError(f"Host '{args.host}' not allowed. Use one of: {allowed_hosts}")

    # A separate sync main function is needed because it is the entry point
    asyncio.run(async_main(args.transport, args.host, args.port))
