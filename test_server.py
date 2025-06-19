import asyncio
from mcp.client.sse import sse_client
from mcp import ClientSession
import traceback

async def main():
    try:
        print("Connecting to MCP server...")
        async with sse_client("https://payment-ol-mcp.onrender.com/sse") as streams:
            print("✅ Connected to SSE streams:", streams)
            async with ClientSession(*streams) as session:
                print("✅ Session initialized.")
                tools = await session.list_tools()
                print("Available tools:", [tool.name for tool in tools.tools])
    except Exception as e:
        print("❌ Error:", str(e))
        print("🔍 Full traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
