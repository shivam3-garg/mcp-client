import asyncio
import json
import os
from typing import Optional, Dict
from contextlib import AsyncExitStack

from fastmcp import Client  # Updated import for Streamable HTTP
from openai import OpenAI
from openai.types import Completion
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import base64
import httpx
import traceback

load_dotenv()

session_memory: Dict[str, list] = {}

SYSTEM_PROMPT = """
[REDACTED FOR BREVITY — same prompt as before]
"""

class MCPClient:
    def __init__(self):
        self.session: Optional[Client] = None
        self.exit_stack = AsyncExitStack()
        self.openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.available_tools = []

    async def connect_to_http_server(self, server_url: str):
        print("Connecting to MCP HTTP server...")
        self._session_context = Client(server_url)
        self.session = await self._session_context.__aenter__()
        await self.get_available_tools()

    async def cleanup(self):
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)

    async def get_available_tools(self):
        print("Fetching available server tools...")
        tools = await self.session.list_tools()
        print("Connected to MCP server with tools:", [tool.name for tool in tools])

        self.available_tools = [
            {
                "type": 'function',
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
                "strict": True,
            }
            for tool in tools
        ]

    async def call_openai(self, messages):
        return self.openai.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=1000,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, *messages],
            tools=self.available_tools
        )

    async def process_openai_response(self, response: Completion, session_id: str) -> str:
        messages = session_memory[session_id]

        for choice in response.choices:
            if choice.finish_reason == "tool_calls":
                messages.append(choice.message)

                for tool_call in choice.message.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)

                    print(f"\n[Calling tool {tool_name} with args {tool_args}]...")
                    result = await self.session.call_tool(tool_name, tool_args)
                    print(f"\nTool response: {result}")
                    if result and getattr(result, "content", None) and isinstance(result.content, list):
                        first_block = result.content[0]
                        tool_output_text = getattr(first_block, "text", "").strip()
                    else:
                        tool_output_text = ""

                    if not tool_output_text:
                        tool_output_text = "Tool executed but no response was returned."

                    print(f"[Tool Call ID: {tool_call.id}] Response: {tool_output_text}")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_output_text,
                    })

                response = await self.call_openai(messages)
                return await self.process_openai_response(response, session_id)

            elif choice.finish_reason == "stop":
                print("\nAssistant: " + choice.message.content)
                messages.append(choice.message)
                return choice.message.content

        return "Sorry, I couldn't complete that request."

    async def process_query(self, query: str, session_id: str) -> str:
        if session_id not in session_memory:
            session_memory[session_id] = []

        session_memory[session_id].append({"role": "user", "content": query})
        response = await self.call_openai(session_memory[session_id])
        return await self.process_openai_response(response, session_id)

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="MCP Tool Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: str

client = MCPClient()

@app.on_event("startup")
async def startup_event():
    await client.connect_to_http_server(server_url="https://payment-ol-mcp.onrender.com/mcp")

@app.on_event("shutdown")
async def shutdown_event():
    await client.cleanup()

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        reply = await client.process_query(request.message, request.session_id)
        return JSONResponse(content={"reply": reply})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("sample-client:app", host="0.0.0.0", port=8000, reload=True)
