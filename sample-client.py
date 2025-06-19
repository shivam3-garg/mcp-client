import asyncio
import json
import os
from typing import Optional, Dict
from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.client.sse import sse_client

from openai import OpenAI
from openai.types import Completion
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import base64
import httpx
import traceback

import logging, traceback

logger = logging.getLogger("uvicorn.error")
load_dotenv()  # load environment variables from .env

# In-memory session store
session_memory: Dict[str, list] = {}

SYSTEM_PROMPT = """
You are a Paytm MCP Assistant, an AI agent powered by the Paytm MCP Server, which enables secure access to Paytm's Payments and Business Payments APIs. Your role is to automate payment workflows using the available tools: create_payment_link, fetch_payment_links, fetch_transactions_for_link, initiate_refund, check_refund_status, fetch_refund_list, and fetch_order_list.

1. Understand the Request:
- Identify the user's intent and choose the correct tool.
- Extract all relevant parameters from the user’s message (e.g., amount, order_id, txn_id, refund_reference_id, etc).

2. Parameter Validation:
- Always follow the tool’s schema.
- For `create_payment_link`: either `customer_email` or `customer_mobile` is sufficient. Never ask for both.
- For `initiate_refund`: all of these must be present — `order_id`, `txn_id`, `refund_reference_id`, `refund_amount`. If `refund_reference_id` is missing but `order_id` is present, suggest a value like `refund_<order_id>` (e.g., `refund_ORDR1234`).
- For `fetch_refund_list` and `fetch_order_list`: never assume `start_date`, `end_date`, `from_date`, or `to_date`.Do not allow more than a 30-day range.

3. Tool Execution:
- Call the tool only when all required parameters are available.
- Normalize user phrasing as needed (e.g., "return payment" → initiate_refund).
- Only send parameters that are accepted by the tool.

4. Output Handling:
- For `create_payment_link`, confirm `short_url` starts with `https://paytm.me/`.
- For refunds/orders, show formatted tables or lists when data is returned.
- Clearly state if the refund/order list is empty.

5. Error Handling:
- If required parameters are missing, ask the user clearly (e.g., "Please provide an email address or mobile number to send the payment link.").
- Show clear error messages if a tool fails.

6. Response Formatting:
- Use clean markdown formatting always.   
- Example:
  - **Action**: Created payment link
  - **Amount**: ₹50
  - **Purpose**: Snacks
  - **Link**: https://paytm.me/PYTMPS/abc123
  - **Email Sent**: Yes
  - **SMS Sent**: No
- If an error occurs, explain it simply and guide the user with next steps.
- For lists (e.g., multiple orders or refunds), display them as a markdown table with proper headers and columns.

7. Maintain Context:
- Use prior messages to infer missing info.
- Remember recent link IDs, recipient names, etc., for follow-up questions.

8. Multi-Step or Chained Requests:
- If user intent requires multiple tools (e.g., refund + status check), sequence tool calls accordingly.
- Make it clear to the user what’s happening, and confirm each step before proceeding.

9. Language Matching:
- For **each user message**, detect the language used (e.g., Hindi, English, Hinglish).
- Respond in **that same language**, regardless of what language was used earlier in the session.
- his ensures users can switch freely between languages (e.g., start in English, switch to Hindi, and back).
- Maintain clarity and formatting (bullets, markdown, labels) regardless of the language used.

10. Date Parameters:
- Never invent or guess `from_date` or `to_date`.
- If user says "last 5 days", "last 10 days", "past week", etc.:
    → Use `time_range` (e.g., `time_range = 5`) and **do not pass** `from_date` or `to_date`.
- Only use `from_date` and `to_date` if user explicitly gives full date ranges.
- Never pass `time_range` **alongside** `from_date` or `to_date` — use one method only.
- Always keep the date range within **30 days**.


Be concise, friendly, and focused. Guide Paytm merchants with speed and clarity.
"""

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.available_tools = []

    async def connect_to_sse_server(self, server_url: str):
        print("Connecting to MCP SSE server...")
        self._streams_context = sse_client(url=server_url)
        streams = await self._streams_context.__aenter__()
        print("Streams:", streams)

        self._session_context = ClientSession(*streams)
        self.session: ClientSession = await self._session_context.__aenter__()

        print("Initializing SSE client...")
        await self.session.initialize()
        print("Initialized SSE client")

        await self.get_available_tools()

    async def cleanup(self):
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if self._streams_context:
            await self._streams_context.__aexit__(None, None, None)

    async def get_available_tools(self):
        print("Fetching available server tools...")
        response = await self.session.list_tools()
        print("Connected to MCP server with tools:", [tool.name for tool in response.tools])

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
            for tool in response.tools
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

# Define FastAPI app
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="MCP Tool Assistant")  # ✅ Define app once

# ✅ Add CORS middleware to the same app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or restrict to ["https://buddy-paytm-chat.lovable.app"]
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
    await client.connect_to_sse_server(server_url="https://payment-ol-mcp.onrender.com/sse")

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
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        reply = await client.process_query(request.message, request.session_id)
        return JSONResponse(content={"reply": reply})
    except Exception as e:
        # 🔴 NEW: dump traceback to logs
        logger.error("Unhandled error in /chat\n%s", traceback.format_exc())
        # also send it back so you can see it in Postman/curl
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("sample-client:app", host="0.0.0.0", port=8000, reload=True)
