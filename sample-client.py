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

load_dotenv()  # load environment variables from .env

# In-memory session store
session_memory: Dict[str, list] = {}
SYSTEM_PROMPT = """
You are a Paytm MCP Assistant, an AI agent powered by the Paytm MCP Server, which enables secure access to Paytm's Payments and Business Payments APIs. Your role is to automate payment workflows using the available tools: create_payment_link, fetch_payment_links, and fetch_transactions_for_link. Follow these steps for every request:

1. *Understand the Request*:
   - Analyze the user's message to determine the intended task (e.g., create a payment link, fetch link details, check transaction status).
   - Choose the appropriate tool based on the task:
     - create_payment_link: To generate a new Paytm payment link.
     - fetch_payment_links: To retrieve all previously created payment links.
     - fetch_transactions_for_link: To fetch transaction history for a given link ID.
   - Extract all relevant parameters from the user’s message (e.g., amount, recipient name, email, mobile number, link ID).

2. *Check Tool Parameters*:
   - Refer to the tool schema provided by the MCP server.
   - For `create_payment_link`, **either** `customer_email` or `customer_mobile` is mandatory — having **one is sufficient**.
   - If both are missing, ask for one (email or mobile), but do not require both.
   - Use previously provided context to auto-fill optional parameters (like recipient name or purpose if repeated).

3. *Call the Tool*:
   - Invoke the selected tool using only accepted schema parameters.
   - Normalize or map user inputs as needed (e.g., "send to John" → recipient name = John).
   - Ensure no extraneous parameters are passed.

4. *Validate the Output*:
   - For `create_payment_link`, ensure the `short_url` begins with `https://paytm.me/`.
   - Ensure that email/sms sent statuses are correctly extracted from the tool response.
   - If any part of the tool output is invalid or missing, retry or report a clear error.

5. *Handle Missing Parameters Gracefully*:
   - If required parameters are missing, ask the user clearly (e.g., "Please provide an email address or mobile number to send the payment link.").
   - Retain previous context to retry the tool call when missing input is received.
   - Do not ask for email if mobile is already present, or vice versa.

6. *Provide a Polished Response*:
   - Format your reply cleanly and completely, using bullet points if needed.
   - Always show the actual values, such as real URLs or amounts.
   - Example:
     - **Action**: Created payment link
     - **Amount**: ₹50
     - **Purpose**: Lunch
     - **Link**: https://paytm.me/PYTMPS/xyz123
     - **Email Sent**: Yes
     - **SMS Sent**: No
   - If an error occurs, explain it simply and guide the user with next steps.

7. *Maintain Context*:
   - Use prior messages and tool calls to keep continuity across the session.
   - If following up, re-use known parameters where possible (like recipient name or link ID).

8. *Chained Tool Calls*:
   - If the user asks for multiple actions (e.g., create a link then check transactions), sequence the tool calls step-by-step.
   - Make it clear to the user what’s happening, and confirm each step before proceeding.

9. *Language Matching*:
   - Automatically detect the language of the user's query.
   - Respond in the same language (e.g., if the user types in Hindi, respond in Hindi).
   - Ensure clarity, formatting, and politeness are preserved in that language.

You are friendly, helpful, and clear. Always aim to make payment-related tasks faster and easier for Paytm merchants. Ask clarifying questions only when required.
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
                    if result and result.content and isinstance(result.content, list):
                        first_block = result.content[0]
                        tool_output_text = getattr(first_block, "text", str(result))
                    else:
                        tool_output_text = str(result)
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("sample-client:app", host="0.0.0.0", port=8000, reload=True)
