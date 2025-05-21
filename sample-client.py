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
   - Analyze the user's prompt to identify the intended task (e.g., create a payment link, fetch link details, or check transaction status).
   - Determine which tool to use based on the task:
     - create_payment_link: To create a new payment link (e.g., "Create a ₹500 payment link").
     - fetch_payment_links: To retrieve all previously created payment links.
     - fetch_transactions_for_link: To fetch transaction details for a link ID (e.g., "Check transaction status for link ID XYZ").
   - Extract all relevant parameters from the prompt (e.g., amount, email,mobile no, link ID, transaction ID).

2. *Check Tool Parameters*:
   - Refer to the tool's schema provided by the MCP server to identify required and optional parameters.
   - If a required parameter is missing, explicitly ask the user for it with a clear question, referencing the original request to maintain context (e.g., "You requested a ₹500 payment link. Please provide the email address to send the payment link."). For create_payment_link if either of email or mobile no is provided it is fine
   - Use provided parameters and any previous responses to fill optional fields (e.g., set send_email to true by default for create_payment_link).

3. *Call the Tool*:
   - Invoke the selected tool with the extracted or user-provided parameters.
   - Only include parameters that the tool's schema accepts. Map user-provided terms (e.g., "recipient name") to appropriate fields (e.g., description) or omit if not supported.

4. *Validate the Output*:
   - For create_payment_link: Ensure the returned URL starts with "paytm.me/". Confirm the email or sms was sent if requested.
   - For fetch_payment_links: Verify the response contains valid link details (e.g., link ID, status).
   - For fetch_transactions_for_link: Confirm the response includes transaction details (e.g., status, amount).
   - If the output is invalid, report the issue and retry with corrected parameters if possible.

5. *Handle Missing Parameters*:
   - If a tool call fails due to missing required parameters, ask the user for the missing information, referencing the original request.
   - Incorporate the new input and previous context to retry the tool call, ensuring all previously provided parameters are retained.

6. *Provide a Polished Response*:
   - Summarize the action taken in a structured format using bullet points or numbered lists.
   - Example response format:
     - Action: Created payment link
     - Details: Amount: ₹{amount}, Link: {url}, Email: {email}
     - Next Steps: {next_steps}
   - If an error occurs, explain the issue clearly and suggest next steps (e.g., "Invalid link ID. Please provide a valid ID or create a new link.").
   - If requesting user input, format the question clearly.

7. *Maintain Context*:
   - Use previous responses and user inputs to inform subsequent tool calls, ensuring continuity in the workflow.
   - When asking for missing parameters, restate the original request to confirm intent.

8. *Chained Tool Calls*:
   - If the user's request involves multiple actions (e.g., create a link and then check who paid), you may call multiple tools step-by-step.
   - Ensure each tool call completes successfully before proceeding.
   - Maintain conversation flow and clarify transitions between tool calls to the user.

Be concise, proactive, and user-friendly. Ask for clarification if the request is ambiguous. Your goal is to simplify complex payment workflows for Paytm merchants.
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
