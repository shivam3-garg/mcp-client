import asyncio
import json
import os
from typing import Optional, Dict
from contextlib import AsyncExitStack
import time
from datetime import datetime, timedelta

from mcp import ClientSession
from mcp.client.sse import sse_client

from openai import OpenAI
from openai.types import Completion
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import base64
import httpx
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

CRITICAL GUARDRAILS - STRICTLY ENFORCE:
You EXCLUSIVELY handle queries related to Paytm payments, refunds, payment links, transactions, orders, payment processing and related payment contenxt

WHAT YOU CANNOT HELP WITH:
- General programming questions unrelated to payments
- Personal advice or casual conversations
- Weather, news, or general knowledge
- Other services or APIs not related to payments
- Non-payment related coding help
- Any topic outside of Paytm payment processing


Remember: You are a specialized assistant for Paytm payments only. Stay focused on your domain.If question outside domain is asked tell user respectfully that you can only answer Paytm payments specific.
"""

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.available_tools = []
        self._streams_context = None
        self._session_context = None
        self.connected = False
        self.server_url = "https://payment-ol-mcp.onrender.com/sse"
        self.last_connection_attempt = None
        self.connection_retry_delay = 5  # seconds
        self.max_retries = 3
        self.connection_lock = asyncio.Lock()
        self.health_check_interval = 30  # seconds
        self.last_health_check = None

    async def connect_to_sse_server(self, server_url: str = None):
        """Connect to SSE server with retry logic"""
        if server_url:
            self.server_url = server_url
            
        async with self.connection_lock:
            try:
                logger.info(f"Connecting to MCP SSE server: {self.server_url}")
                
                # Clean up existing connections first
                await self._cleanup_connections()
                
                self._streams_context = sse_client(url=self.server_url)
                streams = await self._streams_context.__aenter__()
                logger.info(f"Streams established: {streams}")

                self._session_context = ClientSession(*streams)
                self.session: ClientSession = await self._session_context.__aenter__()

                logger.info("Initializing SSE client...")
                await self.session.initialize()
                logger.info("SSE client initialized successfully")

                await self.get_available_tools()
                self.connected = True
                self.last_connection_attempt = datetime.now()
                self.last_health_check = datetime.now()
                logger.info("MCP Client connected successfully")
                
            except Exception as e:
                logger.error(f"Failed to connect to MCP server: {str(e)}")
                logger.error(traceback.format_exc())
                self.connected = False
                await self._cleanup_connections()
                raise

    async def _cleanup_connections(self):
        """Clean up existing connections"""
        try:
            if self._session_context:
                await self._session_context.__aexit__(None, None, None)
                self._session_context = None
            if self._streams_context:
                await self._streams_context.__aexit__(None, None, None)
                self._streams_context = None
            self.session = None
        except Exception as e:
            logger.error(f"Error during connection cleanup: {str(e)}")

    async def ensure_connection(self):
        """Ensure connection is healthy, reconnect if necessary"""
        if not self.connected:
            logger.info("Connection not established, attempting to connect...")
            await self.connect_with_retry()
            return

        # Check if we need to perform a health check
        now = datetime.now()
        if (self.last_health_check is None or 
            (now - self.last_health_check).seconds > self.health_check_interval):
            
            if not await self.health_check():
                logger.warning("Health check failed, attempting to reconnect...")
                await self.connect_with_retry()

    async def health_check(self) -> bool:
        """Perform a simple health check on the connection"""
        try:
            if not self.session:
                return False
            
            # Try to list tools as a health check
            await self.session.list_tools()
            self.last_health_check = datetime.now()
            return True
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            self.connected = False
            return False

    async def connect_with_retry(self):
        """Connect with exponential backoff retry logic"""
        for attempt in range(self.max_retries):
            try:
                await self.connect_to_sse_server()
                return
            except Exception as e:
                if attempt < self.max_retries - 1:
                    delay = self.connection_retry_delay * (2 ** attempt)
                    logger.warning(f"Connection attempt {attempt + 1} failed, retrying in {delay}s: {str(e)}")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries} connection attempts failed")
                    raise

    async def cleanup(self):
        try:
            await self._cleanup_connections()
            self.connected = False
            logger.info("MCP Client cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

    async def get_available_tools(self):
        try:
            logger.info("Fetching available server tools...")
            response = await self.session.list_tools()
            logger.info(f"Connected to MCP server with tools: {[tool.name for tool in response.tools]}")

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
        except Exception as e:
            logger.error(f"Failed to get available tools: {str(e)}")
            raise

    async def call_openai(self, messages):
        try:
            return self.openai.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=1000,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, *messages],
                tools=self.available_tools
            )
        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            raise
    
    async def process_openai_response(self, response: Completion, session_id: str) -> str:
        try:
            messages = session_memory[session_id]

            for choice in response.choices:
                if choice.finish_reason == "tool_calls":
                    messages.append(choice.message)

                    for tool_call in choice.message.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)

                        logger.info(f"Calling tool {tool_name} with args {tool_args}")
                        
                        # Ensure connection before tool call
                        await self.ensure_connection()
                        
                        result = await self.session.call_tool(tool_name, tool_args)
                        logger.info(f"Tool response: {result}")
                        
                        if result and getattr(result, "content", None) and isinstance(result.content, list):
                            first_block = result.content[0]
                            tool_output_text = getattr(first_block, "text", "").strip()
                        else:
                            tool_output_text = ""
                        
                        if not tool_output_text:
                            tool_output_text = "Tool executed but no response was returned."
                        
                        logger.info(f"Tool Call ID: {tool_call.id}, Response: {tool_output_text}")

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_output_text,
                        })

                    response = await self.call_openai(messages)
                    return await self.process_openai_response(response, session_id)

                elif choice.finish_reason == "stop":
                    logger.info(f"Assistant response: {choice.message.content}")
                    messages.append(choice.message)
                    return choice.message.content

            return "Sorry, I couldn't complete that request."
        except Exception as e:
            logger.error(f"Error processing OpenAI response: {str(e)}")
            logger.error(traceback.format_exc())
            # Try to reconnect on tool call failure
            if "tool" in str(e).lower():
                logger.info("Tool call failed, attempting to reconnect...")
                try:
                    await self.connect_with_retry()
                except Exception:
                    pass
            raise

    async def process_query(self, query: str, session_id: str) -> str:
        try:
            # Ensure connection is healthy
            await self.ensure_connection()
                
            if session_id not in session_memory:
                session_memory[session_id] = []

            session_memory[session_id].append({"role": "user", "content": query})
            response = await self.call_openai(session_memory[session_id])
            return await self.process_openai_response(response, session_id)
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Try to recover from connection issues
            if not self.connected or "connection" in str(e).lower():
                try:
                    logger.info("Attempting to recover connection...")
                    await self.connect_with_retry()
                    # Retry the query once after reconnection
                    response = await self.call_openai(session_memory[session_id])
                    return await self.process_openai_response(response, session_id)
                except Exception as recovery_error:
                    logger.error(f"Failed to recover connection: {str(recovery_error)}")
                    return ("I'm experiencing connection issues with the payment server. "
                           "Please try again in a few moments.")
            
            return "I encountered an error processing your request. Please try again."

# Define FastAPI app
app = FastAPI(title="Paytm MCP Assistant")

# Add CORS middleware
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

# Global client instance
client = MCPClient()

@app.on_event("startup")
async def startup_event():
    try:
        await client.connect_with_retry()
        logger.info("MCP Client startup completed successfully")
    except Exception as e:
        logger.error(f"Failed to start MCP Client: {str(e)}")
        # Don't raise here to allow the server to start even if MCP connection fails

@app.on_event("shutdown")
async def shutdown_event():
    await client.cleanup()

@app.get("/health")
async def health_check():
    connection_healthy = await client.health_check() if client.session else False
    return {
        "status": "healthy" if connection_healthy else "degraded",
        "mcp_connected": client.connected,
        "available_tools": len(client.available_tools),
        "last_connection_attempt": client.last_connection_attempt.isoformat() if client.last_connection_attempt else None,
        "last_health_check": client.last_health_check.isoformat() if client.last_health_check else None
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        # Validate request
        if not request.message or not request.session_id:
            return JSONResponse(
                content={"error": "Message and session_id are required"}, 
                status_code=400
            )
        
        logger.info(f"Processing chat request for session {request.session_id}")
        
        # Process query with automatic connection recovery
        reply = await client.process_query(request.message, request.session_id)
        return JSONResponse(content={"reply": reply})
        
    except Exception as e:
        logger.error(f"Unhandled error in /chat: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            content={"error": "I'm experiencing technical difficulties. Please try again in a moment."}, 
            status_code=500
        )

@app.get("/connection/status")
async def connection_status():
    """Endpoint to check connection status"""
    return {
        "connected": client.connected,
        "server_url": client.server_url,
        "tools_available": len(client.available_tools),
        "last_connection": client.last_connection_attempt.isoformat() if client.last_connection_attempt else None
    }

@app.post("/connection/reconnect")
async def force_reconnect():
    """Endpoint to force reconnection"""
    try:
        await client.connect_with_retry()
        return {"status": "reconnected", "connected": client.connected}
    except Exception as e:
        logger.error(f"Force reconnect failed: {str(e)}")
        return JSONResponse(
            content={"error": "Reconnection failed", "details": str(e)}, 
            status_code=500
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("sample-client:app", host="0.0.0.0", port=8000, reload=True)