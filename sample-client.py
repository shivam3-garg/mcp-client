import asyncio
import json
import os
from typing import Optional, Dict, Any, List, Tuple
from contextlib import AsyncExitStack
import time
import re
import base64
import io
from datetime import datetime

from mcp import ClientSession
from mcp.client.sse import sse_client

from openai import OpenAI
from openai.types import Completion
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import traceback
import logging

# Chart generation imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import defaultdict
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# In-memory session store
session_memory: Dict[str, list] = {}

SYSTEM_PROMPT = """You are a Paytm MCP Assistant, an AI agent powered by the Paytm MCP Server, which enables secure access to Paytm's Payments and Business Payments APIs. Your role is to automate payment workflows using the available tools: create_payment_link, fetch_payment_links, fetch_transactions_for_link, initiate_refund, check_refund_status, fetch_refund_list, and fetch_order_list.More actions

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
- Use clean markdown formatting. 
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

When users ask for graphs, charts, or visualizations of payment data, first fetch the relevant data using the appropriate tools, then indicate that a chart will be generated.
"""

class ChartGenerator:
    @staticmethod
    def extract_payment_data(tool_responses: List[str]) -> List[Dict]:
        """Extract payment data from tool responses"""
        all_data = []
        
        for response in tool_responses:
            try:
                # Try to parse as JSON
                if response.strip().startswith('{') or response.strip().startswith('['):
                    data = json.loads(response)
                    
                    # Handle different response structures
                    if isinstance(data, dict):
                        if 'data' in data and isinstance(data['data'], list):
                            all_data.extend(data['data'])
                        elif 'orders' in data and isinstance(data['orders'], list):
                            all_data.extend(data['orders'])
                        elif 'transactions' in data and isinstance(data['transactions'], list):
                            all_data.extend(data['transactions'])
                        elif 'links' in data and isinstance(data['links'], list):
                            all_data.extend(data['links'])
                        else:
                            all_data.append(data)
                    elif isinstance(data, list):
                        all_data.extend(data)
                        
            except json.JSONDecodeError:
                # Try to extract data using regex patterns
                patterns = [
                    r'"amount":\s*(\d+\.?\d*)',
                    r'"created_time":\s*"([^"]+)"',
                    r'"date":\s*"([^"]+)"',
                    r'"timestamp":\s*"([^"]+)"'
                ]
                
                amounts = re.findall(patterns[0], response)
                dates = re.findall(patterns[1], response) or re.findall(patterns[2], response) or re.findall(patterns[3], response)
                
                if amounts and dates:
                    for i, (amount, date) in enumerate(zip(amounts, dates)):
                        all_data.append({
                            'amount': float(amount),
                            'date': date,
                            'id': f'extracted_{i}'
                        })
                        
        return all_data

    @staticmethod
    def create_amount_chart(data: List[Dict]) -> str:
        """Create a chart showing amounts over time"""
        if not data:
            return None
            
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Extract and sort data
        chart_data = []
        for item in data:
            amount = None
            date = None
            
            # Extract amount
            if 'amount' in item:
                amount = float(item['amount'])
            elif 'total_amount' in item:
                amount = float(item['total_amount'])
            elif 'paid_amount' in item:
                amount = float(item['paid_amount'])
                
            # Extract date
            if 'created_time' in item:
                date = item['created_time']
            elif 'date' in item:
                date = item['date']
            elif 'timestamp' in item:
                date = item['timestamp']
                
            if amount is not None and date:
                chart_data.append((date, amount))
        
        if not chart_data:
            return None
            
        # Sort by date
        chart_data.sort(key=lambda x: x[0])
        
        dates, amounts = zip(*chart_data)
        
        # Create the chart
        ax.plot(dates, amounts, marker='o', linewidth=2, markersize=6, 
                color='#00BAF2', markerfacecolor='#FF6B35')
        
        ax.set_title('Payment Amounts Over Time', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Amount (₹)', fontsize=12)
        
        # Format y-axis for currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x:.2f}'))
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        
        chart_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        return chart_base64

    @staticmethod
    def create_summary_chart(data: List[Dict]) -> str:
        """Create a summary chart (bar chart of daily totals)"""
        if not data:
            return None
            
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Group data by date
        daily_totals = defaultdict(float)
        
        for item in data:
            amount = None
            date = None
            
            # Extract amount
            if 'amount' in item:
                amount = float(item['amount'])
            elif 'total_amount' in item:
                amount = float(item['total_amount'])
            elif 'paid_amount' in item:
                amount = float(item['paid_amount'])
                
            # Extract date
            if 'created_time' in item:
                date = item['created_time'][:10]  # Get date part only
            elif 'date' in item:
                date = item['date'][:10]
            elif 'timestamp' in item:
                date = item['timestamp'][:10]
                
            if amount is not None and date:
                daily_totals[date] += amount
        
        if not daily_totals:
            return None
            
        # Sort by date
        sorted_data = sorted(daily_totals.items())
        dates, amounts = zip(*sorted_data)
        
        # Create bar chart
        bars = ax.bar(dates, amounts, color='#00BAF2', alpha=0.8, edgecolor='#FF6B35', linewidth=1)
        
        # Add value labels on bars
        for bar, amount in zip(bars, amounts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(amounts)*0.01,
                   f'₹{amount:.2f}', ha='center', va='bottom', fontsize=10)
        
        ax.set_title('Daily Payment Totals', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Total Amount (₹)', fontsize=12)
        
        # Format y-axis for currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x:.2f}'))
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
        
        # Adjust layout
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        
        chart_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        return chart_base64

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.available_tools = []
        self._streams_context = None
        self._session_context = None
        self.connected = False
        self.server_url = ""
        self.last_connection_time = 0
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 3
        self.chart_generator = ChartGenerator()
        self.recent_tool_responses = []  # Store recent tool responses for chart generation

    async def connect_to_sse_server(self, server_url: str):
        self.server_url = server_url
        return await self._connect_with_retry()

    async def _connect_with_retry(self):
        """Connect with automatic retry logic"""
        for attempt in range(self.max_reconnect_attempts):
            try:
                logger.info(f"Connecting to MCP SSE server (attempt {attempt + 1}/{self.max_reconnect_attempts}): {self.server_url}")
                
                # Clean up any existing connections
                await self._cleanup_connection()
                
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
                self.last_connection_time = time.time()
                self.reconnect_attempts = 0
                logger.info("MCP Client connected successfully")
                return True
                
            except Exception as e:
                logger.error(f"Connection attempt {attempt + 1} failed: {str(e)}")
                await self._cleanup_connection()
                
                if attempt < self.max_reconnect_attempts - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Waiting {wait_time}s before retry...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error("Max reconnection attempts reached")
                    raise

        return False

    async def _cleanup_connection(self):
        """Clean up existing connection"""
        try:
            if self._session_context:
                await self._session_context.__aexit__(None, None, None)
                self._session_context = None
            if self._streams_context:
                await self._streams_context.__aexit__(None, None, None)
                self._streams_context = None
            self.session = None
            self.connected = False
        except Exception as e:
            logger.error(f"Error during connection cleanup: {str(e)}")

    async def cleanup(self):
        await self._cleanup_connection()
        logger.info("MCP Client cleaned up")

    async def ensure_connected(self):
        """Ensure we have a valid connection, reconnect if needed"""
        if not self.connected:
            logger.info("Connection lost, attempting to reconnect...")
            await self._connect_with_retry()
        
        # Check if connection is stale (older than 5 minutes)
        if time.time() - self.last_connection_time > 300:
            logger.info("Connection appears stale, refreshing...")
            await self._connect_with_retry()

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
    
    async def call_tool_with_retry(self, tool_name: str, tool_args: dict):
        """Call MCP tool with automatic retry on connection errors"""
        for attempt in range(2):  # Try twice
            try:
                await self.ensure_connected()
                result = await self.session.call_tool(tool_name, tool_args)
                return result
            except Exception as e:
                error_msg = str(e).lower()
                if "closed" in error_msg or "connection" in error_msg:
                    logger.warning(f"Connection error on attempt {attempt + 1}, retrying...")
                    self.connected = False
                    if attempt == 0:  # Only retry once
                        await asyncio.sleep(1)
                        continue
                raise e

    def should_generate_chart(self, query: str) -> bool:
        """Check if the query requests a chart/graph"""
        chart_keywords = [
            'graph', 'chart', 'visualiz', 'plot', 'show', 'display',
            'trend', 'analytics', 'dashboard', 'visual', 'diagram'
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in chart_keywords)

    def generate_chart_from_data(self, chart_type: str = "line") -> Optional[str]:
        """Generate chart from recent tool responses"""
        try:
            data = self.chart_generator.extract_payment_data(self.recent_tool_responses)
            if not data:
                return None
                
            if chart_type == "bar" or "daily" in " ".join(self.recent_tool_responses).lower():
                return self.chart_generator.create_summary_chart(data)
            else:
                return self.chart_generator.create_amount_chart(data)
                
        except Exception as e:
            logger.error(f"Error generating chart: {str(e)}")
            return None
        
    async def process_openai_response(self, response: Completion, session_id: str) -> Dict[str, Any]:
        try:
            messages = session_memory[session_id]
            chart_data = None

            for choice in response.choices:
                if choice.finish_reason == "tool_calls":
                    messages.append(choice.message)
                    self.recent_tool_responses = []  # Reset for new tool calls

                    for tool_call in choice.message.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)

                        logger.info(f"Calling tool {tool_name} with args {tool_args}")
                        
                        try:
                            result = await self.call_tool_with_retry(tool_name, tool_args)
                            logger.info(f"Tool response: {result}")
                        except Exception as e:
                            logger.error(f"Tool call failed after retries: {str(e)}")
                            tool_output_text = f"Error calling {tool_name}: {str(e)}"
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": tool_output_text,
                            })
                            continue
                        
                        if result and getattr(result, "content", None) and isinstance(result.content, list):
                            first_block = result.content[0]
                            tool_output_text = getattr(first_block, "text", "").strip()
                        else:
                            tool_output_text = ""
                        
                        if not tool_output_text:
                            tool_output_text = "Tool executed but no response was returned."
                        
                        # Store tool response for potential chart generation
                        self.recent_tool_responses.append(tool_output_text)
                        
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
                    
                    # Check if we should generate a chart
                    if self.recent_tool_responses:
                        last_user_message = ""
                        for msg in reversed(messages):
                            if msg.get("role") == "user":
                                last_user_message = msg.get("content", "")
                                break
                        
                        if self.should_generate_chart(last_user_message):
                            chart_data = self.generate_chart_from_data()
                    
                    return {
                        "reply": choice.message.content,
                        "chart": chart_data
                    }

            return {"reply": "Sorry, I couldn't complete that request.", "chart": None}
        except Exception as e:
            logger.error(f"Error processing OpenAI response: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    async def process_query(self, query: str, session_id: str) -> Dict[str, Any]:
        try:
            await self.ensure_connected()
                
            if session_id not in session_memory:
                session_memory[session_id] = []

            session_memory[session_id].append({"role": "user", "content": query})
            response = await self.call_openai(session_memory[session_id])
            return await self.process_openai_response(response, session_id)
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "reply": f"I'm experiencing connection issues with the payment service. Please try again in a moment. Error: {str(e)}",
                "chart": None
            }

# Define FastAPI app
app = FastAPI(title="MCP Tool Assistant with Charts")

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

class ChatResponse(BaseModel):
    reply: str
    chart: Optional[str] = None

# Global client instance
client = MCPClient()

@app.on_event("startup")
async def startup_event():
    try:
        await client.connect_to_sse_server(server_url="https://payment-ol-mcp.onrender.com/sse")
        logger.info("MCP Client startup completed successfully")
    except Exception as e:
        logger.error(f"Failed to start MCP Client: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    await client.cleanup()

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "mcp_connected": client.connected,
        "available_tools": len(client.available_tools),
        "last_connection": client.last_connection_time,
        "server_url": client.server_url
    }

@app.post("/reconnect")
async def force_reconnect():
    """Force reconnection to MCP server"""
    try:
        await client._connect_with_retry()
        return {"status": "reconnected", "connected": client.connected}
    except Exception as e:
        return {"status": "failed", "error": str(e)}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        if not request.message or not request.session_id:
            return JSONResponse(
                content={"error": "Message and session_id are required"}, 
                status_code=400
            )
        
        logger.info(f"Processing chat request for session {request.session_id}")
        result = await client.process_query(request.message, request.session_id)
        
        return JSONResponse(content={
            "reply": result["reply"],
            "chart": result["chart"]
        })
        
    except Exception as e:
        logger.error(f"Unhandled error in /chat: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            content={
                "reply": f"Service temporarily unavailable. Please try again.",
                "chart": None,
                "error": str(e)
            }, 
            status_code=503
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("sample-client:app", host="0.0.0.0", port=8000, reload=True)