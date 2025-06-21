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
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
from matplotlib.dates import DateFormatter
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()  # load environment variables from .env

# In-memory session store
session_memory: Dict[str, list] = {}
session_chart_data: Dict[str, dict] = {}


# Set style for better looking charts
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def store_session_data(tool_response, session_id):
    """Store data from tool response for future chart generation"""
    try:
        # Parse order data from text format
        data = parse_order_text_to_json(tool_response)
        
        if data:
            if session_id not in session_chart_data:
                session_chart_data[session_id] = {}
            session_chart_data[session_id]['data'] = data
            session_chart_data[session_id]['timestamp'] = datetime.now()
            logger.info(f"Stored chart data for session {session_id}: {len(data)} records")
    except Exception as e:
        logger.error(f"Error storing session data: {str(e)}")

def generate_chart(data, chart_type, x_field, y_field, title="Chart"):
    """Generate chart from data and return base64 encoded image"""
    try:
        # Convert data to DataFrame
        df = pd.DataFrame(data)
        
        # Sort by date if x_field is date
        if 'date' in x_field.lower():
            df[x_field] = pd.to_datetime(df[x_field])
            df = df.sort_values(x_field)
        
        # Create figure with better styling
        plt.figure(figsize=(12, 6))
        
        if chart_type.lower() == 'line':
            plt.plot(df[x_field], df[y_field], marker='o', linewidth=2, markersize=6)
        elif chart_type.lower() == 'bar':
            plt.bar(range(len(df)), df[y_field], alpha=0.8)
            plt.xticks(range(len(df)), df[x_field], rotation=45)
        elif chart_type.lower() == 'pie':
            plt.pie(df[y_field], labels=df[x_field], autopct='%1.1f%%', startangle=90)
        elif chart_type.lower() == 'scatter':
            plt.scatter(df[x_field], df[y_field], alpha=0.7, s=60)
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        
        if chart_type.lower() != 'pie':
            plt.xlabel(x_field.replace('_', ' ').title(), fontsize=12)
            plt.ylabel(y_field.replace('_', ' ').title(), fontsize=12)
            if chart_type.lower() != 'bar':
                plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.grid(True, alpha=0.3)
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', facecolor='white')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    except Exception as e:
        logger.error(f"Chart generation failed: {str(e)}")
        return None

def generate_chart_config(data, chart_type, x_field, y_field, title="Chart"):
    """Generate chart configuration instead of image"""
    try:
        # Convert data to simple format for frontend
        chart_data = []
        for item in data:
            chart_data.append({
                "x": item.get(x_field, ''),
                "y": float(item.get(y_field, 0)) if isinstance(item.get(y_field, 0), (int, float, str)) else 0,
                "label": item.get(x_field, '')
            })
        
        return {
            "type": chart_type,
            "data": chart_data,
            "config": {
                "title": title,
                "xLabel": x_field.replace('_', ' ').title(),
                "yLabel": y_field.replace('_', ' ').title(),
                "xField": x_field,
                "yField": y_field
            }
        }
    except Exception as e:
        logger.error(f"Chart config generation failed: {str(e)}")
        return None

def should_generate_chart(query, tool_response, session_id=None):
    """Determine if a chart should be generated based on query and response"""
    chart_keywords = ['chart', 'graph', 'plot', 'visualize', 'show trends', 'visual', 'draw', 'create a chart']
    reference_keywords = ['above data', 'previous data', 'that data', 'this data', 'earlier data']
    
    has_chart_request = any(keyword in query.lower() for keyword in chart_keywords)
    has_data_reference = any(keyword in query.lower() for keyword in reference_keywords)
    has_order_data = 'Order ID:' in tool_response or 'order' in tool_response.lower()
    has_session_data = session_id and session_id in session_chart_data
    
    # Debug logging
    logger.info(f"Chart check - Query: '{query}', Has chart request: {has_chart_request}, Has order data: {has_order_data}")
    
    return has_chart_request and (has_order_data or (has_data_reference and has_session_data))

def detect_chart_fields(data, query):
    """Detect appropriate fields for chart based on data and query"""
    if not data:
        return 'date', 'amount'
    
    # Get available fields
    fields = list(data[0].keys()) if data else []
    
    # Default fields
    x_field = 'date'
    y_field = 'amount'
    
    # Look for date field
    date_fields = [f for f in fields if 'date' in f.lower() or 'time' in f.lower()]
    if date_fields:
        x_field = date_fields[0]
    
    # Look for amount field
    amount_fields = [f for f in fields if 'amount' in f.lower() or 'value' in f.lower() or 'total' in f.lower()]
    if amount_fields:
        y_field = amount_fields[0]
    
    return x_field, y_field

def extract_chart_data(tool_response, query, session_id=None):
    """Extract data for chart generation from tool response or session data"""
    try:
        # Case 1: Parse text format order data
        data = parse_order_text_to_json(tool_response)
        
        # Case 2: Chart request referencing previous data
        if not data and session_id and session_id in session_chart_data:
            data = session_chart_data[session_id].get('data', [])
        
        if not data:
            return None
            
        # Determine chart type and fields based on query
        chart_type = 'bar'  # default
        if 'line' in query.lower() or 'trend' in query.lower():
            chart_type = 'line'
        elif 'pie' in query.lower():
            chart_type = 'pie'
        elif 'scatter' in query.lower():
            chart_type = 'scatter'
            
        # Smart field detection
        x_field, y_field = detect_chart_fields(data, query)
        
        return {
            'data': data,
            'chart_type': chart_type,
            'x_field': x_field,
            'y_field': y_field,
            'title': f'{y_field.replace("_", " ").title()} vs {x_field.replace("_", " ").title()}'
        }
    except Exception as e:
        logger.error(f"Error extracting chart data: {str(e)}")
    return None

def parse_order_text_to_json(text_response):
    """Parse the text format order response into JSON array"""
    try:
        import re
        from datetime import datetime
        
        orders = []
        # Split by the separator line
        order_blocks = text_response.split('--------------------------------------------------')
        
        for block in order_blocks:
            if 'Order ID:' not in block:
                continue
                
            order = {}
            
            # Extract Order ID
            order_id_match = re.search(r'Order ID:\s*(\S+)', block)
            if order_id_match:
                order['order_id'] = order_id_match.group(1)
            
            # Extract Amount (remove ₹ symbol and convert to float)
            amount_match = re.search(r'Amount:\s*₹?([0-9.]+)', block)
            if amount_match:
                order['amount'] = float(amount_match.group(1))
            
            # Extract Created Time and convert to date
            created_match = re.search(r'Created Time:\s*([0-9-]+ [0-9:]+)', block)
            if created_match:
                try:
                    dt = datetime.strptime(created_match.group(1), '%Y-%m-%d %H:%M:%S')
                    order['date'] = dt.strftime('%Y-%m-%d')
                    order['created_time'] = created_match.group(1)
                except:
                    order['date'] = created_match.group(1).split(' ')[0]
            
            # Extract other fields
            status_match = re.search(r'Status:\s*(\S+)', block)
            if status_match:
                order['status'] = status_match.group(1)
                
            payment_mode_match = re.search(r'Payment Mode:\s*([^\n]+)', block)
            if payment_mode_match:
                order['payment_mode'] = payment_mode_match.group(1).strip()
            
            if order:  # Only add if we extracted some data
                orders.append(order)
        
        logger.info(f"Parsed {len(orders)} orders from text response")
        return orders
        
    except Exception as e:
        logger.error(f"Error parsing order text: {str(e)}")
        return []

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

7. Chart Generation:
- When users request charts, graphs, plots, or visualizations, automatically generate visual representations of the data.
- Supported chart types: line, bar, pie, scatter.
- Common requests: "show chart of orders", "plot sales trends", "visualize refund amounts", etc.
- Always provide both tabular data and visual chart when visualization is requested.
- Charts work best with numerical data from orders, refunds, and transaction amounts.

8. Maintain Context:
- Use prior messages to infer missing info.
- Remember recent link IDs, recipient names, etc., for follow-up questions.

9. Multi-Step or Chained Requests:
- If user intent requires multiple tools (e.g., refund + status check), sequence tool calls accordingly.
- Make it clear to the user what’s happening, and confirm each step before proceeding.

10. Language Matching:
- For **each user message**, detect the language used (e.g., Hindi, English, Hinglish).
- Respond in **that same language**, regardless of what language was used earlier in the session.
- his ensures users can switch freely between languages (e.g., start in English, switch to Hindi, and back).
- Maintain clarity and formatting (bullets, markdown, labels) regardless of the language used.

11. Date Parameters:
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

                        # Check if chart should be generated - find the last user message
                        user_query = ""
                        for msg in reversed(messages):
                            if isinstance(msg, dict) and msg.get('role') == 'user':
                                user_query = msg.get('content', '')
                                break
                            elif hasattr(msg, 'role') and msg.role == 'user':
                                user_query = msg.content
                                break
                        
                        # Store data for future chart generation
                        store_session_data(tool_output_text, session_id)

                        # Check if chart should be generated
# Check if chart should be generated
                        if should_generate_chart(user_query, tool_output_text, session_id):
                            logger.info(f"Generating chart for query: {user_query}")
                            chart_info = extract_chart_data(tool_output_text, user_query, session_id)
                            if chart_info:
                                chart_config = generate_chart_config(
                                    chart_info['data'],
                                    chart_info['chart_type'],
                                    chart_info['x_field'],
                                    chart_info['y_field'],
                                    chart_info['title']
                                )
                                if chart_config:
                                    tool_output_text += f"\n\n**CHART_CONFIG:**{json.dumps(chart_config)}"
                                    logger.info("Chart config generated successfully")

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

class ChatResponse(BaseModel):
    reply: str
    chart_config: Optional[dict] = None

def extract_chart_from_response(response_text):
    """Extract chart config from response if present"""
    if "**CHART_CONFIG:**" in response_text:
        parts = response_text.split("**CHART_CONFIG:**")
        try:
            chart_config = json.loads(parts[1].strip())
            return parts[0].strip(), chart_config
        except json.JSONDecodeError:
            return response_text, None
    return response_text, None

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
        text_reply, chart_config = extract_chart_from_response(reply)
        response = {"reply": text_reply}
        if chart_config:
            response["chart_config"] = chart_config
        return JSONResponse(content=response)
        
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