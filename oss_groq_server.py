from fastapi import FastAPI, Request, Body
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import os
import json
from openai import AsyncOpenAI
import re
import time
from loguru import logger
import asyncio
from dotenv import load_dotenv
load_dotenv()
app = FastAPI(title="GPT-OSS-120B Groq Tool Calling Server")

# Groq client setup using OpenAI client with Groq base URL
groq_client = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)

#groq_client = AsyncOpenAI(
#  base_url = "https://integrate.api.nvidia.com/v1",
#  api_key=os.getenv("NVIDIA_API_KEY")
#)

class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]
    temperature: Optional[float] = 0.0
    model: Optional[str] = None  # Will be ignored, hardcoded to gpt-oss-120b
    tools: Optional[List[Dict[str, Any]]] = None
    max_tokens: Optional[int] = 60000
    stream: Optional[bool] = False

@app.post("/v1/chat/completions")
async def complete(req: ChatRequest):
    request_start_time = time.time()
    request_id = f"req-{int(time.time() * 1000)}"
    time.sleep(1)
    try:
        # Hardcoded model configuration
        HARDCODED_MODEL = "openai/gpt-oss-120b"
        
        logger.info("=" * 80)
        logger.info(f"🚀 INCOMING REQUEST [{request_id}]")
        logger.info("=" * 80)
        logger.info(f"Model: {HARDCODED_MODEL} (hardcoded)")
        logger.info(f"Temperature: {req.temperature}")
        logger.info(f"Max Tokens: {req.max_tokens}")
        logger.info(f"Number of messages: {len(req.messages)}")
        logger.info(f"Number of tools: {len(req.tools) if req.tools else 0}")
        if req.model and req.model != HARDCODED_MODEL:
            logger.info(f"⚠️ Requested model '{req.model}' ignored, using hardcoded model")
        
        # Print messages for debugging
        for i, msg in enumerate(req.messages):
            logger.debug(f"Message {i+1}: {json.dumps(msg, indent=2, ensure_ascii=False)}")
        
        # Print tools if available
        if req.tools:
            logger.info("🔧 AVAILABLE TOOLS:")
            for i, tool in enumerate(req.tools):
                tool_name = tool.get('function', {}).get('name', 'Unknown')
                logger.info(f"  {i+1}. {tool_name}")
        
        logger.info("=" * 40)
        logger.info("📡 CALLING GROQ API")
        logger.info("=" * 40)
        
        api_start_time = time.time()
        
        # Prepare the API call parameters with hardcoded model
        api_params = {
            "model": HARDCODED_MODEL,
            "messages": req.messages,
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
            "stream": req.stream
        }
        
        # Add tools if provided
        if req.tools:
            api_params["tools"] = req.tools
            logger.info(f"🔧 Including {len(req.tools)} tools in API call")
        
        # Make the API call
        response = await groq_client.chat.completions.create(**api_params)
        api_time = time.time() - api_start_time
        
        logger.info("=" * 40)
        logger.info("📨 API RESPONSE RECEIVED")
        logger.info("=" * 40)
        logger.info(f"Response ID: {response.id}")
        logger.info(f"Response Model: {response.model}")
        logger.info(f"API Response time: {api_time:.2f}s")
        logger.info(f"Finish reason: {response.choices[0].finish_reason}")
        
        # Check if response has tool calls
        message = response.choices[0].message
        has_tool_calls = hasattr(message, 'tool_calls') and message.tool_calls
        
        if has_tool_calls:
            logger.info("🛠️ TOOL CALLS DETECTED")
            logger.info(f"Number of tool calls: {len(message.tool_calls)}")
            for i, tool_call in enumerate(message.tool_calls):
                logger.info(f"  Tool {i+1}: {tool_call.function.name}")
                logger.debug(f"  Arguments: {tool_call.function.arguments}")
        
        # Get response content
        content = message.content if message.content else ""
        logger.info(f"Response content length: {len(content)}")
        
        if content:
            logger.debug(f"Response content: {content}")
            
            # Clean response content (remove thinking tags if present)
            cleaned_content = re.sub(r"<think.*?>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE).strip()
            
            if len(cleaned_content) != len(content):
                logger.info(f"🧹 Removed thinking tags, content shortened by {len(content) - len(cleaned_content)} characters")
                content = cleaned_content
        
        # Prepare the final response with hardcoded model name
        final_response = {
            "id": response.id,
            "object": "chat.completion",
            "created": response.created,
            "model": "gpt-4.1-mini",  # Hardcoded model name for response
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": response.choices[0].finish_reason
            }],
            "usage": response.usage.model_dump() if response.usage else {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
        
        # Add tool calls to response if present
        if has_tool_calls:
            final_response["choices"][0]["message"]["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                }
                for tool_call in message.tool_calls
            ]
        
        total_time = time.time() - request_start_time
        
        logger.info("=" * 40)
        logger.info("✅ REQUEST COMPLETED")
        logger.info("=" * 40)
        logger.info(f"Request ID: {request_id}")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Final response content length: {len(content)}")
        if has_tool_calls:
            logger.info(f"Tool calls included: {len(message.tool_calls)}")
        logger.info("=" * 80)
        
        return final_response
        
    except Exception as e:
        total_time = time.time() - request_start_time
        logger.error("=" * 80)
        logger.error(f"❌ ERROR IN REQUEST [{request_id}]")
        logger.error("=" * 80)
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error(f"Total time before error: {total_time:.2f}s")
        logger.error("Full error details:", exc_info=True)
        logger.error("=" * 80)
        
        return {
            "error": {
                "message": str(e),
                "type": type(e).__name__,
                "request_id": request_id
            }
        }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "gpt-oss-120b-groq-server",
        "model": "openai/gpt-oss-120b",
        "max_tokens": 60000,
        "timestamp": time.time(),
        "groq_configured": bool(os.getenv("GROQ_API_KEY")),
        "provider": "Groq"
    }

@app.get("/models")
async def list_models():
    """List the single hardcoded model"""
    return {
        "models": ["gpt-oss-120b"],
        "hardcoded_model": "openai/gpt-oss-120b",
        "max_tokens": 60000,
        "provider": "Groq",
        "note": "This server only serves GPT-OSS-120B model via Groq. Model parameter in requests is ignored."
    }

@app.post("/v1/test-tools")
async def test_tools():
    """Test endpoint with sample tools"""
    sample_tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather information for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location to get weather for"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    sample_request = ChatRequest(
        messages=[
            {"role": "system", "content": "You are a helpful assistant with access to tools."},
            {"role": "user", "content": "What's the weather like in New York?"}
        ],
        model="gpt-oss-120b",  # Will be ignored anyway
        tools=sample_tools,
        temperature=0.0
    )
    
    return await complete(sample_request)

if __name__ == "__main__":
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 7000))  # Different port from OpenRouter server
    workers = int(os.getenv("WORKERS", 12))
    
    logger.info("🚀 Starting GPT-OSS-120B Groq Tool Calling Server")
    logger.info(f"Model: openai/gpt-oss-120b (hardcoded)")
    logger.info(f"Max Tokens: 60,000")
    logger.info(f"Provider: Groq")
    logger.info(f"Host: {host}")
    logger.info(f"Port: {port}")
    logger.info(f"Workers: {workers}")
    logger.info(f"Groq API Key configured: {bool(os.getenv('GROQ_API_KEY'))}")
    logger.info("ℹ️ This server only serves GPT-OSS-120B via Groq - model parameter in requests is ignored")
    
    if not os.getenv("GROQ_API_KEY"):
        logger.warning("⚠️ GROQ_API_KEY not found in environment variables!")
        logger.warning("Please set GROQ_API_KEY to use this server.")
    
    uvicorn.run(
        "oss_groq_server:app", 
        host=host, 
        port=port, 
        workers=workers,
        reload=False
    )
