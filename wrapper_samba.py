from __future__ import annotations
import json,os
from main import run_srm_pipeline
from openai import OpenAI
import json, time
from utils import call_llm
from typing import Any, Dict, List, Optional
import uvicorn, os
from fastapi import Body, FastAPI, Request
from openai import OpenAI
from dotenv import load_dotenv
import logging
from datetime import datetime
from pathlib import Path
from check import check
load_dotenv()

# Configuration
PORT = int(os.getenv("PORT", 7777))  # Different port to avoid conflicts

base_url = "https://api.groq.com/openai/v1"
api_key = os.environ.get("GROQ_API_KEY") 
reasoner_model = "openai/gpt-oss-120b"
base_model = "Meta-Llama-3.3-70B-Instruct"

client = OpenAI(base_url=base_url,api_key=api_key)
samba_client = OpenAI(base_url="https://api.sambanova.ai/v1",api_key=os.getenv("SAMBANOVA_API_KEY"))
# Logging Configuration
LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

def setup_logging():
    """Set up logging configuration for API and SRM pipeline logs"""
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # API Logger
    api_logger = logging.getLogger('api_logger')
    api_logger.setLevel(logging.INFO)
    api_handler = logging.FileHandler(LOGS_DIR / 'api_logs.log')
    api_handler.setFormatter(detailed_formatter)
    api_logger.addHandler(api_handler)
    
    # SRM Pipeline Logger
    srm_logger = logging.getLogger('srm_logger')
    srm_logger.setLevel(logging.INFO)
    srm_handler = logging.FileHandler(LOGS_DIR / 'srm_pipeline_logs.log')
    srm_handler.setFormatter(detailed_formatter)
    srm_logger.addHandler(srm_handler)
    
    return api_logger, srm_logger

# Initialize loggers
api_logger, srm_logger = setup_logging()

def log_api_request_response(request_id: str, request_data: dict, response_data: dict, source: str = "unknown"):
    """Log API request and response data"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "request_id": request_id,
        "source": source,
        "request": {
            "messages_count": len(request_data.get("messages", [])),
            "messages": request_data.get("messages", [])
        },
        "response": {
            "finish_reason": response_data.get("choices", [{}])[0].get("finish_reason") if response_data.get("choices") else None,
            "content": response_data.get("choices", [{}])[0].get("message", {}).get("content") if response_data.get("choices") else None
        }
    }
    
    api_logger.info(f"API_REQUEST_RESPONSE: {json.dumps(log_entry, indent=2, ensure_ascii=False)}")

def log_srm_pipeline_results(request_id: str, scratchpad: dict, unit_tasks: str, candidate_calls: Any):
    """Log SRM pipeline results including scratchpad, unit_tasks, and candidate_calls"""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "request_id": request_id,
        "srm_results": {
            "scratchpad": scratchpad,
            "unit_tasks": unit_tasks,
            "candidate_calls": candidate_calls
        }
    }
    
    srm_logger.info(f"SRM_PIPELINE_RESULTS: {json.dumps(log_entry, indent=2, ensure_ascii=False, default=str)}")

def log_verifier_results(request_id: str, verifier_input: dict, verifier_output: dict):
    """Log verifier input and output"""
    # Remove tools information from verifier_input
    clean_verifier_input = {k: v for k, v in verifier_input.items() if k != "tools_available_count"}
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "request_id": request_id,
        "verifier_input": clean_verifier_input,
        "verifier_output": {
            "finish_reason": verifier_output.get("choices", [{}])[0].get("finish_reason") if verifier_output.get("choices") else None,
            "content": verifier_output.get("choices", [{}])[0].get("message", {}).get("content") if verifier_output.get("choices") else None
        }
    }
    
    srm_logger.info(f"VERIFIER_RESULTS: {json.dumps(log_entry, indent=2, ensure_ascii=False, default=str)}") 

def format_tool_calls(tool_calls):
    readable_output = []
    for idx, call in enumerate(tool_calls, 1):
        fn_name = call.function.name
        args = json.loads(call.function.arguments)
        
        formatted = f"{idx}. {fn_name}"
        for k, v in args.items():
            formatted += f"\n   - {k}: {v}"
        
        readable_output.append(formatted)
    return "\n\n".join(readable_output)

app = FastAPI(title="TC-SRM Direct Tools Wrapper")

def create_empty_response(unit_task):
    """Create a standardized empty response"""
    prompt = f"""
    Ask the user for more information which can help us to solve: {unit_task}
    If any information is assumed in the unit task, just ignore it and ask for more information.
    keep it short and precise. And I should look like a conversation you are having with a human.
    """
    response=call_llm(client, [{"role": "user", "content": prompt}], base_model)
    print(response)
    response = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": base_model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
    return response

def validate_tool_calls(tool_calls, tools_available):
    """Validate that tool calls are properly formatted and use available tools"""
    if not tool_calls:
        return True, "No tool calls to validate"
    
    if not tools_available:
        return False, "Tool calls present but no tools available"
    
    available_tool_names = {tool.get("function", {}).get("name") for tool in tools_available}
    
    for call in tool_calls:
        if not hasattr(call, 'function') or not hasattr(call.function, 'name'):
            return False, "Malformed tool call structure"
        
        if call.function.name not in available_tool_names:
            return False, f"Tool '{call.function.name}' not in available tools: {available_tool_names}"
        
        try:
            json.loads(call.function.arguments)
        except json.JSONDecodeError:
            return False, f"Invalid JSON in arguments for tool '{call.function.name}'"
    
    return True, "Tool calls valid"

def run_verifier(
    messages: List[Dict[str, str]], 
    candidate_calls: Any,
    scratchpad: Dict[str, Any],
    unit_tasks: str,
    tools_available: Optional[List[Dict[str, Any]]] = None,
    max_retries: int = 5,
    request_id: str = None,
):      
        messages.extend([{"role": "system", "content": f"Hints:\n {scratchpad['current_state_analysis']}"}])
        messages.extend([{"role": "system", "content": f"Unit Task\n{unit_tasks}"}])
        messages.extend([{"role": "system", "content": f"Possible Tool Call which you can make: \n{candidate_calls}"}])

        api_params = {
            "model": base_model,
            "messages": messages,
        }
        api_params["tools"] = tools_available

        response = samba_client.chat.completions.create(**api_params)
        
        # Check if response has tool calls
        message = response.choices[0].message
        has_tool_calls = hasattr(message, 'tool_calls') and message.tool_calls
        
        # Get response content
        content = message.content if message.content else ""
        
        # Prepare the final response with hardcoded model name
        final_response = {
            "id": response.id,
            "object": "chat.completion",
            "created": response.created,
            "model": "gpt-oss-120b",  # Hardcoded model name for response
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
            tool_calls=message.tool_calls
            print(format_tool_calls(tool_calls))
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
        
        return final_response
        
def sanitize_messages(messages):
    allowed_keys = {"role", "content", "tool_calls", "tool_call_id", "name"}

    cleaned = []
    for msg in messages:
        # coerce to strings (handle None)
        content_str = "" if msg.get("content") is None else str(msg.get("content"))
        reasoning_str = "" if msg.get("reasoning_content") is None else str(msg.get("reasoning_content"))

        # combine (space when both present)
        if content_str and reasoning_str:
            combined = content_str + " " + reasoning_str
        else:
            combined = content_str or reasoning_str  # whichever is non-empty (or "")

        # keep only allowed keys from original message
        cleaned_msg = {k: v for k, v in msg.items() if k in allowed_keys}

        # set the combined content (overwrites any existing content)
        cleaned_msg["content"] = combined

        cleaned.append(cleaned_msg)

    return cleaned

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, body: Any = Body(...)):

        raw_body = body if isinstance(body, dict) else json.loads(body or "{}")
        source = raw_body.get("source", "unknown").upper()
        print(f"[{source}] Incoming /v1/chat/completions payload")
        
        # Print entire request details
        print("=" * 80)
        print("COMPLETE USER REQUEST:")
        print("=" * 80)
        
        messages = raw_body.get("messages", [])
        messages=sanitize_messages(messages)
        tools = raw_body.get("tools", [])  # Direct tools from API

        print(f"[DEBUG] Received {len(tools)} tools from request")
        print(f"[DEBUG] Total messages: {len(messages)}")
       
        if not tools:
            print("[DEBUG] No tools available, passing directly to Groq model")
            try:
                direct_response = client.chat.completions.create(
                    model=base_model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=60000,
                    stream=False,
                )
                print("[DEBUG] Direct Groq response received successfully")
                print(f"[DEBUG] Response finish_reason: {direct_response.choices[0].message.content}")
                
                # Log direct API response
                direct_request_id = f"direct_{int(time.time() * 1000)}"
                log_api_request_response(
                    request_id=direct_request_id,
                    request_data=raw_body,
                    response_data=direct_response.model_dump(),
                    source=source
                )
                
                return direct_response
            except Exception as e:
                print(f"[ERROR] Direct Groq call failed: {e}")
                # Fallback to empty response
                return {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": base_model,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "I apologize, but I'm unable to process your request at the moment."},
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                }
        else:
            print("[DEBUG] Tools available, running SRM pipeline")
                
        print("[DEBUG] Running SRM pipeline...")
        
        # Generate request ID for logging
        request_id = f"srm_{int(time.time() * 1000)}"
        print(f"[DEBUG] Request ID: {request_id}")
        
        if len(messages) <=3:
            print("Checking for more information from user")
            return check(messages)
        scratchpad,unit_tasks,candidate_calls=run_srm_pipeline(client,messages,reasoner_model,tools,request_id)
        
        # Log SRM pipeline results
        log_srm_pipeline_results(request_id, scratchpad, unit_tasks, candidate_calls)
    
        final_decision = run_verifier(
            messages=messages,
            candidate_calls=candidate_calls,
            scratchpad=scratchpad,
            unit_tasks=unit_tasks,
            tools_available=tools,
            request_id=request_id,
        )
        
        # Log final API response
        log_api_request_response(
            request_id=request_id,
            request_data=raw_body,
            response_data=final_decision.model_dump() if hasattr(final_decision, 'model_dump') else final_decision,
            source=source
        )
        
        return final_decision
        
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, reload=False,workers=1)

