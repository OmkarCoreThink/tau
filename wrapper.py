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
load_dotenv()

# Configuration
PORT = int(os.getenv("PORT", 7777))  # Different port to avoid conflicts

base_url = "https://api.groq.com/openai/v1"
api_key = os.environ.get("GROQ_API_KEY") 
reasoner_model = "openai/gpt-oss-120b"
base_model = "openai/gpt-oss-120b"

client = OpenAI(base_url=base_url,api_key=api_key)

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
    max_retries: int = 3,
    request_id: str = None,
):
    """
    Robust verifier that ensures proper tool calls are returned when needed.
    
    Args:
        messages: Conversation history
        candidate_calls: Candidate tool calls from reasoner
        scratchpad: Context and analysis
        unit_tasks: Current tasks
        tools_available: Available tools
        question: Question number (for logging)
        step: Step number (for logging)
        max_retries: Maximum retry attempts
    
    Returns:
        OpenAI-compatible response with tool calls or empty response
    """
    print(f"[VERIFIER] Starting verification")
    
    # Handle explicit "No tool call needed" case
    if candidate_calls == "No tool call needed":
        print("[VERIFIER] No tool call needed, returning empty response")
        return create_empty_response(unit_tasks)
    # Safety check for tools availability
    if not tools_available:
        print("[VERIFIER] No tools available, returning empty response")
        return create_empty_response(unit_tasks)
    
    new_system_prompt = """
        You are an assistant that verifies and (if needed) repairs a *single* candidate_tool_call against the user's requested task, current state, and available tools. Follow these steps **in order**:
        The idea is to solve the problem by doing one tool call at a time

        0. **SUCCESS CHECK** — If the requested task is already satisfied (i.e., candidate calls array is empty or prior calls already completed the task), return a success message and **do not** make any function calls.
        1. **ANALYZE** — Compare the candidate function call(s) to the user request and current state. Confirm the call is necessary and that it aligns with the available tools.
        2. **VALIDATE** — Verify the candidate call uses a known tool, correct function name, required parameters, valid parameter types, and correct syntax. You must take a look at the tools shared with you for the same.
        3. **REPAIR** — If validation fails, produce a corrected function call that fixes the errors (tool name, argument names/types, formatting). Represent all fixes only as function calls (no free-form answers). Do not ask questions or perform unrelated actions.
        4. **FINALIZE** — When the candidate call is valid and appropriate, return that single function call. If the task is fully satisfied, return a success message and **no** function calls.

        **Hard rules (must follow):**
        * Make sure to make a call when it is asked by the candidate call *
        * If the candidate call list is empty thoroughly analyzed if the task is done, and if it is actually done, return no tool call and give a detailed reasoning for the same.
        * Make **one and only one** function call per assistant response, and only call a different function than the candidate if you have a strong, explicit reason to do so.
        * Output **only** the required function call (or only the success message when satisfied). No extra commentary or explanation.
        * If you repair the candidate, return the corrected function call (one call per response).
        * Do not repeat prior successful calls unless absolutely required by the current state.
        * Do not take any actions not explicitly requested by the user.
        * Iterate: if further validation or actions are needed after your call, continue in subsequent responses (one function call per response) until the task is complete.

        **Output formats:**

        * If satisfied: plain success message (e.g., `{"status":"success","message":"Task satisfied — no function calls required."}`)
        * If calling a function: return exactly the function call payload required by the execution environment (one function call object only).

        Focus on correctness and precision. Take one deliberate step at a time.  
    """
    
    # Build verifier messages with proper error handling
    verifier_messages = []    
    verifier_messages.append({"role": "system", "content": new_system_prompt})

    # Safely build message history with error handling
    try:
        for usr_msg in messages:
            if usr_msg["role"] == "tool":
                verifier_messages.append({
                    "role": "tool",
                    "tool_call_id": usr_msg.get("tool_call_id"),
                    "content": usr_msg.get("content", ""),
                })
            elif "tool_calls" in usr_msg and usr_msg["tool_calls"]:
                verifier_messages.append({
                    "role": usr_msg["role"],
                    "content": usr_msg.get("content", ""),
                    "tool_calls": usr_msg["tool_calls"],
                })
            else:
                verifier_messages.append({
                    "role": usr_msg["role"],
                    "content": usr_msg.get("content", ""),
                })

        context_info = {
            "Summary": scratchpad.get("turns_info", "No summary available"),
            "current_state_analysis": scratchpad.get("current_state_analysis", "No state analysis available"),
            "candidate_tool_calls": candidate_calls,
        }
        
        verifier_messages.append({
            "role": "user", 
            "content": f"Stage Analysis and candidate function calls:\n\n{json.dumps(context_info, indent=2)}\n\nPlease do a thorough analysis and return the function calls."
        })
    except Exception as e:
        print(f"[VERIFIER ERROR] Failed to build messages: {e}")
        return create_empty_response(unit_tasks)

    # Single retry loop with smart fallback
    result = None
    last_error = None
    
    for attempt in range(max_retries):
        try:
            print(f"[VERIFIER] Attempt {attempt + 1}/{max_retries}")
            result = client.chat.completions.create(
                        model=base_model,
                        messages=verifier_messages,
                        tools=tools_available,
                        temperature=attempt*0.1,
                        max_tokens=1024,
                        stream=False,
                    )
            if result.choices[0].finish_reason == "tool_calls":
                tool_calls=result.choices[0].message.tool_calls
                print(format_tool_calls(tool_calls))
                is_valid, validation_message = validate_tool_calls(tool_calls, tools_available)
                if is_valid:
                    # Log verifier results
                    if request_id:
                        verifier_input = {
                            "candidate_calls": candidate_calls,
                            "scratchpad": scratchpad,
                            "unit_tasks": unit_tasks
                        }
                        log_verifier_results(request_id, verifier_input, result.model_dump())
                    
                    return result
                else:
                    print("Try Again")
            else:
                last_error = f"Unexpected finish_reason: {result.choices[0].finish_reason}"
        except Exception as e:
            print(f"[ERROR] Verifier call failed: {e}")
            user_message = f"""
            The previous model failed to make a correct function call with exact parameters. 
            It sometimes adds the function name in arguments.
            The error: {e}

            Your task is to make a correct function call based on the available tools 
            by extracting the correct arguments from the error.

            *Bad Example:*
            '{{"name": "<|constrain|>json", "arguments": {{"name": "cd", "arguments": {{"folder": "archive"}}}}}}'

            *Good Example:*
            '{{"name": "cd", "arguments": {{"folder": "archive"}}}}'

            *Note:* Do not use "<|constrain|>json" or any other placeholder 
            that is not in the tools list. Use the correct tool names as provided in the tools list.
            Focus on correctness and precision. Take one deliberate step at a time."""
            verifier_messages = [
            {"role": "system", "content": "You are a function calling assistant. You are provided with available tools, use them to give appropriate function calls"},
            {"role": "user", "content": user_message},
            ]
           
            result = client.chat.completions.create(
                        model=base_model,
                        messages=verifier_messages,
                        tools=tools_available,
                        temperature=0.0,
                        max_tokens=1024,
                        stream=False,
                    )
    
            # Check if we got a valid response
            finish_reason = result.choices[0].finish_reason
            if finish_reason in ["tool_calls", "stop"]:
                print(f"[VERIFIER] Got response with finish_reason: {finish_reason}")
                
                # Validate tool calls if present
                tool_calls = result.choices[0].message.tool_calls
                if tool_calls:
                    is_valid, validation_message = validate_tool_calls(tool_calls, tools_available)
                    if is_valid:
                        print(f"[VERIFIER] Valid tool calls: {format_tool_calls(tool_calls)}")
                        
                        # Log verifier results
                        if request_id:
                            verifier_input = {
                                "candidate_calls": candidate_calls,
                                "scratchpad": scratchpad,
                                "unit_tasks": unit_tasks
                            }
                            log_verifier_results(request_id, verifier_input, result.model_dump())
                        
                        return result
                    else:
                        print(f"[VERIFIER] Invalid tool calls: {validation_message}")
                        last_error = f"Invalid tool calls: {validation_message}"
                        # Try again with next attempt
                        continue
                else:
                    # No tool calls - accept as completion if content suggests it
                    content = result.choices[0].message.content or ""
                    if content.strip():  # Has some content
                        print("[VERIFIER] No tool calls, retrying")
                        last_error = "No tool calls in response"
                    else:
                        print("[VERIFIER] Empty response, retrying")
                        last_error = "Empty response with no tool calls"
                    continue
            else:
                last_error = f"Unexpected finish_reason: {finish_reason}"
                print(f"[VERIFIER] {last_error}")
                continue
                
        except Exception as e:
            last_error = str(e)
            print(f"[VERIFIER RETRYING] Attempt {attempt + 1} failed: {e}")
            continue
    
    # If we've exhausted all attempts, return empty response
    print(f"[VERIFIER] All {max_retries} attempts failed. Last error: {last_error}")
    return create_empty_response(unit_tasks)

def sanitize_messages(messages):
    allowed_keys = {"role", "content", "tool_calls", "tool_call_id", "name"}
    
    cleaned = []
    for msg in messages:
        cleaned_msg = {k: v for k, v in msg.items() if k in allowed_keys}
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
        #print(f"Request Headers: {dict(request.headers)}")
        #print(f"Raw Body: {json.dumps(raw_body, indent=2, ensure_ascii=False)}")
        print("=" * 80)
        
        messages = raw_body.get("messages", [])
        messages=sanitize_messages(messages)
        tools = raw_body.get("tools", [])  # Direct tools from API

        print(f"[DEBUG] Received {len(tools)} tools from request")
        print(f"[DEBUG] Total messages: {len(messages)}")
        
        # Print each message clearly
        #for i, msg in enumerate(messages):
        #    print(f"Message {i+1}: {json.dumps(msg, indent=2, ensure_ascii=False)}")
        #print("-" * 40)
        
        # If no tools available, pass directly to Groq model
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

