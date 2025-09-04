from openai import OpenAI

def call_llm(client, prompt, model_name, allow_tool_calls=False):
    import time
    start_time = time.time()
    
    #print(f"[DEBUG] Request details: {len(prompt)} messages, max_tokens=32768")
    #print(f"[DEBUG] Using model: {model_name}, tool_calls: {allow_tool_calls}")
    
    max_retries = 3
    last_error = None
    
    for attempt in range(max_retries):
        try:
            attempt_start = time.time()
            #print(f"[DEBUG] Attempt {attempt+1}/{max_retries}: Making API call to {model_name}...")
            
            # Create request parameters
            request_params = {
                "model": model_name,
                "messages": prompt,
                "max_tokens": 32768,
                "temperature": 0.0,
                "timeout": 60  # 60 second timeout
            }
            
            messages = []
            for msg in request_params["messages"]:
                messages.append(msg)
            
            response = client.chat.completions.create(model=model_name, messages=messages)
            
            elapsed = time.time() - attempt_start
            total_elapsed = time.time() - start_time
            #print(f"[SUCCESS] API call completed in {elapsed:.2f}s (total: {total_elapsed:.2f}s)")
            return response.choices[0].message.content.strip() if response.choices else None
            
        except Exception as e:
            elapsed = time.time() - attempt_start
            print(f"[ERROR] Attempt {attempt+1} failed after {elapsed:.2f}s: {str(e)[:100]}...")
            print(request_params)
            last_error = e
            
            # If this was the last attempt, we'll raise the error
            if attempt == max_retries - 1:
                break
            
            # Wait a bit before retrying (exponential backoff)
            wait_time = 2 ** attempt  # 1s, 2s, 4s...
            print(f"[DEBUG] Waiting {wait_time}s before retry...")
            time.sleep(wait_time)
            continue
    
    # If we get here, all attempts failed
    total_elapsed = time.time() - start_time
    print(f"[ERROR] All {max_retries} attempts failed after {total_elapsed:.2f}s")
    print(f"[ERROR] Model: {model_name}")
    raise last_error
