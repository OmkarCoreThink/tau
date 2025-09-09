from openai import OpenAI

def call_llm(client, prompt, model_name):
    import time
    start_time = time.time()
    
    max_retries = 3
    last_error = None
    
    for attempt in range(max_retries):
        try:
            attempt_start = time.time()     
            prompt.extend([{"role":"system","content":"You are not supposed to make any tool calls. Return only the final answer."}])       
            response = client.chat.completions.create(model=model_name, messages=prompt,tool_choice="none")
            elapsed = time.time() - attempt_start
            total_elapsed = time.time() - start_time
            return response.choices[0].message.content.strip() if response.choices else None
            
        except Exception as e:
            elapsed = time.time() - attempt_start
            print(f"[ERROR] Attempt {attempt+1} failed after {elapsed:.2f}s: {str(e)[:100]}...")
            last_error = e
            
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
