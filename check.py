from openai import OpenAI
import os 

def check(messages):

    PROMPT = f"""
    You would be given user request. Your task is to plan on how to solve the request and for that whatever information is needed from the user, you need to ask for that information in your response.
    Ask the user for detailed information about their request so that you can help them better.
    Make sure to ask each and every detailed information that you would need to solve the user request.
    """

    check_messages = []
    check_messages.append({"role": "system", "content": PROMPT})
    check_messages.append({"role": "user", "content": "MESSAGE TRANSCRIPT {messages}"})
    check_messages.append({"role": "system", "content": "You are not supposed to make any tool calls. Just ask the user for more and detailed information about their request."})

    base_url = "https://api.groq.com/openai/v1"
    api_key = os.environ.get("GROQ_API_KEY")
    client = OpenAI(base_url=base_url,api_key=api_key)
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
    )
    message = response.choices[0].message
    content = message.content
    final_response = {
            "id": response.id,
            "object": "chat.completion",
            "created": response.created,
            "model": "gpt-4.1",  # Hardcoded model name for response
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": response.choices[0].finish_reason
            }],
            "usage": response.usage.model_dump() if response.usage else {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
    return final_response

