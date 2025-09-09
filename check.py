from openai import OpenAI
import os 

def check(messages):

    PROMPT = f"""
    You would be given user request. Your task is to plan on how to solve the request and for that whatever information is needed from the user, you need to ask for that information in your response.
    Ask the user for detailed information about their request so that you can help them better.
    """

    check_messages = []
    check_messages.append({"role": "system", "content": PROMPT})
    check_messages.extend(messages)

    base_url = "https://api.groq.com/openai/v1"
    api_key = os.environ.get("GROQ_API_KEY")
    client = OpenAI(base_url=base_url,api_key=api_key)
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
    )

    return response

