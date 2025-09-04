import os
from openai import OpenAI

def main():
    # Initialize OpenAI client with Groq API endpoint
    client = OpenAI(
        api_key=os.environ.get("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )
    
    try:
        # Create a chat completion using an open-source model
        # Popular OSS models available on Groq: llama3-8b-8192, llama3-70b-8192, mixtral-8x7b-32768
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "What are the benefits of using open-source AI models?"
                }
            ],
            model="openai/gpt-oss-120b",  # Using Llama 3 8B model
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            stream=False
        )
        
        # Print the response
        print("Response from Groq OSS model:")
        print("-" * 40)
        print(chat_completion.choices[0].message.content)
        print("-" * 40)
        print(f"Model used: {chat_completion.model}")
        print(f"Tokens used: {chat_completion.usage.total_tokens if chat_completion.usage else 'N/A'}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your GROQ_API_KEY environment variable is set correctly.")

def chat_interactive():
    """Interactive chat function"""
    client = OpenAI(
        api_key=os.environ.get("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )
    
    print("Interactive chat with Groq OSS model (type 'quit' to exit)")
    print("=" * 50)
    
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
            
        messages.append({"role": "user", "content": user_input})
        
        try:
            response = client.chat.completions.create(
                messages=messages,
                model="openai/gpt-oss-120b",
                temperature=0.7,
                max_tokens=1024,
                stream=False
            )
            
            assistant_message = response.choices[0].message.content
            print(f"\nAssistant: {assistant_message}")
            
            messages.append({"role": "assistant", "content": assistant_message})
            
        except Exception as e:
            print(f"Error: {e}")
            break

def stream_example():
    """Example with streaming response"""
    client = OpenAI(
        api_key=os.environ.get("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )
    
    print("Streaming response example:")
    print("-" * 30)
    
    try:
        stream = client.chat.completions.create(
            messages=[
                {"role": "user", "content": "Write a short story about AI and creativity."}
            ],
            model="llama3-8b-8192",
            stream=True,
            temperature=0.8,
            max_tokens=500
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="")
        
        print("\n" + "-" * 30)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Check if API key is set
    if not os.environ.get("GROQ_API_KEY"):
        print("Please set your GROQ_API_KEY environment variable.")
        print("Example: export GROQ_API_KEY='your-api-key-here'")
        exit(1)
    
    print("Choose an option:")
    print("1. Single query example")
    print("2. Interactive chat")
    print("3. Streaming response example")
    
    choice = input("Enter choice (1-3): ")
    
    if choice == "1":
        main()
    elif choice == "2":
        chat_interactive()
    elif choice == "3":
        stream_example()
    else:
        print("Invalid choice. Running single query example...")
        main()