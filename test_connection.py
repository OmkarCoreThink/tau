#!/usr/bin/env python3
"""
Quick OpenRouter connection test
"""
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def test_connection():
    api_key = os.environ.get("OPEN_ROUTER_API_KEY")
    
    print("🔍 Testing OpenRouter Connection")
    print("=" * 40)
    
    if not api_key:
        print("❌ OPEN_ROUTER_API_KEY not found!")
        print("Set it with: export OPEN_ROUTER_API_KEY='your_key'")
        return False
    
    print(f"✅ API Key: {api_key[:8]}...{api_key[-4:]}")
    
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )
    
    models_to_test = [
        "openai/gpt-4o-mini",
        "openai/gpt-3.5-turbo", 
        "anthropic/claude-3-haiku"
    ]
    
    for model in models_to_test:
        try:
            print(f"🧪 Testing {model}...")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Say 'OK' if you can hear me."}
                ],
                max_tokens=10,
                temperature=0.0,
                timeout=30
            )
            print(f"✅ {model}: {response.choices[0].message.content}")
            return True
        except Exception as e:
            print(f"❌ {model}: {e}")
    
    return False

if __name__ == "__main__":
    success = test_connection()
    if success:
        print("\n🎉 Connection working! Your SRM pipeline should work now.")
    else:
        print("\n💡 Try these steps:")
        print("1. Check your OpenRouter API key")
        print("2. Visit https://openrouter.ai/ to verify your account")
        print("3. Check if you have credits remaining")
