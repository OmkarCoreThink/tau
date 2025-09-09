#!/bin/bash

# Base URL (your custom endpoint)
BASE_URL="http://35.226.163.54:8002/v1"

# A simple prompt
PROMPT="What is the weather in Pune today?"

# Define some dummy tools (functions) in JSON
TOOLS='[
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get current weather for a city",
      "parameters": {
        "type": "object",
        "properties": {
          "city": {
            "type": "string",
            "description": "The name of the city"
          }
        },
        "required": ["city"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "get_time",
      "description": "Get current time for a city",
      "parameters": {
        "type": "object",
        "properties": {
          "city": {
            "type": "string",
            "description": "The name of the city"
          }
        },
        "required": ["city"]
      }
    }
  }
]'

# Make the API call
curl -s -X POST "$BASE_URL/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"any\",
    \"messages\": [
      {\"role\": \"user\", \"content\": \"$PROMPT\"}
    ],
    \"tools\": $TOOLS
  }" | jq
