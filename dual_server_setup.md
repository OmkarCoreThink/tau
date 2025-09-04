# 🚀 Dual Server Setup for Tau-Bench

This setup uses two different endpoints for different parts of tau-bench evaluation:

## 📊 **Port Configuration:**

| Server | Port | Purpose | Model/Strategy |
|--------|------|---------|----------------|
| **wrapper.py** | 7777 | Main Agent | SRM Pipeline + Groq |
| **oss_open_router_server.py** | 7000 | User Strategy LLM | GPT-OSS-120B Direct |

## 🔧 **Environment Setup:**

```bash
# For Main Agent (wrapper.py - port 7777)
export GROQ_API_KEY="your_groq_api_key"

# For User Strategy (oss_open_router_server.py - port 7000) 
export OPEN_ROUTER_API_KEY="your_openrouter_api_key"
```

## 🚀 **Starting Both Servers:**

### Terminal 1 - Main Agent Server (Port 7777):
```bash
cd /home/jay/tc_reasoner
uvicorn wrapper:app --workers 2 --port 7777 --host 0.0.0.0
```

### Terminal 2 - User Strategy Server (Port 7000):
```bash
cd /home/jay/tc_reasoner  
python oss_open_router_server.py
# or
PORT=7000 WORKERS=4 python oss_open_router_server.py
```

## 🎯 **Tau-Bench Command Configuration:**

### Option 1: Using Custom Agent and User URLs
```bash
python run.py \
  --agent-strategy tool-calling \
  --env airline \
  --model gpt-4o \
  --model-provider openai \
  --user-model gpt-oss-120b \
  --user-model-provider openai \
  --user-strategy llm \
  --max-concurrency 5 \
  --agent-url http://localhost:7777/v1/chat/completions \
  --user-url http://localhost:7777/v1/chat/completions
```

### Option 2: Standard Configuration (if tau-bench supports endpoint routing)
```bash
python run.py \
  --agent-strategy tool-calling \
  --env airline \
  --model gpt-4o \
  --model-provider custom \
  --agent-base-url http://localhost:7777 \
  --user-model gpt-oss-120b \
  --user-model-provider custom \
  --user-base-url http://localhost:7000 \
  --user-strategy llm \
  --max-concurrency 5
```

## 🔍 **Health Checks:**

### Check Main Agent Server (7777):
```bash
curl http://localhost:7777/health
```

### Check User Strategy Server (7000):
```bash
curl http://localhost:7000/health
```

## 🎛️ **Server Behavior:**

### **Port 7777 (Main Agent):**
- ✅ Complex tool calling scenarios
- ✅ SRM reasoning pipeline
- ✅ Groq model integration
- ✅ Advanced conversation handling
- ✅ Full airline domain tools (14 tools)

### **Port 7000 (User Strategy):**
- ✅ Simple user simulation
- ✅ GPT-OSS-120B model
- ✅ Fast response times
- ✅ Natural user-like interactions
- ✅ No complex reasoning overhead

## 📝 **Testing Setup:**

### Test Main Agent (7777):
```bash
curl -X POST http://localhost:7777/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Book a flight from NYC to LAX"}],
    "model": "gpt-4o",
    "tools": [{"type": "function", "function": {"name": "search_flights"}}]
  }'
```

### Test User Strategy (7000):
```bash
curl -X POST http://localhost:7777/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hi, I need to book a flight"}],
    "model": "any-model-ignored"
  }'
```

## 🎯 **Expected Flow:**

```
Tau-Bench Orchestrator
       ↓
┌─────────────────┐         ┌─────────────────┐
│ User Simulator  │         │   Main Agent    │
│   (Port 7000)   │←──────→ │   (Port 7777)   │
│ GPT-OSS-120B    │         │ SRM + Groq      │
│ Simple & Fast   │         │ Complex Tools   │
└─────────────────┘         └─────────────────┘
```

## ⚡ **Benefits of This Setup:**

- **🎯 Specialized**: Each server optimized for its role
- **⚡ Performance**: User simulation doesn't need complex reasoning
- **🔧 Scalable**: Different worker counts per server
- **📊 Clear Separation**: Easy debugging and monitoring
- **💰 Cost Effective**: Right model for the right job

## 🚨 **Troubleshooting:**

1. **Port conflicts**: Make sure no other services use 7000/7777
2. **API Keys**: Check both GROQ_API_KEY and OPEN_ROUTER_API_KEY
3. **Health checks**: Verify both servers respond to /health
4. **Logs**: Check both server logs for any errors
