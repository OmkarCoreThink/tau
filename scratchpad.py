from utils import call_llm
import json

def break_into_turns(client,model_name,prev_messages,tool_understanding):
    system_prompt = """
    You are an expert function-calling conversation analyst. Your job: analyze the ENTIRE conversation transcript and produce a concise, evidence-based breakdown of every user request and its execution status.

    SCOPE
    - Focus ONLY on the latest unsolved user request for any follow-up actions, but include all requests for context.
    - Do NOT invent facts or assume tool executions that have no visible evidence.

    CORE RULES
    1. COMPREHENSIVE: Identify ALL user requests present in the transcript (up to 5 requests; if more exist, include the 5 most recent).
    2. EVIDENCE-BASED: Count tool calls only when there is visible execution evidence in the transcript (mark otherwise NOT_ATTEMPTED).
    3. CHRONOLOGICAL: Order requests as they appeared.
    4. STATUS: Mark each request as COMPLETE, INCOMPLETE, or BLOCKED.
    5. LATEST UNSOLVED: Clearly identify which request is the latest unsolved one.
    6. NO EXTRANEOUS TEXT: Output only the required structured analysis; do not add commentary outside the template.

    OUTPUT (strict; produce a single markdown code block)
    For each request produce this block (repeat for each request up to the allowed max):

    #### Request <N>: <one-line summary>
    **User Message:** "<exact user message text>"

    **Tool Calls Made:**
    - `tool_name(arg1="value", ...)` — [Status: EXECUTED / FAILED / NOT_ATTEMPTED]  
    (only list calls that appear in the transcript; if a call executed, include any visible result text)

    **Results:**  
    - Concise summary of what happened (include visible evidence: returned values, logs, error messages).

    **Completion Status:** [COMPLETE / INCOMPLETE / BLOCKED]  
    **Final Notes:** Brief reason for the status and any missing evidence or blocker.

    At the very end, include a single-line summary:
    **Latest Unsolved Request:** Request <N> — <one-line reason why it is unsolved>.

    Strict formatting rules:
    - Produce the full analysis in one markdown code block (no extra paragraphs outside it).
    - Do not exceed 5 requests. If the conversation contains fewer than 5, include all.
    - Keep language factual and succinct.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"CONVERSATION TRANSCRIPT:\n{json.dumps(prev_messages, indent=2)}"},
    ]

    return call_llm(client, messages, model_name)

def current_state_analysis(turns_info, client,model_name,messages):
    system_prompt = (
    "You are an expert function-calling conversation analyst. Your job: analyze the LATEST UNSOLVED user request and determine how to complete it.\n\n"
    
    "SCOPE: Focus ONLY on the current unsolved request. Use the conversation breakdown as context but analyze only the latest incomplete task.\n\n"
    
    "CORE PRINCIPLES:\n"
    "1. TASK-FOCUSED: Primary goal is to solve the user's request effectively and completely\n"
    "2. POLICY-AWARE: Identify relevant policies upfront and clearly flag when actions conflict with them\n"
    "3. EVIDENCE-BASED: Only count actions with visible execution proof in transcript\n"
    "4. SOLUTION-ORIENTED: Explore ALL possibilities, including alternative approaches when policies create constraints\n"
    "5. AUTONOMOUS: Make reasonable assumptions, ask users only when critical\n"
    "6. PRECISE: Distinguish EXECUTED vs PLANNED vs NEEDED actions clearly\n"
    "7. THOROUGH: Identify ALL variables needed before attempting solutions\n\n"
    
    "KEY WORKFLOWS:\n"
    "- Entity Lookup: ID search → get_user_details(ID) → alternative searches if needed\n"
    "- Operations: verify prerequisites → execute → handle errors with alternatives\n"
    "- Searches: exact → broaden criteria → alternative methods → related queries\n\n"
    
    "ASK USERS ONLY FOR:\n"
    "- Critical missing info that cannot be inferred from context\n"
    "- Destructive actions needing confirmation\n"
    "- Security credentials\n"
    "- Significantly different outcome interpretations\n"
    "BALANCE: Don't assume critical details, but infer reasonable defaults from context\n\n"
    
    "EXECUTION VERIFICATION:\n"
    "- EXECUTED: Tool call with visible results/errors\n"
    "- PLANNED: Discussed but no execution evidence\n"
    "- NEEDED: Required next actions\n"
    "- State 'NO EXECUTIONS FOUND' if no tool calls occurred\n\n"
)

    current_state_mapping = (
    "CURRENT STATE MAPPING\n"
    "Analyze what has actually been executed vs planned FOR THE LATEST UNSOLVED REQUEST.\n\n"
    
    "OUTPUTS REQUIRED:\n"
    "- EXECUTED ACTIONS: Tool calls with visible results/errors only\n"
    "- AVAILABLE DATA: Information actually obtained from transcript\n"
    "- CONSTRAINTS: Known limitations (permissions, rate limits, etc.)\n"
    "- UNKNOWNS: Missing info needed to proceed\n"
    "- ENVIRONMENT: System type and current state\n"
    "- ASSUMPTIONS: What you're inferring and why\n"
)

    request_focus = (
    "REQUEST FOCUS ANALYSIS\n"
    "Define the LATEST UNSOLVED user request and determine how to achieve it.\n\n"
    
    "OUTPUTS REQUIRED:\n"
    "- POLICY REQUIREMENTS: First, identify ALL policies that apply to this task (data access, user privacy, modification rules, etc.)\n"
    "- USER REQUEST: One clear sentence of what user wants\n"
    "- DELIVERABLES: Exact outputs and success criteria\n"
    "- ALL VARIABLES NEEDED: Complete list of required data/parameters (mark AVAILABLE/MISSING)\n"
    "- POLICY ANALYSIS: Identify any policy conflicts and clearly state if the request violates policies\n"
    "- FEASIBILITY: Achievable? If policy conflicts exist, clearly state them and provide alternative approaches\n"
    "- ASSUMPTIONS: What you're inferring and confidence level (HIGH/MEDIUM/LOW)\n"
    "- ACTION PLAN: 3-5 concrete next steps with alternatives (note policy considerations where relevant)\n"
    "- CRITICAL: If user ID found → immediately call get_user_details(user_id)\n"
)

    error_analysis = (
    "ERROR ANALYSIS\n"
    "Examine what actually executed and what went wrong FOR THE LATEST UNSOLVED REQUEST.\n\n"
    
    "OUTPUTS REQUIRED:\n"
    "- EXECUTION STATUS: List actual tool calls with EXECUTED/FAILED/NOT_ATTEMPTED\n"
    "- RESULTS vs EXPECTED: What happened vs what was intended\n"
    "- ERROR DETAILS: Specific error messages, codes, root causes\n"
    "- ALTERNATIVE APPROACHES: 2-3 different methods to try\n"
    "- IMMEDIATE FIXES: Concrete steps to resolve issues\n"
    "- If NO executions found, state this and recommend what to do\n"
)
    understanding_task = (
    "WHAT REMAINS\n"
    "Identify what's completed vs what still needs to be done.\n\n"
    
    "OUTPUTS REQUIRED:\n"
    "- PROGRESS: What's actually been completed (with evidence)\n"
    "- REMAINING TASKS: What still needs to be done\n"
    "- BLOCKERS: What's preventing progress and how to resolve\n"
    "- MULTIPLE PATHS: Primary approach + 2 alternatives for each task\n"
    "- PRIORITIES: P0/P1/P2 for each remaining item\n"
    "- NEXT ACTIONS: 5-8 concrete immediate steps\n"
)

    next_actions = (
    "NEXT ACTIONS\n"
    "List immediate, actionable steps to complete the LATEST UNSOLVED user request.\n\n"
    
    "FORMAT: One action per line, imperative form, no bullets\n"
    "SCOPE: Only actions for the current request\n"
    "ORDER: Sequence to execute for completion\n"
    
    "Examples:\n"
    "Call get_user_details with user ID 12345\n"
    "Search flights from LAX to JFK on 2024-01-15\n"
    "Update customer preferences in database\n"
    "\nREQUIREMENT: Ensure all needed variables are identified before proceeding\n"
)

    final_instructions = (
        "OUTPUT REQUIREMENTS:\n"
        "- Use clear markdown format with sections\n"
        "- Be specific and actionable\n"
        "- PRIMARY FOCUS: Solve the user's request - be solution-oriented\n"
        "- POLICY HANDLING: Check policy compliance and clearly state when conflicts prevent task completion\n"
        "- Always provide multiple solution paths, including alternatives when policy constraints exist\n"
        "- State assumptions and confidence levels (HIGH/MEDIUM/LOW)\n"
        "- Focus on what can be done next, not just problems\n"
        "- If user ID obtained → immediately call get_user_details(user_id)\n"
        "- Identify ALL required variables before recommending actions\n"
        "- When policy violations make a request impossible, clearly explain why and suggest alternative approaches\n"
    )
    
    # Define section names for better structure
    section_names = [
        "Current State Mapping",
        "Request Focus Analysis", 
        "Error Analysis",
        "Understanding What Remains",
        "Next Actions",
    ]
    
    prompts = [current_state_mapping, request_focus, error_analysis, understanding_task, next_actions]
    
    # Simple format - clean section-by-section analysis
    current_state_analysis = ""
    
    for i, (section_name, prompt) in enumerate(zip(section_names, prompts)):
        system_prompt+= f"\n\n{section_name}\n\n{prompt}\n\n"
    analysis_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"TRANSCRIPT_ANALYSIS:\n{turns_info}"},
        {"role": "user", "content": f"CONVERSATION_TRANSCRIPT:\n{json.dumps(messages, indent=2)}"},
        {"role": "system", "content": f"ADDITIONAL_INSTRUCTIONS:\n{final_instructions}"},
        {"role": "system", "content": "Try to be thorough and do not leave any room in planning. Be very detailed in your analysis. Think of all the possibilities while keeping rules and Policies in mind. FOCUS: Prioritize solving the user's task effectively. When policies create constraints, clearly identify them and explain if they prevent task completion, then suggest practical alternatives that still achieve the user's goals."},
    ]

    section_result = call_llm(client, analysis_messages, model_name)
    
    # Build current context for next section (simple format)
    current_state_analysis += f"\n## {section_name}\n{section_result}\n"
    
    return current_state_analysis

def scratch_pad_generation(client,messages,model_name,tools):
    tool_understanding = json.dumps(tools, indent=2)  # Use JSON dump for better formatting
    turns_info = break_into_turns(client, model_name, messages,tool_understanding)
    current_state_result = current_state_analysis(turns_info, client, model_name, messages)

    scratch = {}
    scratch["tool_awareness"] = tool_understanding
    scratch["turns_info"] = turns_info
    scratch["current_state_analysis"] = current_state_result

    return scratch