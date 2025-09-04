from utils import call_llm
import re

def unit_task(scratchpad, messages,client, model_name):
    # Updated system prompt (preserves your original rules but forces self-contained unit tasks)
    base_guidelines = (
    "Decompose instructions into precise, atomic steps with awareness of current state and tools. Use natural language for tasks.\n\n"
    "CORE PRINCIPLES:\n"
    "1. CRITICAL: Build from current state, leveraging successes and avoiding repeated failures.\n"
    "2. MANDATORY: Only create tasks for actions that are not yet completed - NEVER generate tasks for already solved requests.\n"
    "3. FOCUS: Identify and work ONLY on the latest unsolved request or incomplete parts.\n"
    "4. Use precise parameters and identifiers.\n"
    "5. Respect limitations and continue from previous attempts.\n"
    "6. Be domain agnostic and understand tool implications.\n\n"   
    "7. If user doesn't have the information, use you common sense and go ahead with the task.\n\n"
    "8. Do not ask the user for review. If you have all the information, go ahead and make the next action.\n\n"
    "MANDATORY: Follow planning directives from execution state.\n"
    "CRITICAL FILTERING RULES:\n"
    "- NEVER include tasks for requests that are already completed or solved\n"
    "- If a previous request has been fully satisfied, do NOT create any tasks for it\n"
    "- Only include tasks for the LATEST UNSOLVED request or incomplete portions\n"
    "- Output should contain ONLY tasks that still need execution - no skip tags, no completed tasks, no review tasks\n"
    "- If all user goals are satisfied, return an empty list []\n\n"
    "TASK REQUIREMENTS:\n"
    "- Each task must be specific with precise identifiers.\n"
    "- Avoid vague actions.\n"
    "- Ensure tasks are atomic, focusing on a single action or decision point.\n\n"
    "SEMANTIC PRECISION:\n"
    "- Respect user requests and preserve intent.\n"
    "- VERY IMPORTANT: Do not create or perform any actions not explicitly requested by the user.\n"
    "- If any unintended actions are identified, immediately undo them.\n\n"
    "CLEANUP TASKS:\n"
    "- Generate tasks to remove unwanted artifacts only if explicitly requested.\n\n"
    "FAILURE RESPONSE:\n"
    "- Include discovery, parameter adjustment, and connectivity setup as needed, but only if explicitly requested.\n\n"
    "CRITICAL: Output natural language task descriptions.\n"
    "- Tasks should be specific with clear action verbs.\n\n"
    "SELF-CONTAINMENT:\n"
    "- Unit tasks MUST be actionable using ONLY the information already provided in TRANSCRIPT, TRANSCRIPT_ANALYSIS, and CURRENT_STATE_ANALYSIS.\n"
    "IMMEDIATE-UNIT-TASK RULE (UPDATED):\n"
    "- If multiple tasks could be produced, the assistant MUST select the single most immediate, atomic task required to progress the latest unsolved request and output ONLY that task.\n"
    "- The selected single task must be returned inside a single fenced code block and must contain only the task text (no numbering, no extra commentary, no metadata).\n"
    "- Example: If the immediate task is 'Authenticate to API service' and a default is required, output exactly:\n"
    "```\n"
    "Authenticate to API service (assume OAuth2 client ID available in ENV: OAUTH_CLIENT_ID)\n"
    "```\n"
    "- If there are no remaining tasks, output exactly:\n"
    "```\n"
    "DONE\n"
    "```\n\n"
    "OUTPUT FORMAT (UPDATED):\n"
    "- MANDATORY: For normal situations where a list of tasks is required and the user explicitly expects a multi-step plan, respond with ONLY a markdown code block containing a numbered list of tasks as previously specified.\n"
    "- CRITICAL UPDATE: When more than one task is possible but only the NEXT IMMEDIATE atomic action should be returned (per the IMMEDIATE-UNIT-TASK RULE), respond with a single fenced code block containing exactly the one task text (no list, no numbering, no explanation).\n"
    "- NO explanatory text outside the code block in either mode.\n"
    "- For empty task lists, use:\n"
    "```\n"
    "DONE\n"
    "```\n"
    "- Tasks must be in correct execution order when a list is requested.\n"
    "- CRITICAL: Only include tasks that are needed and not completed.\n"
    "- NEVER include tasks for requests that are already solved or completed.\n"
    "- Do NOT include any SKIP tasks or mention skipping - output only executable tasks.\n"
    "- Focus ONLY on the latest unsolved request.\n\n"
    "TASK VALIDATION:\n"
    "✅ Is it a complete sentence with specific identifiers?\n"
    "✅ Does it use action verbs?\n"
)

    repair_addendum = (
    "\n\nREPAIR MODE:\n"
    "- Use history to avoid repeating failures.\n"
    "- Include diagnostics to clarify state.\n"
    "- Focus on incremental progress for UNSOLVED requests only.\n"
    "- NEVER include tasks for already resolved requests, even in repair mode.\n"
    "- Output should contain ONLY tasks that need execution - no mentions of skipping.\n"
    "- If all goals are resolved, return {\"tasks\": []}.\n"
)

    system_prompt = base_guidelines + repair_addendum

    analysis_messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": f"TRANSCRIPT:\n{messages},"},
    {"role": "user", "content": f"TRANSCRIPT_ANALYSIS:\n{scratchpad['turns_info']}"},
    {"role": "user", "content": f"CURRENT_STATE_ANALYSIS:\n{scratchpad['current_state_analysis']}"},
    {"role": "user", "content": f"TOOL_AWARENESS:\n{scratchpad['tool_awareness']}"},
    {"role": "user", "content": f"UNIT_TASK_INSTRUCTIONS:\nCreate a single atomic unit task that advances the LATEST unsolved request using ONLY the information provided above."}
    ]

    output = call_llm(client, analysis_messages, model_name)
    code_block_pattern = r'```(.*?)```'
    code_matches = re.findall(code_block_pattern, output, re.DOTALL)
    return code_matches[0].strip() if code_matches else output.strip()

def check_prerequisites(scratchpad,messages,task, client, model_name):
    system_prompt = """
    You are an Assistant which sees if we need more information from the user.
    You will be given following fields: CURRENT_STATE, UNIT_TASK.
    Check if the Unit Task is to ask more information from the user.
    Sometimes, the task is DONE but more steps are required to complete the task. In that case, output the missing prerequisite step that needs to be performed. If you are even a little unsure of the task that needs to be done, create a task that needs to be done.
    Output rules (strict):
    1. Always begin with a detailed Reasoning paragraph explaining why you concluded the task is ready or what is missing.
    2. After the Reasoning paragraph, produce exactly one of the following canonical outputs:
    A) If the UNIT_TASK is not to ask more information from the user, end the OUTPUT:
    ```NO```
    B) If the single most urgent missing prerequisite is *to ask the user for more information* (i.e., you cannot proceed until you get specific data from the user), output `YES` on its own line followed immediately by a triple-backticked TASK block containing that missing user-request step. Example:
    ```YES```
    TASK - - <Ask the user for their first name, last name, etc>

    EXAMPLES:
    1. UNIT_TASK: Retrieve user profile for user_id "mia_li_3668" using get_user_details
        OUTPUT: Reasoning ..... ```NO```
    2. UNIT_TASK: Ask the user for their user ID (or another identifier) to retrieve their profile and payment methods.
        OUTPUT: Reasoning .... ```YES```

    Many times, once the basic details are gathered, you can proceed with the UNIT_TASK using tools. So only ask for the absolute minimum information needed to proceed.
    (Important: YES must be used only when a user-facing information request. Do not use YES for internal prerequisites.)
    DO NOT FORGET THE TRIPLE BACKSTICKS
    """
    analysis_messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": f"UNIT_TASK:\n{task}"},
    ]

    output = call_llm(client, analysis_messages, model_name)
    code_block_pattern = r'```(.*?)```'
    code_matches = re.findall(code_block_pattern, output, re.DOTALL)
    return code_matches[0].strip() if code_matches else output.strip()

def final_task(scratchpad, messages,client, model_name):
    task = unit_task(scratchpad, messages,client, model_name)
    verdict = check_prerequisites(scratchpad,messages,task, client, model_name)
    return task, verdict
