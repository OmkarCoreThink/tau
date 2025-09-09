from utils import call_llm
import re
from pydantic import BaseModel, ConfigDict
from typing import Literal, Optional
import json

class PrerequisiteResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')
    reasoning: str
    needs_user_input: bool
    task_description: str

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
    "7. Do not ask the user for review or confirmations. If you have all the information, go ahead and make the next action.\n\n"
    "8. If the user is unable to share required information, it means that you can yourself find it. Thus, create the unit task for the same.\n\n"
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
    "- Example: 1. If the immediate task is 'Authenticate to API service' and a default is required, output exactly:\n"
    "```\n"
    "Authenticate to API service (assume OAuth2 client ID available in ENV: OAUTH_CLIENT_ID)\n"
    "```\n"
    "2. If there are no remaining tasks, output exactly:\n"
    "```\n"
    "DONE\n"
    "```\n\n"
    "3. If the immediate task is 'Book flight HAT271 from ORD to PHL on May 10 in economy class using the user's default payment method', output exactly:\n"
    "```\n"
    "Book flight HAT271 from ORD to PHL on May 10 in economy class using the user's default payment method.\n"
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
    "✅ Is it a simple sentence and does not use AND\n"
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
    {"role": "system", "content": system_prompt}]
    analysis_messages.extend([{"role": "user","content":"Previous Conversation Transcript:\n"+json.dumps(messages, indent=2)}])
    new_msgs = [
    {"role": "user", "content": f"TRANSCRIPT_ANALYSIS:\n{scratchpad['turns_info']}"},
    {"role": "user", "content": f"CURRENT_STATE_ANALYSIS:\n{scratchpad['current_state_analysis']}"},
    {"role": "system", "content": f"UNIT_TASK_INSTRUCTIONS:\nCreate a single atomic unit task that advances the LATEST unsolved request using ONLY the information provided above. It should be pure natural language and does not include any tool calls or function calls. It should be a simple sentence and should not include AND"}
    ]
    analysis_messages.extend(new_msgs)
    output = call_llm(client, analysis_messages, model_name)
    code_block_pattern = r'```(.*?)```'
    code_matches = re.findall(code_block_pattern, output, re.DOTALL)
    return code_matches[0].strip() if code_matches else output.strip()

def check_prerequisites(scratchpad, messages, task, client, model_name):
    system_prompt = """
        You are an Assistant that determines if we need more information from the user to proceed with a task.
        You will be given: **CURRENT_STATE**, **UNIT_TASK**
        
        ## Your Job:
        1. Analyze the CURRENT_STATE to understand what information/context is already available
        2. Examine the UNIT_TASK to understand what needs to be accomplished
        3. Determine if the task can proceed with available information OR if user input is required
        
        ## Decision Logic:
        - **TRUE**: If the UNIT_TASK explicitly mentions asking the user for information, OR if the task cannot be completed without specific user-provided data that's missing from CURRENT_STATE
        - **FALSE**: If the UNIT_TASK can proceed using available tools/information without user input
        
          EXAMPLES:
        1. UNIT_TASK: Retrieve user profile for user_id "mia_li_3668" using get_user_details
            needs_user_input: FALSE
        2. UNIT_TASK: Ask the user for their user ID (or another identifier) to retrieve their profile and payment methods.
            needs_user_input: TRUE
        3. UNIT_TASK: Need to ask the user for the reservation ID (or any identifying details) of the May 10 ORD\u2192PHL flight, confirm the desired cabin class for the new booking (basic_economy, economy, or business), and ask which payment method from their profile should be used.\
            needs_user_input: TRUE
        4. UNIT_TASK: Please confirm proceeding with booking flight HAT271
            needs_user_input: TRUE
        5. UNIT_TASK: Book flight HAT271 from ORD to PHL on May 10 in economy class using the user's default payment method.
            needs_user_input: FALSE

        ## Response Format:
        Always provide:
        1. "reasoning": A paragraph that summarizes what's available in CURRENT_STATE, identifies what the UNIT_TASK requires, and explains why user input is/isn't needed
        2. "needs_user_input": boolean (true if user input is required, false otherwise)
        3. "task_description": If needs_user_input is true, describe what specific information to ask the user for (e.g., "Ask the user for their user ID", "Ask the user for the reservation ID and confirm cabin class")
        
        Many times, once the basic details are gathered, you can proceed with the UNIT_TASK using tools. So only ask for the absolute minimum information needed to proceed.
        (Important: needs_user_input=true must be used only when a user-facing information request is needed. Do not use true for internal prerequisites.)
    """
    
    analysis_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"UNIT_TASK:\n{task}"},
        {"role": "user", "content": f"CURRENT_STATE_ANALYSIS:\n{scratchpad['current_state_analysis']}"},
    ]
    
    # Using structured outputs with Groq
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=analysis_messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "prerequisite_check",
                "schema": PrerequisiteResponse.model_json_schema(),
            }
        }
    )
    
    # Parse the structured response
    result = PrerequisiteResponse.model_validate_json(response.choices[0].message.content)
    
    # Return in the original format for backward compatibility
    if result.needs_user_input:
        return "YES"
    else:
        return "NO"

def final_task(scratchpad, messages,client, model_name):
    task = unit_task(scratchpad, messages,client, model_name)
    verdict = check_prerequisites(scratchpad,messages,task, client, model_name)
    return task, verdict
