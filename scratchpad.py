from utils import call_llm
import json

def tool_awareness(client,model_name,tools):
    system_prompt = f"""
    You are a Tools Analyst. You will be given a JSON array `tools` describing many tools. DO NOT call any tool. For each tool output a Markdown block (one header per tool) that fits max 3–4 short lines. Each tool block must include:
    1) One-line Purpose (what the tool does). 
    2) Inputs — list required params with types; mention optional params if any (comma-separated).
    3) Output — main response fields & types. 
    4) When to use / caution (one short phrase).
    Preserve exact parameter names and types from the schema. If a field is missing, infer concisely. Return ONLY the Markdown summary for all tools.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Tools info:\n{tools}"}
    ]
    return call_llm(client, messages, model_name)

def break_into_turns(client,model_name,messages,tool_understanding):
    system_prompt = """
    You are tasked with analyzing a conversation transcript to verify if the intended tasks have been completed correctly. Focus strictly on the tasks explicitly mentioned, ignoring any extraneous information.

    Do not make any assumptions and put the analysis to the best of your knowledge.
    Analysis should only focus on what is asked although there might be some extra information. We should not focus things which are not explicitly asked.

    Your task is to identify and verify for each user request:
    1. User requests (messages with role='user')
    2. Tool calls made in response to each request
    3. Tool call results/outcomes
    4. The current state and what the user expects
    5. You do not have to plan to solve the user task, only analysis
    6. Based on the previous tool call made, make sure to clearly understand the units of the tool call output based on the tool awareness shared with you\n"
    7. Once the task is done, it doesn't need to be repeated again, so make sure to not repeat the task if it is already done\n"
    8. Once the task is done, it doesn't need to confirm the message back to the user\n"

    Ensure that your analysis is thorough and only includes what is explicitly asked for. Do not miss anything.
    Output your analysis in the following structured markdown format:

    ## Conversation Analysis

    ### User Requests and Tool Execution History
    #### Request 1: [Brief summary of the user's request]
    **User Message:** "[Full user message content]"

    **Tool Calls Made:**
    - `tool_name(param1="value1", param2="value2")` - [Status: Success/Failed]

    **Results:**
    - [Summary of what was accomplished or any errors]

    **Completion Status:** [Incomplete / Complete]  
    **Final Notes:** [Any concise observations; e.g., why marked incomplete, missing evidence, or why it's complete]

    #### Request 2: [Brief summary]
    **User Message:** "[Full user message content]"

    **Tool Calls Made:**
    - `tool_name(params)` - [Status]

    **Results:**
    - [Summary of results including errors and map of things happened till time of the request]

    **Completion Status:** [Incomplete / Complete]  
    **Final Notes:** [...]

    Output the entire analysis in a single markdown code block, ensuring it is well-structured and easy to read. Do not include any additional explanations or comments outside the code block.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Conversation transcript:\n{messages}"},
        {"role": "user", "content": f"Tool Awareness:\n{tool_understanding}"}
    ]
    return call_llm(client, messages, model_name)

def check_if_anything_is_missing(turns_info,client,model_name,messages):

    system_prompt = """
    TASK:
    You will be given a JSON object with `communication_history` (ordered earlier turns) and `latest_turn` (most recent user request). Your job: determine whether to carry out latest request, do we have enough **sufficient** information in all the provided messages and tool calls outputs.

    DECISION:
    - If sufficient -> output JSON: {"sufficient": true, "message": "no more information needed"}
    - If not sufficient -> output JSON: {"sufficient": false, "message": "find out what tasks needs to be performed so that we get this information", "tasks": [...actionable tasks...]}

    RULES:
    1. Identify the request type (e.g., booking, deployment, file operation, report, code change).
    2. Determine the **minimal** required fields/steps to execute that request (credentials, files, dates, formats, access, acceptance criteria).
    3. Check `communication_history` for those items. Treat ambiguous or uncertain items as missing.
    4. If missing, list concise, actionable tasks that, when completed, will produce the missing information.
    5. Output **only** the JSON described above. No extra text, no explanation.

    INPUT FORMAT (you will receive exactly):
    {
    "communication_history": [
        {"speaker":"user"|"assistant","content":"..."},
        ...
    ],
    "latest_turn": {"speaker":"user"|"assistant","content":"..."}
    }

    OUTPUT FORMAT (must be valid JSON only):
    - Sufficient case:
    {
    "sufficient": true,
    "message": "no more information needed"
    }
    - Insufficient case:
    {
    "sufficient": false,
    "message": "find out what tasks needs to be performed so that we get this information",
    "tasks": [
        "Task 1 (actionable and specific)",
        "Task 2",
        ...
    ]
    }

    BEHAVIOR NOTES:
    - Prefer minimal task lists (only what's necessary).
    - Use plain, specific actions (e.g., "Upload the repo URL and grant read/write access", "Provide passport number, nationality, expiry").
    - Do not ask questions in the output; state tasks.
    - Always return valid JSON and nothing else.

        """
   
    analysis_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"BFCL_TRANSCRIPT:\n{messages},"},
        {"role": "user", "content": f"TRANSCRIPT_ANALYSIS:\n{turns_info}"},
        #{"role": "user", "content": f"TOOL_AWARENESS:\n{tool_awarenes}"}
    ]

    return call_llm(client, analysis_messages, model_name)


def current_state_analysis(turns_info, client,model_name,messages,miss_info):
    system_prompt = (
        "You are an expert at analyzing function calling conversations to determine the current state and propose the next action for the latest user request.\n\n"
        "Your task is to thoroughly examine the entire transcript to understand what has been done and what remains, ensuring no repetition of steps.\n\n"
        
       "STEP 1 - CURRENT STATE MAPPING: Create a high-level map of the environment\n" 
       "- Review previous requests and their outcomes to understand the current state\n" 
       "- Identify the current state of the system you are interacting with, whether it's a file system, database, or any other environment\n" 
       "- Mention all the information about your current environment that is relevant to the task at hand\n" 
       "- Do not make any assumptions regarding the system that you are dealing with and find out if you are not sure of your current state\n"
       "- Note any data that has been loaded or connections established\n" 
       "- Based on the previous tool call made, make sure to clearly understand the units of the tool call output based on the tool awareness shared with you\n"
       "- Understand the overall state of the working environment\n\n"

        "STEP 2 - REQUEST FOCUS: Analyze the latest user request\n"
        "- Identify the most recent task that needs completion\n"
        "- Determine if the request has been addressed or remains unmet\n"
        "- Note any constraints or errors that may impact fulfilling the request\n\n"
        
        "STEP 3 - ERROR ANALYSIS: Examine any tool call outcomes for the current request\n"
        "- Review the results of tool calls made in response to the current request\n"
        "- Identify any errors or unexpected outcomes from these tool calls\n"
        "- Consider how these errors impact the ability to fulfill the request\n\n"
        
        "STEP 4 - UNDERSTSNDING THE  TASK\n"
        "- Analyze the task at hand and understand what is being asked\n"
        "- Identify the specific actions that need to be taken to complete the task\n"
        "- Break down the task into smaller, manageable steps\n"
        "- Do do over-simplify the task, rather do a detailed analysis\n"
        "- Do not make assumptions of the starting position and if unsure, add it to your tasks to find the starting position\n"
        "- Once the task is done, it doesn't need to confirm the message back to the user\n"
        "- Try to find out what needs to be done first before starting the task i.e. do we have any pre-requisites to complete the task\n\n"

       "STEP 5 — NEXT ACTIONS (apply to the latest request only)\n\n"
       "Produce a plain list of atomic what-to-do actions for the latest request, using only the current state. Follow these rules exactly:\n\n"
       "1. Output format: a plain list with one action per line.\n"
       "2. Scope: include only actions that directly complete the latest request. Do not include requests for more information, diagnostics, investigation, or any work outside the user's explicit request.\n"
       "3. Atomicity: each action must be a single, simple, imperative sentence. Do not use compound or complex sentences. Avoid coordinating conjunctions (and, or), semicolons, commas that join clauses, or subordinate clauses.\n"
       "4. No how-to: do not include implementation steps, commands, methods, tool names, or procedural details. State what to do, not how to do it.\n"
       "5. No repetition: do not restate actions that are already completed or duplicated elsewhere.\n"
       "6. Sequence: list atomic actions in the precise order they should be executed to finish the request as requested by the user.\n"
       "7. Specificity: keep each action specific and unambiguous. Include identifiers when relevant (e.g., feature name, branch, build number, document name). Do not over-simplify to the point of losing necessary detail.\n"
       "8. Nothing remaining: if there are no remaining actions, return an empty list (i.e., no lines).\n"
       "9. Tone and extras: do not add explanations, notes, justifications, or commentary. Provide only the ordered action lines.\n\n"
       "10. Do not ask the user for review. If you have all the information, go ahead and make the next action.\n\n"
      
        "IMPORTANT: Use natural language descriptions throughout. Describe actions and outcomes in plain English. "
        "Be very very detailed about what is currently happening and what needs to be done next only for the latest unsolved request\n\n"
        
        "Respond with your analysis directly in natural language. Focus on being factual and precise, "
        "and provide actionable insights about what should happen next.\n\n"
        
        "OUTPUT FORMAT: Ensure your response is strictly in markdown format."
    )
    
    analysis_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"BFCL_TRANSCRIPT:\n{messages},"},
        {"role": "user", "content": f"TRANSCRIPT_ANALYSIS:\n{turns_info}"},
        {"role": "user", "content": f"POSSIBILTY_OF_MISSING_INFORMATION:\n{miss_info}"}
    ]

    return call_llm(client, analysis_messages, model_name)

def scratch_pad_generation(client,messages,model_name,tools):
    #tool_understanding = tool_awareness(client, model_name, tools)
    tool_understanding = json.dumps(tools, indent=2)  # Use JSON dump for better formatting
    turns_info = break_into_turns(client, model_name, messages,tool_understanding)
    miss_info = check_if_anything_is_missing(turns_info, client, model_name, messages)
    current_state = current_state_analysis(turns_info, client, model_name, messages,miss_info)

    scratch = {}
    scratch["tool_awareness"] = tool_understanding
    scratch["turns_info"] = turns_info
    scratch["current_state_analysis"] = current_state

    return scratch