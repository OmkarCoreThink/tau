import json, re
from utils import call_llm
from pydantic import BaseModel, ConfigDict

class PrerequisiteResponse(BaseModel):
    model_config = ConfigDict(extra='forbid')
    reasoning: str
    parameter_value_available: bool
    parameter_value: str

def structured_tools(tools):
    result = {}

    for tool in tools:
        func = tool["function"]
        name = func["name"]
        description = func["description"]
        params = func["parameters"]

        required = params.get("required", [])
        all_props = params.get("properties", {})

        # required as (name, type)
        required_list = [(p, all_props[p]["type"]) for p in required]
        # not required as (name, type)
        not_required_list = [(p, v["type"]) for p, v in all_props.items() if p not in required]

        result[name] = {
            "description": description,
            "required": required_list,
            "not_required": not_required_list
        }

    return result

def match_tool(unit_task,tools,client,model_name,k=0):
    repair_note = (
        "- Previous attempts failed. Avoid repeating the same tool unless fixed prerequisites are now included.\n"
        "- Prefer tools that have not failed or tools whose prior failure can be resolved by earlier tasks.\n"
    ) 

    system_prompt = (
        "You are a precise function-calling agent. Your task is to select exactly one tool that directly fulfills the given task.\n\n"
        "CONTEXT AWARENESS:\n"
        "- You have access to conversation history, current execution state, and previous tool outcomes\n"
        "- Use this context to make informed decisions about which tool is most appropriate\n"
        "- Consider what has already been attempted and what the user is ultimately trying to achieve\n"
        "- Pay attention to any constraints or preferences revealed in the conversation\n\n"
        "OUTPUT FORMAT - You MUST OUTPUT the tool selection in a markdown code block format like this:\n"
        "```\n"
        "cd\n"
        "```\n\n"
        "rationale: explanation for why this tool is best given the context\n"
        "REQUIREMENTS:\n"
        "- Use the exact tool name from the available tools list\n"
        "- Provide a brief rationale that considers the conversation context\n"
        "- Only include the code block with tool selection, no other text\n"
        "- Focus only on selecting the correct tool name; parameters will be filled later\n"
        "- Consider the execution history and current state when making your choice"
        f"{repair_note}"
    )

    examples = [
        {
            "task": "Find the current weather in London.",
            "tools": [
                {"name": "get_weather", "description": "Fetches the current weather for a city.", "parameters": {"city": ""}},
                {"name": "get_stock_price", "description": "Gets the latest price for a stock symbol.", "parameters": {"symbol": ""}}
            ],
            "output": "```get_weather```\nrationale: This tool is specifically designed to fetch weather information for cities, which directly matches the task requirement.\n"
        },
        {
            "task": "Book a flight to New York from London",
            "tools": [
                {"name": "Google Hotels", "description": "Finds hotels in a city.", "parameters": {"city": ""}},
                {"name": "book_flight", "description": "Books a flight between two cities.", "parameters": {"origin": "", "destination": ""}}
            ],
            "output": "```book_flight```\nrationale: This tool handles flight booking between two cities, which is exactly what the task requires.\n"
        },
        {
            "task": "Check if 'final_report.pdf' exists in the document directory",
            "tools": [
                {"name": "find", "description": "Search for files and directories.", "parameters": {"name": "", "type": "", "path": ""}},
                {"name": "ls", "description": "List directory contents.", "parameters": {"path": ""}}
            ],
            "output": "```find```\nrationale: The find tool is specifically designed to search for files, which is perfect for checking if a specific file exists.\n"
        }
    ]

    block = []
    for ex in examples:
        block.append("### EXAMPLE ###")
        block.append(f"TASK:\n{ex['task']}")
        block.append(f"TOOLS AVAILABLE:\n{json.dumps(ex['tools'], indent=2)}")
        block.append(f"OUTPUT:\n{ex['output']}\n")
    
    analysis_messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": f"UNIT_TASK:\n{unit_task},"},
    {"role": "user", "content": f"TOOLS to choose from:\n{tools},"}
    ]

    system_prompt = system_prompt+ "\n".join(block)
    output = call_llm(client, analysis_messages, model_name)
    code_block_pattern = r'```(.*?)```'
    code_matches = re.findall(code_block_pattern, output, re.DOTALL)
    return code_matches[0].strip() if code_matches else output.strip()

def check_parameter(tools,tool_call,messages, scratchpad,unit_task,client, model_name,k):
    tools_dict = structured_tools(tools)
    for i in range(3):
        try:
            parameters = tools_dict[tool_call]
            break
        except:
            if i == 2:
                return f"No Tool call Needed. {tool_call}", unit_task
            tool_call = match_tool(unit_task,tools,client,model_name)
            continue

    system_prompt = (
        "You are a context-aware parameter-filling assistant. You will be given a TASK, a TOOL, "
        "its DESCRIPTION, PARAMETER SCHEMA, and optionally its RESPONSE SCHEMA, along with conversation context and execution history.\n\n"
        
        "CONTEXT AWARENESS:\n"
        "- You have access to the conversation history and current execution state\n"
        "- Use this context to understand what the user is trying to accomplish\n"
        "- Consider any constraints, preferences, or specific details mentioned in the conversation\n"
        "- Pay attention to previous tool executions and their outcomes\n"
        "- Extract parameter values that align with the user's overall goal\n"
        "- Use the tool description to understand the tool's purpose, behavior and parameters data types correctly\n"
        "- If provided, use the response schema to understand what the tool will return\n\n"
       """RESPONSE RULES (follow exactly):
            1. Output **only one** markdown code block containing a single JSON object that matches the PrerequisiteResponse model above. No extra text, no explanations, no surrounding prose.
            2. The JSON object must contain exactly three keys: "reasoning", "parameter_value_available", and "parameter_value". No additional keys allowed.
            3. Types:
            - "reasoning": a short, factual explanation (string) describing how you determined availability/value. Keep it concise and do **not** reveal chain-of-thought—only a brief summary of the basis for the decision.
            - "parameter_value_available": boolean true/false.
            - "parameter_value": the parameter's value as a string. If the parameter is not applicable or cannot be determined, set parameter_value_available to false and parameter_value to the string "NA".
            4. If the parameter is unavailable or not applicable, also set "reasoning" to a short statement such as "parameter not found in input" or "insufficient information".
            5. Do not include punctuation or formatting outside the JSON inside the markdown code block. The code block must contain only the JSON object and nothing else.
            6. Ensure the JSON is valid (use double quotes for strings, true/false for booleans, no trailing commas).

            EXAMPLE (for reference only — do not include this example in your output):
            ```json
            {
            "reasoning": "found exact key in prompt",
            "parameter_value_available": true,
            "parameter_value": "example_value"
            }"""
                )

    # Updated examples with markdown format
    few_shots = [
        {
            "TASK": "Convert 100 USD to EUR",
            "TOOL": "compute_exchange_rate",
            "DESCRIPTION": "Convert an amount from one currency to another using current exchange rates",
            "PARAMETER": "(base_currency,integer)",
            "parameter_value": "```100```",
            "parameter_value_available": True,
        },
        {
            "TASK": "Find first-class fares from JFK to LHR on 2025-07-04",
            "TOOL": "get_flight_cost",
            "PARAMETER": "(travel_from, string)",
            "parameter_value": "```JFK```",
            "parameter_value_available": True,
        },
        {
            "TASK": "Check if 'final_report.pdf' exists in the document directory",
            "TOOL": "find",
            "PARAMETER": "(Destination_path,string)",
            "parameter_value": "NA",
            "parameter_value_available": False,
        }
    ]
    
    examples_block = ""
    for ex in few_shots:
        description_line = f"TOOL DESCRIPTION: {ex.get('DESCRIPTION', '(no description available)')}\n" if ex.get('DESCRIPTION') else "TOOL DESCRIPTION: (no description available)\n"
        
        example_parts = [
            "### EXAMPLE ###",
            f"TASK: {ex['TASK']}",
            f"TOOL: {ex['TOOL']}",
            description_line.rstrip(),
            "PARAMETER:",
            f"```json\n{json.dumps(ex['PARAMETER'], indent=2)}\n```"
        ]
        
        # Add response schema if available in example
        if ex.get('RESPONSE'):
            example_parts.extend([
                "TOOL RESPONSE SCHEMA:",
                f"```json\n{json.dumps(ex['RESPONSE'], indent=2)}\n```"
            ])
        
        example_parts.extend([
            "EXPECTED OUTPUT:",
            f"{ex['parameter_value']}\n"
        ])
        
        examples_block += "\n".join(example_parts) + "\n"

    if len(parameters["required"]) == 0:
        values = f"Tool call {tool_call} does not have any parameters or arguments:\n"
    else:
        values = f"For {tool_call}: \n"
    for param in parameters["required"]:
        analysis_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"TRANSCRIPT:\n{messages},"},
        {"role": "user", "content": f"TRANSCRIPT_ANALYSIS:\n{scratchpad['turns_info']}"},
        {"role": "user", "content": f"CURRENT_STATE_ANALYSIS:\n{scratchpad['current_state_analysis']}"},
        {"role": "user", "content": f"UNIT_TASK:\n{unit_task},"},
        {"role": "user", "content": f"TOOL:\n{tool_call},"},
        {"role": "user", "content": f"TOOL DESCRIPTION:\n{parameters['description']},"},
        {"role": "user", "content": f"PARAMETER SCHEMA:\n{json.dumps(param, indent=2)},"},
        {"role": "user", "content": f"EXAMPLES:\n{examples_block}"},
        {"role": "system", "content": "Respond ONLY in the specified JSON format inside a markdown code block. If the user says he is not aware of the parameter value, assume something"}
        ]
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
        result = PrerequisiteResponse.model_validate_json(response.choices[0].message.content)
        if result.parameter_value_available:
            values += f"Parameter {param[0]} has value of {result.parameter_value}\n"
        else:
            print(f"Value of {param[0]} not found, asking user for the same")
            return f"Parameter {param[0]} is required for this task '{unit_task}' but its value is not available. Ask the user for the same.", "Ask user for " + param[0] + " value which is required for the unit task: " + unit_task
    for param in parameters["not_required"]:
        analysis_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"TRANSCRIPT:\n{messages},"},
        {"role": "user", "content": f"TRANSCRIPT_ANALYSIS:\n{scratchpad['turns_info']}"},
        {"role": "user", "content": f"CURRENT_STATE_ANALYSIS:\n{scratchpad['current_state_analysis']}"},
        {"role": "user", "content": f"UNIT_TASK:\n{unit_task},"},
        {"role": "user", "content": f"TOOL:\n{tool_call},"},
        {"role": "user", "content": f"TOOL DESCRIPTION:\n{parameters['description']},"},
        {"role": "user", "content": f"PARAMETER SCHEMA:\n{json.dumps(param, indent=2)},"},
        {"role": "user", "content": f"EXAMPLES:\n{examples_block}"},
        {"role": "system", "content": "Respond ONLY in the specified JSON format inside a markdown code block. If the user says he is not aware of the parameter value, assume something"}
        ]
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
        result = PrerequisiteResponse.model_validate_json(response.choices[0].message.content)
        
        if result.parameter_value_available == False:
            unit_task = "Find the value of " + param[0] + f" which can go as input to tool: {tool_call} with parameter description: {parameters['description']}"
            values += "Parameter " + param[0] + " is not required for this task, but ask user once for the value\n"
        else:
            values += "Parameter " + param[0] +" has value of "+result.parameter_value + "\n"
    
    return values.strip() if values else "No parameters required for this tool call.", unit_task
    
def find_parameter(tools,unit_task,messages, scratchpad, client, model_name,k):
    print(f"Finding parameter for unit task: {unit_task}")
    return final_tool_call(unit_task,messages,scratchpad, client, model_name,tools,k)

def final_tool_call(unit_task,messages,scratch_pad, client, model_name,tools,k=0):
    tool_call = match_tool(unit_task,tools,client,model_name,k)
    values,unit_task = check_parameter(tools,tool_call,messages, scratch_pad,unit_task,client, model_name,k)

    return values,unit_task