from typing import Any, List
import json
from auth import create_client
from python_agent.tools.LLMTool import LLMTool
from typing import Callable

def get_tools(tools: list[LLMTool]) -> tuple[list[dict[str, Any]], dict[str, Callable[[Any], Any]]]:
    tools_schemas: list[dict[str, Any]] = []
    tool_execute_map: dict[str, Callable[[Any], Any]] = {}
    for tool in tools:
        llm_tools = tool.agent_tool.get_llm_tools()
        for llm_tool in llm_tools:
            json_schema = llm_tool.get_json_schema()
            tools_schemas.append({
                "type": "function",
                "function": {
                    "name": tool.agent_tool.externalId,
                    "description": tool.agent_tool.instructions,
                    "parameters": json_schema
                }
            })
            
            tool_execute_map[tool.agent_tool.externalId] = lambda x: llm_tool.invoke(**x)
    return tools_schemas, tool_execute_map

def call_llm(messages: list[dict[str, Any]], model: str, tools: List[LLMTool] | None = None) -> dict[str, Any]:
    cognite_client = create_client()
    new_messages = messages.copy()
    tool_calls: list[dict[str, Any]] = []

    if tools:
        tools_schemas, tool_execute_map = get_tools(tools)
    else:
        tools_schemas = []
        tool_execute_map = {}
    
    while True:
        json_body = {
            "model": model,
            "messages": new_messages,
        }
        if tools:
            json_body["tools"] = tools_schemas
        
        response = cognite_client.post(f'/api/v1/projects/{cognite_client.config.project}/ai/chat/completions',
            json = json_body,
            headers={"cdf-version": "alpha"}
        )
        
        response.raise_for_status()
        data = response.json()
        
        # print("data: ", json.dumps(data, indent=2))
        if "toolCalls" in data['choices'][0]['message']:
            # Add the tool call to the messages
            new_messages.append(data['choices'][0]['message'])
            tool_call = data['choices'][0]['message']['toolCalls'][0]
            tool_name = tool_call['function']['name']
            tool_arguments = json.loads(tool_call['function']['arguments'])
            tool_execute = tool_execute_map[tool_name]
            try:
                tool_result = tool_execute(tool_arguments)
            except Exception as e:
                print("Error executing tool: ", e)
                tool_result = "Error executing tool: " + str(e)
            new_messages.append({
                "role": "tool",
                "content": tool_result,
                "toolCallId": tool_call['id']
            })
            tool_calls.append({
                "tool_name": tool_name,
                "tool_arguments": tool_arguments,
                "tool_result": tool_result
            })
        else:
            return {
                "content": data['choices'][0]['message']['content'],
                "tool_calls": tool_calls
            }
