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

def call_llm(messages: list[dict[str, Any]], model: str, tools: List[LLMTool]) -> str:
    cognite_client = create_client()
    new_messages = messages.copy()
    
    tools_schemas, tool_execute_map = get_tools(tools)
    # print("tools: ", json.dumps(tools, indent=2))
    
    while True:
        response = cognite_client.post(f'/api/v1/projects/{cognite_client.config.project}/ai/chat/completions',
            json = {
                "model": model,
                "messages": new_messages,
                "tools": tools_schemas
            },
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
            print(" Performing tool call with tool name: ", tool_name, " and arguments: ", tool_arguments)
            tool_result = tool_execute(tool_arguments)
            new_messages.append({
                "role": "tool",
                "content": tool_result,
                "toolCallId": tool_call['id']
            })
        else:
            return data['choices'][0]['message']['content']