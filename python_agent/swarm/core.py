# Standard library imports
import copy
import json
from collections import defaultdict
from typing import List, Any


# Local imports
from .util import function_to_json, debug_print, merge_chunk
from .swarm_types import (
    Agent,
    AgentFunction,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    Function,
    Response,
    Result,
)

__CTX_VARS_NAME__ = "context_variables"


class Swarm:
    def __init__(self, client=None):
        self.client = client

    def get_chat_completion(
        self,
        agent: Agent,
        history: List,
        context_variables: dict,
        model_override: str,
        stream: bool,
        debug: bool,
    ) -> ChatCompletionMessage:
        context_variables = defaultdict(str, context_variables)
        instructions = (
            agent.instructions(context_variables)
            if callable(agent.instructions)
            else agent.instructions
        )
        messages = [{"role": "system", "content": f"You are agent with name {agent.name} with the following instructions:\n\n{instructions}"}] + history
        messages = copy.deepcopy(messages)
        # We need to clean up the messages to be in the correct format.
        # We need to remove the 'sender' if it exists
        for message in messages:
            message.pop("sender", None)
        debug_print(debug, "Getting chat completion for...:", messages)

        tools = [function_to_json(f) for f in agent.functions]
        # hide context_variables from model
        for tool in tools:
            params = tool["function"]["parameters"]
            params["properties"].pop(__CTX_VARS_NAME__, None)
            if __CTX_VARS_NAME__ in params["required"]:
                params["required"].remove(__CTX_VARS_NAME__)
        
        # Replace tool calls no longer in list with message
        messages_to_llm = []
        agent_transfer_to = {}
        tool_call_senders = {}
        for message in messages:
            if message.get("toolCalls"):
                for tool_call in message["toolCalls"]:
                    # Store sender for later
                    sender = message["toolCalls"][0].get("sender")
                    tool_call_senders[tool_call["id"]] = sender
                    # Remove sender from tool call
                    message["toolCalls"][0].pop("sender")
                    messages_to_llm.append(message)
                    
            elif message["role"] == "tool":
                if "assistant" in message['content']:
                    agent_transfer_to[message["toolCallId"]] = json.loads(message["content"])["assistant"]
                else:
                    # Normal tool call, keep it for now
                    messages_to_llm.append(message)
            else:
                # Keep the original message if no 'toolCalls' exist
                messages_to_llm.append(message)
        # Now we need to replace tool calls with assistant message
        messages = copy.deepcopy(messages_to_llm)
        messages_to_llm = []
        
        last_message_was_agent_transfer = False
        for message in messages:
            if message.get("toolCalls"):
                id = message["toolCalls"][0]["id"]
                if id in agent_transfer_to:
                    last_message_was_agent_transfer = True
                    sender = tool_call_senders[id]
                    receiver = agent_transfer_to[id]
                    new_message = {
                        "role": "assistant",
                        #"content": f"{sender} has now transferred control to {receiver}. {receiver}, it is currently you who are speaking."
                        "content": f"You are now talking to {receiver}."
                    }
                    messages_to_llm.append(new_message)
                else:
                    messages_to_llm.append(message)
            else:
                last_message_was_agent_transfer = False
                messages_to_llm.append(message)
        # print('messages_to_llm: ', messages_to_llm)
        create_params = {
            "model": model_override or agent.model,
            "messages": messages_to_llm,
            "tools": tools or None,
            "toolChoice": agent.tool_choice,
            "stream": stream,
        }

        response = self.client.post(f'/api/v1/projects/{self.client.config.project}/ai/chat/completions', json={
          **create_params
        })
        data = response.json()
        if last_message_was_agent_transfer:
            data["choices"][0]["message"]["content"] = messages_to_llm[-1]["content"]
            data["choices"][0]["message"].pop("toolCalls", None)
        return data, last_message_was_agent_transfer

    def handle_function_result(self, result, debug) -> Result:
        match result:
            case Result() as result:
                return result

            case Agent() as agent:
                return Result(
                    value=json.dumps({"assistant": agent.name}),
                    agent=agent,
                )
            case _:
                try:
                    return Result(value=str(result))
                except Exception as e:
                    error_message = f"Failed to cast response to string: {result}. Make sure agent functions return a string or Result object. Error: {str(e)}"
                    debug_print(debug, error_message)
                    raise TypeError(error_message)

    def handle_tool_calls(
        self,
        tool_calls: List[ChatCompletionMessageToolCall],
        functions: List[AgentFunction],
        context_variables: dict,
        debug: bool,
    ) -> Response:
        function_map = {f.__name__: f for f in functions}
        partial_response = Response(
            messages=[], agent=None, context_variables={})

        for tool_call in tool_calls:
            name = tool_call.function.name
            # handle missing tool case, skip to next tool
            if name not in function_map:
                debug_print(debug, f"Tool {name} not found in function map.")
                partial_response.messages.append(
                    {
                        "role": "tool",
                        "toolCallId": tool_call.id,
                        # "toolName": name,
                        "content": f"Error: Tool {name} not found.",
                    }
                )
                continue
            args = json.loads(tool_call.function.arguments)
            debug_print(
                debug, f"Processing tool call: {name} with arguments {args}")

            func = function_map[name]
            # pass context_variables to agent functions
            if __CTX_VARS_NAME__ in func.__code__.co_varnames:
                args[__CTX_VARS_NAME__] = context_variables
            raw_result = function_map[name](**args)

            result: Result = self.handle_function_result(raw_result, debug)
            partial_response.messages.append(
                {
                    "role": "tool",
                    "toolCallId": tool_call.id,
                    # "toolName": name,
                    "content": result.value,
                }
            )
            partial_response.context_variables.update(result.context_variables)
            if result.agent:
                partial_response.agent = result.agent

        return partial_response

    def run_and_stream(
        self,
        agent: Agent,
        messages: List,
        context_variables: dict = {},
        model_override: str = None,
        debug: bool = False,
        max_turns: int = float("inf"),
        execute_tools: bool = True,
    ):
        active_agent = agent
        context_variables = copy.deepcopy(context_variables)
        history = copy.deepcopy(messages)
        init_len = len(messages)

        while len(history) - init_len < max_turns:
            message = {
                "content": "",
                "sender": agent.name,
                "role": "assistant",
                "functionCall": None,
                "toolCalls": defaultdict(
                    lambda: {
                        "function": {"arguments": "", "name": ""},
                        "id": "",
                        "type": "",
                    }
                ),
            }

            # get completion with current history, agent
            completion, last_message_was_agent_transfer = self.get_chat_completion(
                agent=active_agent,
                history=history,
                context_variables=context_variables,
                model_override=model_override,
                stream=True,
                debug=debug,
            )

            yield {"delim": "start"}
            for chunk in completion:
                delta = chunk['choices'][0]['delta']
                if delta.get('role') == "assistant":
                    delta["sender"] = active_agent.name
                yield delta
                delta.pop("role", None)
                delta.pop("sender", None)
                merge_chunk(message, delta)
            yield {"delim": "end"}

            message["toolCalls"] = list(
                message.get("toolCalls", {}).values())
            if not message["toolCalls"]:
                message["toolCalls"] = None
            debug_print(debug, "Received completion:", message)
            history.append(message)

            if not message["toolCalls"] or not execute_tools or last_message_was_agent_transfer:
                debug_print(debug, "Ending turn.")
                break

            # convert tool_calls to objects
            tool_calls = []
            for tool_call in message["toolCalls"]:
                function = Function(
                    arguments=tool_call["function"]["arguments"],
                    name=tool_call["function"]["name"],
                )
                tool_call_object = ChatCompletionMessageToolCall(
                    id=tool_call["id"], function=function, type=tool_call["type"]
                )
                tool_calls.append(tool_call_object)

            # handle function calls, updating context_variables, and switching agents
            partial_response = self.handle_tool_calls(
                tool_calls, active_agent.functions, context_variables, debug
            )
            history.extend(partial_response.messages)
            context_variables.update(partial_response.context_variables)
            if partial_response.agent:
                active_agent = partial_response.agent
        
        yield {
            "response": Response(
                messages=history[init_len:],
                agent=active_agent,
                context_variables=context_variables,
            )
        }

    def run(
        self,
        agent: Agent,
        messages: list[dict[str, Any]],
        context_variables: dict[str, Any] = {},
        model_override: str | None = None,
        stream: bool = False,
        debug: bool = False,
        max_turns: int = float("inf"),
        execute_tools: bool = True,
    ) -> Response:
        if stream:
            return self.run_and_stream(
                agent=agent,
                messages=messages,
                context_variables=context_variables,
                model_override=model_override,
                debug=debug,
                max_turns=max_turns,
                execute_tools=execute_tools,
            )
        active_agent = agent
        context_variables = copy.deepcopy(context_variables)
        history = copy.deepcopy(messages)
        init_len = len(messages)

        while len(history) - init_len < max_turns and active_agent:
            # get completion with current history, agent
            completion, last_message_was_agent_transfer = self.get_chat_completion(
                agent=active_agent,
                history=history,
                context_variables=context_variables,
                model_override=model_override,
                stream=stream,
                debug=debug,
            )
            message = {
                "content": completion['choices'][0]['message'].get('content', ''),
                "role": completion['choices'][0]['message'].get('role', ''),
                "toolCalls": completion['choices'][0]['message'].get('toolCalls', None),
                "sender": active_agent.name
            }
            if message["toolCalls"]:
                # append sender in tool calls. This way we can keep track of who called what
                # when we have switched agents.
                for tool_call in message["toolCalls"]:
                    tool_call["sender"] = active_agent.name
            
            debug_print(debug, "Received completion:", message)
            history.append(message)
            
            if not message["toolCalls"] or not execute_tools or last_message_was_agent_transfer:
                debug_print(debug, "Ending turn.")
                break

            # Convert camelCase toolCalls to snake_case tool_calls format
            tool_calls = []
            if message["toolCalls"]:
                # print("Did have tool calls: ", message)
                for tool_call in message["toolCalls"]:
                    function = Function(
                        arguments=tool_call["function"]["arguments"],
                        name=tool_call["function"]["name"],
                    )
                    tool_call_object = ChatCompletionMessageToolCall(
                        id=tool_call["id"], 
                        function=function, 
                        type=tool_call["type"]
                    )
                    tool_calls.append(tool_call_object)

            # handle function calls, updating context_variables, and switching agents
            partial_response = self.handle_tool_calls(
                tool_calls, active_agent.functions, context_variables, debug
            )
            history.extend(partial_response.messages)
            context_variables.update(partial_response.context_variables)
            if partial_response.agent:
                active_agent = partial_response.agent
        return Response(
            messages=history[init_len:],
            agent=active_agent,
            context_variables=context_variables,
        )
