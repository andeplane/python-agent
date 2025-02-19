from __future__ import annotations
from dataclasses import dataclass
from python_agent.reasoning.reasoning_base import ReasoningBase
from typing import Any, TYPE_CHECKING
from python_agent.llm import call_llm
if TYPE_CHECKING:
    from python_agent.Agent import Agent

@dataclass
class PlainReasoning(ReasoningBase):
    def __init__(self, model: str, agent: Agent, debug: bool, log_file_name: str):
        super().__init__(model, agent, debug, log_file_name)

    def think(self, user_message: str, messages: list[dict[str, Any]]) -> str:
        new_messages = messages.copy()
        system_prompt = self.agent.get_system_prompt()
        new_messages.append({"role": "system", "content": system_prompt})
        new_messages.append({"role": "user", "content": user_message})
        llm_tools = [llm_tool for tool in self.agent.tools for llm_tool in tool.get_llm_tools()]
        response = call_llm(new_messages, self.model, llm_tools)
        return response["content"]
        