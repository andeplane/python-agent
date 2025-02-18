from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from python_agent.Agent import Agent

class ReasoningBase(ABC):
    model: str
    agent: Agent
    debug: bool
    log_file_name: str

    def __init__(self, model: str, agent: Agent, debug: bool, log_file_name: str):
        self.model = model
        self.agent = agent
        self.debug = debug
        self.log_file_name = log_file_name

    @abstractmethod
    def think(self, user_message: str, messages: list[dict[str, Any]]) -> str:
        pass