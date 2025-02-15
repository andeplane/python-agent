from abc import ABC, abstractmethod
from typing import Callable, Any
from python_agent.tools.AgentTool import AgentTool

class ReasoningBase(ABC):
    model: str
    tools: list[dict[str, Callable[[], Any] | AgentTool]]
    debug: bool
    log_file_name: str

    def __init__(self, model: str, tools: list[dict[str, Callable[[], Any] | AgentTool]], debug: bool, log_file_name: str):
        self.model = model
        self.tools = tools
        self.debug = debug
        self.log_file_name = log_file_name

    @abstractmethod
    def think(self, messages: list[dict[str, Any]], user_message: str) -> str:
        pass