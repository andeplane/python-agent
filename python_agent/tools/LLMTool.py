from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from python_agent.tools.AgentTool import AgentTool

class LLMTool(ABC):
    agent_tool: AgentTool
    
    @abstractmethod
    def invoke(self, **kwargs: Any) -> Any:
        """
        Invoke the tool with the provided parameters.
        """
        pass

    @abstractmethod
    def get_json_schema(self) -> Dict[str, Any]:
        """
        Return the JSON schema describing the parameters accepted by this tool.
        """
        pass