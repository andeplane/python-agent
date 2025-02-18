from dataclasses import dataclass, field
from typing import Dict, Any, List
from python_agent.tools.LLMTool import LLMTool
@dataclass
class AgentTool:
    externalId: str
    name: str
    type: str
    instructions: str
    configuration: Dict[str, Any]
    log_file_name: str
    system_prompt_contribution: str | None = field(default=None)

    def __post_init__(self):
        self.current_thought_log: List[str] = []
        self.all_thoughts_log: List[str] = []

    def reset_thought_log(self):
        self.all_thoughts_log.extend(self.current_thought_log)
        self.current_thought_log = []

    def retrieve_current_thoughts_log(self) -> str:
        return '\n'.join(self.current_thought_log)
    
    def get_llm_tools(self) -> List[LLMTool]:
        """Subclasses must implement this to expose one or more LLM tools."""
        raise NotImplementedError("Subclasses must implement get_llm_tools()")
