from enum import Enum
from python_agent.reasoning.chain_of_thought.chain_of_thought import ChainOfThought
from python_agent.reasoning.plain import PlainReasoning
from python_agent.reasoning.reasoning_base import ReasoningBase
from typing import Any, Dict
from python_agent.tools.AgentTool import AgentTool
from python_agent.tools.tool_factory import create_agent_tool
from python_agent.utils import send_cog_ai_request
import logging
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger('agent')

class ReasoningStrategy(Enum):
    PLAIN = "Plain"
    COT = "Chain of Thought"

@dataclass
class AgentParameters:
    reasoning_strategy: ReasoningStrategy
    max_thoughts: int = 10
    debug: bool = False
    log_file_name: str = "agent.log"


@dataclass
class Agent:
    externalId: str
    instructions: str
    name: str
    description: str
    model: str
    reasoning_strategy: ReasoningStrategy
    reasoning_engine: ReasoningBase = field(init=False)
    exampleQuestions: list[Dict[str, Any]] = field(default_factory=list)
    tools: list[AgentTool] = field(default_factory=list)
    max_thoughts: int = 10
    ownerId: str = ""
    # Optional internal fields
    debug: bool = False
    log_file_name: str = ""

    def __post_init__(self):
        self.reasoning_engine = PlainReasoning(
            model=self.model,
            agent=self,
            debug=self.debug,
            log_file_name=self.log_file_name
        )
        if self.reasoning_strategy == ReasoningStrategy.PLAIN:
            self.reasoning_engine = PlainReasoning(
                model=self.model,
                agent=self,
                debug=self.debug,
                log_file_name=self.log_file_name
            )
        elif self.reasoning_strategy == ReasoningStrategy.COT:
            self.reasoning_engine = ChainOfThought(
                model=self.model,
                agent=self,
                debug=self.debug,
                log_file_name=self.log_file_name
            )
    
    def think(self, user_message: str, messages: list[dict[str, Any]] = []) -> str:
        return self.reasoning_engine.think(user_message, messages)
    
    def get_system_prompt(self) -> str:
        current_local_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_timezone = datetime.now().astimezone().tzname()
        system_prompt: str = f"""# Context
Current local time: ${current_local_time}
Time zone: ${current_timezone}

# General Instructions
You are an industrial AI agent, and expert on Cognite Data Fusion. 
You are an agent in the Fusion app and will help users finding data and solving problems with it.

When getting large amount of instances, do not repeat all. If there are more than 5, mention how many and display some of them.
If you get errors from API, try reformulations of the question and try at least 3 times.

# Agent specific instructions
{self.instructions}

# Tool specific instructions
"""
        already_added_tools: set[str] = set()
        for tool in self.tools:
            if tool.type in already_added_tools:
                continue
            already_added_tools.add(tool.type)

            if tool.system_prompt_contribution:
                system_prompt += tool.system_prompt_contribution+"\n\n"
        return system_prompt
    @classmethod
    def from_json(cls, data: Dict[str, Any], parameters: AgentParameters) -> "Agent":
        # Convert each tool dict into an AgentTool instance.
        tools_data = data.get("tools", [])
        tools = [t for t in (create_agent_tool(tool, parameters.log_file_name) for tool in tools_data) if t is not None]
        
        return cls(
            externalId=data.get("externalId", ""),
            instructions=data.get("instructions", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            model=data.get("model", ""),
            exampleQuestions=data.get("exampleQuestions", []),
            tools=tools,
            ownerId=data.get("ownerId", ""),
            log_file_name=parameters.log_file_name,
            reasoning_strategy=parameters.reasoning_strategy,
            max_thoughts=parameters.max_thoughts,
            debug=parameters.debug
        )

    @classmethod
    def load(cls, project: str, identifier: str, parameters: AgentParameters) -> "Agent":
        url = f"/api/v1/projects/{project}/ai/agents/byids"
        # Assuming send_cog_ai_request is defined elsewhere in your code.
        resp = send_cog_ai_request(url, "POST", payload={'items': [{'externalId': identifier}]})
        resp.raise_for_status()
        data = resp.json()
        items = data.get("items", [])
        if not items:
            raise Exception(f"Agent {identifier} not found")
        agent_data = items[0]
        return cls.from_json(agent_data, parameters)