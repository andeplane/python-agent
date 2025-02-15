from enum import Enum
from python_agent.reasoning.chain_of_thought.chain_of_thought import ChainOfThought
from python_agent.reasoning.plain import PlainReasoning
from python_agent.reasoning.reasoning_base import ReasoningBase
from typing import Callable, Any, Dict
from python_agent.tools.AgentTool import AgentTool
import logging

logger = logging.getLogger('agent')

class ReasoningStrategy(Enum):
    PLAIN = "Plain"
    COT = "Chain of Thought"

class Agent:
    model: str
    debug: bool
    tools: list[Dict[str, Callable[[], Any] | AgentTool]]
    reasoning_engine: ReasoningBase
    log_file_name: str

    def __init__(self, model: str = "gpt-4o-mini", reasoning_strategy: ReasoningStrategy = ReasoningStrategy.PLAIN, debug: bool = False, tools: list[Callable] = [], log_file_name: str = "agent_log.log"):
        """
        Initializes the Agent.

        Args:
            model (str): The OpenAI model to use.
            debug (bool): If True, enables debug mode to print all LLM interactions.
        """
        self.model = model
        self.debug = debug
        self.tools = tools
        self.log_file_name = log_file_name
        if reasoning_strategy == ReasoningStrategy.COT:
            self.reasoning_engine = ChainOfThought(self.model, self.tools, self.debug, self.log_file_name)
        elif reasoning_strategy == ReasoningStrategy.PLAIN:
            self.reasoning_engine = PlainReasoning(self.model, self.tools, self.debug, self.log_file_name)

        self.messages = []

    def chat(self, message: str) -> str:
        """
        Receives a user message, appends it to the conversation history, invokes the
        think method to process the message, and returns the assistant's reply.

        Args:
            message (str): The user's message.

        Returns:
            str: The assistant's response.
        """
        if self.debug:
            print(f"User: {message}")
        
        answer = self.reasoning_engine.think(self.messages, message)
        
        self.messages.append({"role": "user", "content": message})
        self.messages.append({"role": "assistant", "content": answer})
        return answer
