from dataclasses import dataclass
from python_agent.reasoning.reasoning_base import ReasoningBase
from python_agent.swarm import Swarm, Agent
from auth import create_client
from typing import Callable

@dataclass
class PlainReasoning(ReasoningBase):
    model: str
    debug: bool = False
    
    def __init__(self, model: str, tools: list[Callable], debug: bool):
        super().__init__(model, tools, debug)

    def think(self, messages, user_message: str) -> str:
        cognite_client = create_client()
        client = Swarm(client=cognite_client)

        agent = Agent(
            client=client,
            functions=self.tools
        )
        messages.append({"role": "user", "content": user_message})

        answer = client.run(
            agent=agent,
            model_override=self.model,
            debug=self.debug,
            messages=messages,
        )
        
        if not answer:
            return "I am sorry, I could not find an answer to your query."
        return answer
        