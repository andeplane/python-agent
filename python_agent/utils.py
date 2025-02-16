from auth import create_client
from python_agent.swarm import Swarm, Agent 
from typing import Any
def create_task_agent(model: str, log_file_name: str, system_prompt: str):
    cognite_client = create_client()
    client = Swarm(client=cognite_client)

    agent = Agent(
        model=model,
        instructions=system_prompt
    )

    def call_llm(messages: list[dict[str, Any]]):
        return client.run(
            agent=agent, 
            messages=messages
        )

    return call_llm