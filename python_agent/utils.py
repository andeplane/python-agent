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

def send_cog_ai_request(
    url: str,
    method: str,
    payload: dict[str, Any] | None = None,
    is_alpha: bool = True,
):
    client = create_client()
    request_kwargs = {}
    if payload:
        request_kwargs["json"] = payload
    if is_alpha:
        request_kwargs["headers"] = {"cdf-version": "alpha"}
    request_kwargs["url"] = url
    if method == "POST":
        method_fn = client.post
    elif method == "GET":
        method_fn = client.get
    return method_fn(**request_kwargs)