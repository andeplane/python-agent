from dataclasses import dataclass
from openai.types.chat import ChatCompletionMessageParam
from python_agent.llm import chat_completion
from python_agent.reasoning.reasoning_base import ReasoningBase

@dataclass
class PlainReasoning(ReasoningBase):
    model: str
    debug: bool = False
    system_prompt: str = "You are an helpful AI assistant"

    def think(self, messages: list[ChatCompletionMessageParam], user_message: str) -> str:
        answer = chat_completion(messages, user_message, self.system_prompt, self.model)
        if not answer:
            return "I am sorry, I could not find an answer to your query."
        return answer
        