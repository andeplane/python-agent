from abc import ABC, abstractmethod
from openai.types.chat import ChatCompletionMessageParam

class ReasoningBase(ABC):
    @abstractmethod
    def think(self, messages: list[ChatCompletionMessageParam], user_message: str) -> str:
        pass