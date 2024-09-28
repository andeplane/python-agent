from dataclasses import dataclass
from openai import OpenAI
from typing import Literal

from openai.types.chat import ChatCompletionMessageParam, ChatCompletionMessage
client = OpenAI()
@dataclass
class ChatMessageContent:
    type: Literal["text"]
    text: str
@dataclass
class ChatMessage:
    role: Literal["user", "assistant", "system"]  
    content: list[ChatMessageContent]
    def to_openai(self) -> ChatCompletionMessageParam:
        return {
            "role": "user",
            "content": [{"type": content.type, "text": content.text} for content in self.content]
        }

class Agent:
    def __init__(self, model: str = "gpt-4o"):
        self.client = OpenAI()
        self.model = model
        self.messages: list[ChatCompletionMessageParam] = []
    
    def chat(self, message: str) -> str:
        self.messages.append(ChatMessage("user", [ChatMessageContent("text", message)]).to_openai())

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=1,
            max_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            response_format={
                "type": "text"
            }
        )
        if len(response.choices) == 0:
            raise ValueError("No response from the API")
        if response.choices[0].message.content == None:
            raise ValueError("No response from the API")
        
        answer = response.choices[0].message.content
        self.messages.append(ChatMessage("assistant", [ChatMessageContent("text", answer)]).to_openai())
        return answer