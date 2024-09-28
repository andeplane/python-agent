from dataclasses import dataclass
from openai.types.chat import ChatCompletionMessageParam
from python_agent.llm import call_openai
from python_agent.reasoning.reasoning_base import ReasoningBase

@dataclass
class ChainOfThought(ReasoningBase):
    model: str
    debug: bool = False
    cot_system_prompt: str = (
            "You are an AI assistant that uses a Chain of Thought (CoT) approach with reflection to answer queries. Follow these steps:"
            ""
            "1. Think through the problem step by step within the <thinking> tags."
            "2. Reflect on your thinking to check for any errors or improvements within the <reflection> tags."
            "3. Make any necessary adjustments based on your reflection."
            "4. Provide your final, concise answer within the <output> tags."
            ""
            "Important: The <thinking> and <reflection> sections are for your internal reasoning process only. "
            "Do not include any part of the final answer in these sections. "
            "The actual response to the query must be entirely contained within the <output> tags."
            ""
            "Use the following format for your response:"
            "<thinking>"
            "[Your step-by-step reasoning goes here. This is your internal thought process, not the final answer.]"
            "<reflection>"
            "[Your reflection on your reasoning, checking for errors or improvements]"
            "</reflection>"
            "[Any adjustments to your thinking based on your reflection]"
            "</thinking>"
            "<output>"
            "[Your final, concise answer to the query. This is the only part that will be shown to the user.]"
            "</output>"
        )
    default_system_prompt: str = "You are an helpful AI assistant."
    def think(self, messages: list[ChatCompletionMessageParam], user_message: str) -> str:
        is_answered = False
        thoughts: list[str] = []
        while True:
            answer = call_openai(messages, user_message, self.cot_system_prompt, self.model)
            if not answer:
                continue
            if self.debug:
                print(f"Thinking :\n{answer}\n")
            thoughts.append(answer)
            is_answered = self.validate(messages, user_message, thoughts)
            if is_answered:
                return self.formulate_final_answer(messages, user_message, thoughts)
            else:
                print(f"Reasoning did not contain the answer")
    
    def validate(self, messages: list[ChatCompletionMessageParam], user_message: str, thoughts: list[str]) -> bool:
        thoughts_str = "\n\n".join(thoughts)
        prompt = (
            f"Here is the thinking process for the question '{user_message}'.\n\n"
            "Based on the following conversation history\n\n"
            f"{thoughts_str}"
            f"Is the user question {user_message} answered? Answer only 'Yes' or 'No', nothing else."
        )
        response = call_openai(
            messages, 
            prompt, 
            "You are an AI assistant that will validate if a set if reasoning thoughts are answering a user question.",
            self.model
            )

        if self.debug and response:
            print(f"Validation response: {response}")
        if response and "Yes" in response:
            return True
        return False
    
    def formulate_final_answer(self, messages: list[ChatCompletionMessageParam], user_question: str, thoughts: list[str]) -> str:
        """
        Compiles all execution results into a final, coherent answer.

        Args:
            user_question (str): The original user question.
            thoughts (List[str]): The results from executing all plan steps.

        Returns:
            str: The final answer to the user.
        """
        thoughts_str = "\n\n".join(thoughts)
        prompt = (
            f"Given the following question {user_question}, and the following thinking, formulate a clear and concise answer to the user.\n\n"
            f"Compiled Information:\n{thoughts_str}\n\n"
            "Final Answer:"
        )

        response = call_openai(
            messages,
            prompt, 
            "You are an AI assistant that will formulate a final answer based on the thinking process.",
            self.model
            )
        if response:
            return response.strip()
        return "I'm sorry, I couldn't formulate a response based on the information provided."
