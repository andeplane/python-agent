from python_agent.llm import call_openai
from enum import Enum

from openai.types.chat import ChatCompletionMessageParam
import re

class Agent:
    def __init__(self, model: str = "gpt-4o-mini",  debug: bool = False):
        """
        Initializes the Agent.

        Args:
            model (str): The OpenAI model to use.
            debug (bool): If True, enables debug mode to print all LLM interactions.
        """
        self.model = model
        self.debug = debug
        self.cot_system_prompt = (
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
        self.default_system_prompt = (
            "You are an helpful AI assistant."
        )
        self.messages: list[ChatCompletionMessageParam] = []

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
        self.messages.append({"role": "user", "content": message})
        
        answer = self.think(message)
        
        self.messages.append({"role": "assistant", "content": answer})
        return answer

    def validate(self, user_message: str, thoughts: list[str]) -> bool:
        thoughts_str = "\n\n".join(thoughts)
        prompt = (
            f"Here is the thinking process for the question '{user_message}'.\n\n"
            "Based on the following conversation history\n\n"
            f"{thoughts_str}"
            f"Is the user question {user_message} answered? Answer only 'Yes' or 'No', nothing else."
        )
        response = call_openai(
            self.messages, 
            prompt, 
            "You are an AI assistant that will validate if a set if reasoning thoughts are answering a user question.",
            self.model
            )

        if self.debug and response:
            print(f"Validation response: {response}")
        if response and "Yes" in response:
            return True
        return False
        pass
    
    def think(self, user_message: str) -> str:
        is_answered = False
        thoughts: list[str] = []
        while True:
            answer = call_openai(self.messages, user_message, self.cot_system_prompt, self.model)
            if not answer:
                continue
            if self.debug:
                print(f"Thinking :\n{answer}\n")
            thoughts.append(answer)
            is_answered = self.validate(user_message, thoughts)
            if is_answered:
                return self.formulate_final_answer(user_message, thoughts)
            else:
                print(f"Reasoning did not contain the answer")
        

    def verify_step(self, step: str, execution_result: str) -> bool:
        """
        Verifies whether the execution of a step has achieved its goal.

        Args:
            step (str): The plan step that was executed.
            execution_result (str): The result of executing the step.

        Returns:
            bool: True if the goal is achieved, False otherwise.
        """
        prompt = (
            "Evaluate whether the following execution result successfully achieves the goal of the step.\n\n"
            f"Step: {step}\n"
            f"Execution Result: {execution_result}\n\n"
            "Respond with 'Yes' if the goal is achieved, or 'No' otherwise."
        )

        response = call_openai(self.messages, prompt, self.default_system_prompt, self.model)
        if self.debug and response:
            print(f"Verification for Step '{step}': {response}")
        if response:
            return response.strip().lower() in ['yes', 'y']
        return False

    def formulate_final_answer(self, user_question: str, thoughts: list[str]) -> str:
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
            self.messages,
            prompt, 
            "You are an AI assistant that will formulate a final answer based on the thinking process.",
            self.model
            )
        if response:
            return response.strip()
        return "I'm sorry, I couldn't formulate a response based on the information provided."
