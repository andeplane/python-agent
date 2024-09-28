from openai import OpenAI
from typing import Optional, List

from openai.types.chat import ChatCompletionMessageParam
import re

class Agent:
    def __init__(self, model: str = "gpt-4o-mini", debug: bool = False):
        """
        Initializes the Agent.

        Args:
            model (str): The OpenAI model to use.
            debug (bool): If True, enables debug mode to print all LLM interactions.
        """
        self.client = OpenAI()
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
        self.messages: List[ChatCompletionMessageParam] = []

    def chat(self, message: str) -> str:
        """
        Receives a user message, appends it to the conversation history, invokes the
        think method to process the message, and returns the assistant's reply.

        Args:
            message (str): The user's message.

        Returns:
            str: The assistant's response.
        """
        self.messages.append({"role": "user", "content": message})
        if self.debug:
            print(f"User: {message}")
        answer = self.think(message)
        return answer

    def validate(self, user_message: str, thoughts: list[str]) -> bool:
        thoughts_str = "\n\n".join(thoughts)
        prompt = (
            f"Here is the thinking process for the question '{user_message}'.\n\n"
            "Based on the following conversation history\n\n"
            f"{thoughts_str}"
            f"Is the user question {user_message} answered? Answer only 'Yes' or 'No', nothing else."
        )
        response = self.call_openai(prompt, "You are an AI assistant that will validate if a set if reasoning thoughts are answering a user question.")

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
            answer = self.call_openai(user_message, self.cot_system_prompt)
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

        response = self.call_openai(prompt)
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

        response = self.call_openai(prompt, "You are an AI assistant that will formulate a final answer based on the thinking process.")
        if response:
            return response.strip()
        return "I'm sorry, I couldn't formulate a response based on the information provided."

    def call_openai(self, prompt: str, system_prompt: str | None = None) -> Optional[str]:
        """
        Calls the OpenAI API with the given prompt and returns the response.

        Args:
            prompt (str): The prompt to send to the OpenAI API.

        Returns:
            Optional[str]: The API's response or None if failed.
        """
        if not system_prompt:
            system_prompt = self.default_system_prompt
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            if response and response.choices:
                if not response.choices[0].message.content:
                    raise ValueError("OpenAI API response is empty.")
                llm_response = response.choices[0].message.content.strip()
                if self.debug:
                    print(f"Prompt Sent:\n{prompt}\n")
                    print(f"LLM Response:\n{llm_response}\n")
                return llm_response
        except Exception as e:
            print(f"OpenAI API call failed: {str(e)}")
        return None

    def extract_list_from_text(self, text: str) -> Optional[List[str]]:
        """
        Extracts a list of items from the given text.

        Args:
            text (str): The text containing a list.

        Returns:
            Optional[List[str]]: A list of extracted items or None if extraction fails.
        """
        # Match numbered lists (e.g., 1. Item)
        pattern = re.compile(r'^\s*\d+\.\s*(.+)', re.MULTILINE)
        matches = pattern.findall(text)
        if matches:
            return [match.strip() for match in matches]
        
        # If no numbered list, try bulleted list
        pattern = re.compile(r'^\s*[-*]\s*(.+)', re.MULTILINE)
        matches = pattern.findall(text)
        if matches:
            return [match.strip() for match in matches]
        
        # If no list found, return the entire text as a single step
        if text:
            return [text.strip()]
        
        return None

    def format_messages(self) -> str:
        """
        Formats the conversation history into a readable string.

        Returns:
            str: The formatted conversation history.
        """
        return "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in self.messages]) # type: ignore