import logging
from dataclasses import dataclass
from python_agent.reasoning.reasoning_base import ReasoningBase
from python_agent.swarm import Swarm, Agent
from auth import create_client
from typing import Callable, Any
from python_agent.reasoning.chain_of_thought.prompts import cot_system_prompt, planner_prompt, validate_prompt, final_answer_prompt

logger = logging.getLogger('cot')

def create_task_agent(model: str, log_file_name: str, system_prompt: str):
    cognite_client = create_client()
    client = Swarm(client=cognite_client)

    agent = Agent(
        model=model,
        instructions=system_prompt
    )

    def call_llm(messages: list[dict]):
        return client.run(
            agent=agent, 
            messages=messages
        )

    return call_llm

@dataclass
class ChainOfThought(ReasoningBase):
    agent: Agent = None

    def __init__(self, model: str, tools: list[Callable[[], Any]], debug: bool, log_file_name: str):
        super().__init__(model, tools, debug, log_file_name)
        cognite_client = create_client()
        self.client = Swarm(client=cognite_client)

        self.agent = Agent(
            model=model,
            instructions=cot_system_prompt,
            functions=self.tools
        )

        self.plain_agent = Agent(
            model=model
        )
        
    def think(self, messages: list[dict[str, Any]], user_message: str) -> str:
        with open(self.log_file_name, "a", encoding='utf-8') as f:
            f.write("User: " + user_message + "\n\n")

        is_answered = False
        thoughts: list[str] = []
        number_of_thoughts = 0

        call_cot_llm = cognite_client = create_client()
        client = Swarm(client=cognite_client)
        agent = Agent(
            model=self.model,
            instructions=cot_system_prompt,
            functions=self.tools
        )

        cot_messages = messages.copy()
        cot_messages.append({"role": "user", "content": user_message})
        
        plan = self.create_plan(messages, user_message)
        
        cot_messages.append({"role": "assistant", "content": "I have created the following plan for this question: "+plan})
        cot_messages.append({"role": "assistant", "content": "I will now execute the plan step by step."})

        while True:
            number_of_thoughts += 1
            answer = client.run(
                agent=agent,
                messages=[*cot_messages, {"role": "user", "content": "Here is what I have thought so far: " + "\n\n".join(thoughts)}]
            )

            if not answer:
                continue

            for message in answer.messages:
                with open(self.log_file_name, "a", encoding='utf-8') as f:
                    f.write(" ** Agent thinking: " + message['content'] + "\n\n")
            
            thoughts.append(answer.messages[-1]['content'])
            is_answered = self.validate(messages, user_message, thoughts, plan)
            if is_answered == True or number_of_thoughts > 10:
                final_answer = self.formulate_final_answer(messages, user_message, thoughts)
                with open(self.log_file_name, "a", encoding='utf-8') as f:
                    f.write("Agent: " + final_answer + "\n\n")
                return final_answer
            else:
                thoughts.append("Feedback on thoughts so far: " + is_answered)
    
    def create_plan(self, messages: list[dict[str, Any]], user_message: str) -> str:
        call_llm = create_task_agent(self.model, self.log_file_name, planner_prompt)
        planner_messages = [
            *messages, 
            {"role": "user", "content": "Create a plan on how to answer the following message: "+user_message},
        ]
        
        response = call_llm(planner_messages)
        
        response = response.messages[-1]['content']
        
        with open(self.log_file_name, "a", encoding='utf-8') as f:
            f.write("  [Thinking ...] Agent plan: " + response)

        return response

    def validate(self, messages: list[dict[str, Any]], user_message: str, thoughts: list[str], plan: str) -> bool | str:
        thoughts_str = "\n\n".join(thoughts)
        
        validate_messages: list[dict[str, Any]] = []
        
        validate_messages.append({"role": "user", "content": f"User question: '{user_message}'."})
        validate_messages.append({"role": "user", "content": f"Plan to answer question: {plan}"})
        validate_messages.append({"role": "user", "content": f"Reasoning thoughts: {thoughts_str}"})

        validate_messages.append({"role": "user", "content": (
            "If data may be required to answer this question, ensure that we tried that first. Provide feedback if not, so the other agent knows what to do."
            f"Answer only 'Yes' or 'No, here is why: <feedback on what's missing>', nothing else."
            f"Answer: "
        )})
        
        call_llm = create_task_agent(self.model, self.log_file_name, validate_prompt)
        response = call_llm(validate_messages)
        response = response.messages[-1]['content']

        with open(self.log_file_name, "a", encoding='utf-8') as f:
            f.write(" [Thinking ...] Agent answer validation: " + response + "\n\n")
        if response and "Yes" in response:
            return True
        return response
    
    def formulate_final_answer(self, messages: list[dict[str, Any]], user_question: str, thoughts: list[str]) -> str:
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
            f"Question: {user_question}\n\n"
            f"Compiled Information:\n{thoughts_str}\n\n"
            "Final Answer:"
        )

        final_answer_messages: list[dict[str, Any]] = [
            *messages,
            {"role": "user", "content": prompt}
        ]
        call_llm = create_task_agent(self.model, self.log_file_name, final_answer_prompt)
        response = call_llm(final_answer_messages)

        response = response.messages[-1]['content']
        
        if response:
            return response.strip()
        return "I'm sorry, I couldn't formulate a response based on the information provided."
