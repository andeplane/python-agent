from __future__ import annotations
from dataclasses import dataclass
from python_agent.reasoning.reasoning_base import ReasoningBase
from auth import create_client
from python_agent.llm import call_llm
from python_agent.reasoning.chain_of_thought.prompts import cot_system_prompt, planner_prompt, validate_prompt, final_answer_prompt
from typing import Any, TYPE_CHECKING
if TYPE_CHECKING:
    from python_agent.Agent import Agent

@dataclass
class ChainOfThought(ReasoningBase):
    def __init__(self, model: str, agent: Agent, debug: bool, log_file_name: str):
        super().__init__(model, agent, debug, log_file_name)
        self.cognite_client = create_client()

    def think(self, user_message: str, messages: list[dict[str, Any]]) -> str:
        # Reset all tool logs
        for tool in self.agent.tools:
            tool.reset_thought_log()

        with open(self.log_file_name, "a", encoding='utf-8') as f:
            f.write("User: " + user_message + "\n\n")

        is_answered = False
        thoughts: list[str] = []
        number_of_thoughts = 0

        system_prompt = cot_system_prompt
        system_prompt += "\n\nHere are extra instructions for the agent:\n"
        system_prompt += self.agent.get_system_prompt()
        cot_messages = [{"role": "system", "content": system_prompt}]
        cot_messages.extend(messages)
        cot_messages.append({"role": "user", "content": user_message})
        
        plan = self.create_plan(user_message, messages)
        
        cot_messages.append({"role": "assistant", "content": "I have created the following plan for this question: "+plan})
        cot_messages.append({"role": "assistant", "content": "I will now execute the plan step by step."})

        llm_tools = [llm_tool for tool in self.agent.tools for llm_tool in tool.get_llm_tools()]
        while True:
            number_of_thoughts += 1
            answer = call_llm([*cot_messages, {"role": "user", "content": "Here is what I have thought so far: " + "\n".join(thoughts)}], self.model, llm_tools)
            
            with open(self.log_file_name, "a", encoding='utf-8') as f:
                f.write("[Thinking ...]\n" + answer['content'] + "\n\n")
            
            thoughts.append(f"[Thought {number_of_thoughts}]:\nTools calls:\n{answer['tool_calls']}\n\nAnswer:\n{answer['content']}")
            is_answered = self.validate(user_message, messages, thoughts, plan)
            if is_answered == True or number_of_thoughts > 10:
                final_answer = self.formulate_final_answer(user_message, messages, thoughts)
                return final_answer
            else:
                assert isinstance(is_answered, str)
                thoughts.append("Feedback on thoughts so far: " + is_answered)
    
    def create_plan(self, user_message: str, messages: list[dict[str, Any]]) -> str:
        system_prompt = planner_prompt
        system_prompt += "\n\nTool specific instructions:\n"
        
        already_added_tools: set[str] = set()
        for tool in self.agent.tools:
            if tool.type in already_added_tools:
                continue
            already_added_tools.add(tool.type)

            if tool.system_prompt_contribution:
                system_prompt += tool.system_prompt_contribution+"\n\n"
                
        plan_messages = [{"role": "system", "content": system_prompt}]
        plan_messages.extend(messages)
        plan_messages.append({"role": "user", "content": "Create a plan on how to answer the following message: "+user_message})
        response = call_llm(plan_messages, self.model)
        
        with open(self.log_file_name, "a", encoding='utf-8') as f:
            f.write("[Agent plan ...]\n" + response['content'] + "\n\n")

        return response['content']

    def validate(self, user_message: str, messages: list[dict[str, Any]], thoughts: list[str], plan: str) -> bool | str:
        thoughts_str = "\n\n".join(thoughts)
        
        validate_messages = [{"role": "system", "content": validate_prompt}]
        validate_messages.extend(messages)
        validate_messages.append({"role": "user", "content": f"{user_message}"})
        validate_messages.append({"role": "assistant", "content": f"Agent plan to answer question: {plan}"})
        validate_messages.append({"role": "assistant", "content": f"Reasoning thoughts so far: {thoughts_str}"})
        tool_calls_summary: str = ""
        for tool in self.agent.tools:
            tool_calls_summary += tool.retrieve_current_thoughts_log() + "\n"
        validate_messages.append({"role": "assistant", "content": f"Tool calls so far: {tool_calls_summary}"})
        validate_messages.append({"role": "assistant", "content": (
            "If data may be required to answer this question, ensure that we tried that first. Provide feedback if not, so the other agent knows what to do."
            f"Answer only 'Yes' or 'No, here is why: <feedback on what's missing>', nothing else."
            f"Answer: "
        )})
        response = call_llm(validate_messages, self.model)['content']
        with open(self.log_file_name, "a", encoding='utf-8') as f:
            f.write("[Agent validation ...]\nAgent answer validation: " + response + "\n\n")
        if response and "Yes" in response:
            return True
        return response
    
    def formulate_final_answer(self, user_question: str, messages: list[dict[str, Any]], thoughts: list[str]) -> str:
        thoughts_str = "\n\n".join(thoughts)
        final_answer_messages = [{"role": "system", "content": final_answer_prompt}]
        final_answer_messages.extend(messages)
        final_answer_messages.append({"role": "user", "content": user_question})
        final_answer_messages.append({"role": "assistant", "content": f"Thoughts and reflections:\n{thoughts_str}\n\n"})
        final_answer_messages.append({"role": "user", "content": "Write a final answer to the question based on the information provided. Unless explicitly asked, do not include space and externalId in the answer."})

        response = call_llm(final_answer_messages, self.model)

        with open(self.log_file_name, "a", encoding='utf-8') as f:
            f.write("[Agent final answer ...]\n" + response['content'] + "\n\n")
        return response['content']
