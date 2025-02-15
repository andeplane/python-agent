import logging
from dataclasses import dataclass
from python_agent.reasoning.reasoning_base import ReasoningBase
from python_agent.swarm import Swarm, Agent
from auth import create_client
from typing import Callable
import datetime

logger = logging.getLogger('cot')

@dataclass
class ChainOfThought(ReasoningBase):
    agent: Agent = None

    def __init__(self, model: str, tools: list[Callable], debug: bool, log_file_name: str):
        super().__init__(model, tools, debug, log_file_name)
        cognite_client = create_client()
        self.client = Swarm(client=cognite_client)

        self.agent = Agent(
            client=self.client,
            model=model,
            functions=self.tools
        )

        self.plain_agent = Agent(
            client=self.client,
            model=model
        )
        
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
        "<tool instructions>"
        "When using query knowledge graph to query data, express the query in natural language properly. Include operation (list, search or aggregate), and be expressive. Examples:"
        "Search for asset X"
        "Search for time series for asset with space <SPACE> and external id <EXTERNAL_ID>"
        "List children of asset with space <SPACE> and external id <EXTERNAL_ID>"
        "Aggregate data for time series with space <SPACE> and external id <EXTERNAL_ID> over the last 10 minutes"
        "<tool instructions end>"
        "If you have tried something before and it didn't work, you can try one more time, but then move to another strategy."
        "Insanity is doing the same thing over and over again and expecting different results."
        "Use the tools you have available to you. You can use the same tool multiple times if needed. Make guesses when you are not sure. It is always better to try than to fail."
        "Make hypotheses on what to use to answer the question."
        "Then thinking, repeat e.g. how the data was found, which tools that were used etc."
        "When a query fails, try again. It may work the next time."
        "You should normally navigate the knowledge graph using assets, or direct searches on e.g. assets."
        "If you are unsure what to search for, just list assets and time series and see what is there. e.g. `list assets` or `list time series`"
        "Current timestamp is "+datetime.datetime.now().isoformat()
    )
    
    default_system_prompt: str = "You are an helpful AI assistant."
    
    def create_plan(self, messages, user_message: str) -> str:
        plan_messages = [{"role": "system", "content": (
            "You are an agent that creates plans for how to answer questions."
            "You are in Cognite Data Fusion and is an expert on the platform and industrial data."
            "If a question may require querying to find an answer, make a plan on how to do that."
            "If user is just having a conversation, plan may be short.")
            }]
        plan_messages.extend(messages)
        plan_messages.append({"role": "user", "content": "User message: "+user_message})

        response = self.client.run(
            agent=self.plain_agent,
            model_override=self.model,
            debug=self.debug,
            messages=plan_messages,
        )
        response = response.messages[-1]['content']
        with open(self.log_file_name, "a", encoding='utf-8') as f:
            f.write("  [Thinking ...] Agent plan: " + response)

        return response

    def think(self, messages, user_message: str) -> str:
        with open(self.log_file_name, "a", encoding='utf-8') as f:
            f.write("User: " + user_message + "\n\n")

        is_answered = False
        thoughts: list[str] = []
        number_of_thoughts = 0

        think_messages = [{"role": "system", "content": self.cot_system_prompt}]
        think_messages.extend(messages)
        think_messages.append({"role": "user", "content": user_message})
        
        plan = self.create_plan(messages, user_message)
        think_messages.append({"role": "assistant", "content": "I have created the following plan for this question: "+plan})
        think_messages.append({"role": "assistant", "content": "I will now execute the plan step by step."})
        while True:
            number_of_thoughts += 1
            answer = self.client.run(
                agent=self.agent,
                model_override=self.model,
                debug=self.debug,
                messages=[*think_messages, {"role": "user", "content": "Here is what I have thought so far: " + "\n\n".join(thoughts)}]
            )
            if not answer:
                continue
            for message in answer.messages:
                with open(self.log_file_name, "a", encoding='utf-8') as f:
                    f.write(" ** Agent thinking: " + message['content'] + "\n\n")
            # print(" ** Agent thinking: ", [message['content'] for message in  answer.messages])
            thoughts.append(answer.messages[-1]['content'])
            is_answered = self.validate(messages, user_message, thoughts, plan)
            if is_answered == True or number_of_thoughts > 10:
                final_answer = self.formulate_final_answer(messages, user_message, thoughts)
                with open(self.log_file_name, "a", encoding='utf-8') as f:
                    f.write("Agent: " + final_answer + "\n\n")
                return final_answer
            else:
                thoughts.append("Feedback on thoughts so far: " + is_answered)
            
    def validate(self, messages, user_message: str, thoughts: list[str], plan: str) -> bool | str:
        thoughts_str = "\n\n".join(thoughts)
        
        validate_messages = []
        validate_messages.append({"role": "system", "content": (
            "You are a conversation reviewer in Cognite Data Fusion. You will make sure that an agent is able to help the user with their question and conversations."
            "You will be given a message from a user, a set of reasoning thoughts. Based on this, you will reply Yes or No if the thought process includes everything we need to formulate an answer."
            "If the user asks for a specific question about data, you should see if the thought process includes the answer with a high confidence."
            "If the user is just having conversation, you should see if the thought process includes sufficient information to continue conversation."
            "Questions often requires that we query Cognite Data Fusion, so anytime a question may require data, ensure that we tried that first."
            "If you see indications that data has been fetched, it has been fetched from Cognite Data Fusion, so that may be ok."
        )})
        validate_messages.append({"role": "user", "content": f"User question: '{user_message}'."})
        validate_messages.append({"role": "user", "content": f"Plan to answer question: {plan}"})
        validate_messages.append({"role": "user", "content": f"Reasoning thoughts: {thoughts_str}"})

        validate_messages.append({"role": "user", "content": (
            "If data may be required to answer this question, ensure that we tried that first. Provide feedback if not, so the other agent knows what to do."
            f"Answer only 'Yes' or 'No, here is why: <feedback on what's missing>', nothing else."
            f"Answer: "
        )})
        
        # print(" ** Validate messages: ", validate_messages)
        response = self.client.run(
            agent=self.plain_agent,
            model_override=self.model,
            debug=self.debug,
            messages=validate_messages,
        )
        response = response.messages[-1]['content']
        with open(self.log_file_name, "a", encoding='utf-8') as f:
            f.write(" [Thinking ...] Agent answer validation: " + response + "\n\n")
        if response and "Yes" in response:
            return True
        return response
    
    def formulate_final_answer(self, messages, user_question: str, thoughts: list[str]) -> str:
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

        final_answer_messages = [
            {"role": "system", "content": "You are given a question and a set of thoughts. Based on this, you should formulate a final answer to the question, using the previous thoughts you had. You will refer to the thoughts for facts or reasoning."},
            *messages,
            {"role": "user", "content": prompt}
        ]
        
        response = self.client.run(
            agent=self.plain_agent,
            model_override=self.model,
            debug=self.debug,
            messages=final_answer_messages,
        )
        response = response.messages[-1]['content']
        if response:
            return response.strip()
        return "I'm sorry, I couldn't formulate a response based on the information provided."
