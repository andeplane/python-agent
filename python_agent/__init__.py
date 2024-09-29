import readline
from python_agent.agent import Agent, ReasoningStrategy
from python_agent.llm import chat_completion
import logging

# logging.basicConfig(filename="agent.log",
#     filemode='a',
#     format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
#     datefmt='%H:%M:%S',
#     level=logging.DEBUG
# )

def chat_interface():
    print("Welcome to the terminal chat! Type 'exit' to quit.")
    # agent = Agent(model = "gpt-4o-mini", debug=False, reasoning_strategy=ReasoningStrategy.PLAIN)
    agent = Agent(model = "ollama/llama3.2", debug=False, reasoning_strategy=ReasoningStrategy.COT)
    while True:
        # Take user input
        user_input = input("You: ")
        
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        # Call the LLM agent with user input
        agent_response = agent.chat(user_input)
        
        # Display agent response
        print(f"Agent: {agent_response}\n")

if __name__ == "__main__":
    # print(chat_completion([], "What are you", "", "ollama/llama3.2"))
    chat_interface()