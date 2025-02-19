from python_agent.Agent import Agent, ReasoningStrategy, AgentParameters
# from python_agent.utils import send_cog_ai_request

parameters = AgentParameters(
    reasoning_strategy=ReasoningStrategy.COT,
    max_thoughts=10,
)

agent = Agent.load("ai-bluefield", "anders-agent", parameters)

import sys

if len(sys.argv) > 1:
    # Get input from command line argument
    user_input = " ".join(sys.argv[1:])
    response = agent.think(user_input)
    print(f"Agent: {response}")
else:
    # Interactive mode
    print("Welcome to the terminal chat! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        response = agent.think(user_input)
        print(f"Agent: {response}")
