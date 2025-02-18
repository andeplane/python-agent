from python_agent.Agent import Agent
# from python_agent.utils import send_cog_ai_request

agent = Agent.load("ai-bluefield", "anders-agent")

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
# agents = send_cog_ai_request("/api/v1/projects/ai-bluefield/ai/agents/list", "GET").json()
# print(json.dumps(agents, indent=2))