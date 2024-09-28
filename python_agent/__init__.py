from python_agent.agent import Agent
def chat_interface():
    print("Welcome to the terminal chat! Type 'exit' to quit.")
    agent = Agent(model = "gpt-4o-mini", debug=False)
    while True:
        # Take user input
        user_input = input("You: ")
        # user_input = "How many Rs in strawberry?"
        
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        # Call the LLM agent with user input
        agent_response = agent.chat(user_input)
        
        # Display agent response
        print(f"Agent: {agent_response}\n")
        return

if __name__ == "__main__":
    chat_interface()