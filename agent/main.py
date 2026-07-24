from agent import SimpleAgent

agent = SimpleAgent()

while True:
    user_input = input("User: ")

    if user_input.strip().lower() == "exit":
        break

    response = agent.think(user_input)
    print(f"Agent: {response}\n")