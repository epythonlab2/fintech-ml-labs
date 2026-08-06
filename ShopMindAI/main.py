import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from tools import get_order_status, process_return

load_dotenv()

tools = [get_order_status, process_return]

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=None
)

system_prompt = """You are ShopMind AI, an autonomous customer support agent for TechStore.
Always verify order details using tools before answering customer questions.
Be concise, accurate, and polite."""

agent_executor = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt
)

def get_clean_text(content):
    """Extract plain text whether content is a string or a list of blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block["text"] for block in content 
            if isinstance(block, dict) and "text" in block
        )
    return str(content)

if __name__ == "__main__":
    while True:
        user_input = input("Customer: ")
        if user_input.lower() == "exit":
            break
            
        response = agent_executor.invoke({"messages": [HumanMessage(content=user_input)]})
        
        # Extract and print only the final AI text response
        last_msg = response["messages"][-1]
        if last_msg.type == "ai" and last_msg.content:
            clean_text = get_clean_text(last_msg.content)
            print(f"\n[ShopMind AI]: {clean_text}\n")