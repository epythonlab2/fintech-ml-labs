## Build an Autonomous E-Commerce Support Agent with Gemini 2.5 & LangChain

### 1. Introduction

Building a simple chatbot is easy task today.

But building an AI agent that can check inventory, calculate refunds, track customer orders, and even search the web when it needs up-to-date information is a completely different challenge.

In this video, we are going to build **ShopMind AI**, a complete autonomous e commerce support system from scratch using **Google Gemini 3.6 Flash, LangChain, and LangGraph**.

Together, we will build an AI agent that can reason through customer requests, choose the right tools, retrieve live information, and provide accurate responses with minimal human intervention.

Along the way, you will learn how to:


✔️ Turn Python functions into LangChain tools

✔️ Build an autonomous agent with LangGraph

✔️ Connect the agent to the Gemini 3.6 Flash model

✔️ Create a complete ReAct agent workflow

✔️ Add guardrails for real world customer support scenarios


By the end of this tutorial, you will have a fully functional AI agent and a solid understanding of how modern agent frameworks combine reasoning, tool calling, and execution to solve real business problems.

Before we begin, please like this video, subscribe to the channel, and share your thoughts in the comments.

Now, let's start building.


### 2. System Architecture & Capabilities?
```bash
                  ┌────────────────────────┐
                  │ Customer Prompt        │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │  ShopMind AI Engine    │
                  │ (Gemini 3.6 Flash LLM) │
                  └───────────┬────────────┘
                              │
           /──────────── Select Tool ────────────\
          /                   │                   \
         ▼                    ▼                    ▼
┌─────────────────┐  ┌──────────────────┐  ┌─────────────────┐
│  Order Lookup   │  │ Calculate Refund │  │ Web Search Fall │
│  (Python Tool)  │  │  (Python Tool)   │  │ (Live Search)   │
└────────┬────────┘  └────────┬─────────┘  └────────┬────────┘
         │                    │                     │
         └────────────────────┼─────────────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │ ReAct Execution Loop   │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │ Customer Output        │
                  └────────────────────────┘
```
Before we start coding, let's take a look at how **ShopMind AI** works behind the scenes.

Every customer request follows a simple but powerful workflow.

When a customer sends a message, it is first passed to **Gemini 3.6 Flash**, which acts as the reasoning engine for our AI agent.

The model analyzes the customer's intent and decides which tool or combination of tools it needs to complete the task.

For example, if the customer wants to check an order, the agent calls the **Order Lookup** tool to retrieve order details and tracking information from the backend.

If the customer asks for a refund, the agent calls the **Refund Calculator** tool to verify business rules and calculate the correct refund amount.

If the customer asks a general question that requires up to date information, the agent can use **Google Search** to retrieve relevant results before generating a response.

Once the appropriate tool finishes its work, the result is sent back to the agent.

The agent then evaluates whether it has enough information to answer the customer. If more information is needed, it selects another tool and continues the reasoning process.

This continuous cycle of **reasoning, tool selection, execution, and evaluation** is known as the **ReAct execution loop**.

The loop continues until the agent has gathered everything it needs to produce an accurate, policy compliant response.

This architecture allows ShopMind AI to handle both simple and multi step customer requests while making intelligent decisions throughout the conversation.

Now that you understand how the system works, let's start building it step by step.


 
### 3. Project Setup

Now, let's set up our project environment. First, create a new project and name it **shopmindAI**. Once you've created it, right-click on the project folder and select **Integrated Terminal** to open a terminal inside the project directory.

Next, let's create a virtual environment. Run the command:

```python
python3 -m venv .venv
```
Here, `python3 -m venv` tells Python to create a virtual environment, and `.venv` is simply the name of that environment. You're free to choose any name you like, but .venv is a common convention and helps keep your projects organized.

Now, activate the virtual environment.

On Linux or macOS, run:
```python
source .venv/bin/activate
```

On Windows, run:
```python
.venv\Scripts\activate
```

Once the virtual environment is activated, we are ready to install the packages for this project.

Open the terminal and run:

```bash
pip install langchain langchain-google-genai langgraph python-dotenv
```

This command installs three Python packages.
* **langchain** provides the framework for connecting our application with large language models and building AI-powered workflows.
* **langchain-google-genai** lets our application communicate with the **Google Gemini** models.
* **langgraph** provides the framework for building the agent workflow and execution loop.
* **python-dotenv** allows us to securely load environment variables, such as our Gemini API key, from a `.env` file instead of hardcoding them into our application.

These packages provide everything we need to start building our autonomous customer support agent.



Now, let's create the files we'll need for our project.

First, create a file named **`requirements.txt`**. This file lists all the Python packages our project depends on, making it easy to recreate the same environment on another machine.

Inside **`requirements.txt`**, list the required libraries:

```text
langchain
langchain-google-genai
langgraph
python-dotenv
```

Later, you can install all the required packages with a single command:

```bash
pip install -r requirements.txt
```

Using a **`requirements.txt`** file is a standard Python practice that makes your projects easier to share, reproduce, and maintain.

Next, create a file named **`.env`**. This file stores your environment variables, such as your Google Gemini API key, so you do not have to hardcode sensitive information in your application.

Inside the **`.env`** file, add the following:

```text
GEMINI_API_KEY=your_google_api_key
```

Next, sign in to **Google AI Studio**, generate an API key, and replace `your_google_api_key` with your own key.

As a best practice, never share your API key or commit your **`.env`** file to a public GitHub repository.



Next, create the three Python files we will use in this project:

```text
├── main.py
├── mock_db.py
└── tools.py
```

Each file has a specific responsibility:

* **`main.py`** is the entry point of our application.
* **`mock_db.py`** contains our mock e commerce database.
* **`tools.py`** contains the tools that our AI agent can use.

Keeping each component in its own file makes the project easier to understand, maintain, and extend as we add more features.

Before we write any code, let's add our **`mock_data.json`** file to the project.

I already have the **`mock_data.json`** file prepared, so I'll simply copy it into the project directory.

This file contains mock customer transaction data for our e commerce store. Each record represents a customer order and includes information such as the customer name, purchased item, order status, price, tracking number, shipping carrier, and the number of days since the order was placed.

We will use this data throughout the tutorial to simulate real customer support scenarios, including checking order status, tracking shipments, and processing refunds.



### 4. Build Our First Tool

Alright, we have finished setting up our project environment. Now it is time to start building our AI agent.

Let's begin with **`mock_db.py`**.

Since we already have our **`mock_data.json`** file, we can load its contents into a Python dictionary instead of hardcoding the data in our application.

First, import Python's built in **`json`** module, which lets us read JSON files. Then, import **`Path`** from the **`pathlib`** library to locate the data file in our project directory.

Next, create a variable named **`DATA_FILE`** that points to **`mock_data.json`**.

Then, open the file in read mode and use **`json.load()`** to convert the JSON data into a Python dictionary. Finally, store the result in the **`ORDERS_DB`** variable.

Keeping the data in a separate JSON file makes the project cleaner, easier to maintain, and simple to extend with a real database or external API in the future.


```python
import json
from pathlib import Path

# Path to the mock data file
DATA_FILE = Path(__file__).parent / "mock_data.json"

# Load the JSON data into a Python dictionary
with open(DATA_FILE, "r", encoding="utf-8") as file:
    ORDERS_DB = json.load(file)
```


Now that we have our order data loaded, we need a way for our AI agent to access this information.

This is where LangChain tools become useful.

Inside **`tools.py`**, we create our first tool using LangChain’s **`@tool` decorator**.

First, we import the **`tool` decorator** from **`langchain_core.tools`**. This allows LangChain to understand that this function is not just a normal Python function, but a tool that our AI agent can call.

Next, we import **`ORDERS_DB`** from our **`mock_db.py`** file. This gives our tool access to the order information we loaded earlier.

Now, we create the **`get_order_status()`** function.

This function receives an **`order_id`** as input and returns the shipment information for that order.

The function docstring is very important:

```python
"""Fetch real-time shipment status and item info for a given order ID."""
```

LangChain uses this description to explain the purpose of the tool to Gemini. Based on this description, Gemini can decide when this tool should be used.

For example, when a user asks:

"Can you check my order status?"

Gemini understands that it needs order information and can trigger the **`get_order_status`** tool.

Inside the function, we search for the order ID in our database:

```python
order = ORDERS_DB.get(order_id.upper())
```

Using **`upper()`** makes the lookup more flexible by allowing users to enter the order ID in different formats, such as `ord123` or `ORD123`.

If the order does not exist, we return a message saying that the order was not found.

Otherwise, we return the item name and current order status.

```python
# tools.py
from langchain_core.tools import tool
from mock_db import ORDERS_DB

@tool
def get_order_status(order_id: str) -> str:
    """Fetch real-time shipment status and item info for a given order ID."""
    order = ORDERS_DB.get(order_id.upper())
    if not order:
        return f"Order {order_id} not found in ShopMind AI database."
    return f"Order {order_id}: {order['item']} - Status: {order['status']}"
```

With this tool, our AI agent can now move beyond generating responses. It can access external data and perform useful tasks based on the user's request.

This is the foundation of building tool-enabled AI agents with LangChain and Gemini.

Next, we create another tool called **`process_return()`**.

This tool allows our AI agent to handle customer return requests by applying our return policy.

The function receives an **`order_id`** and checks the order details from our database.

It first verifies if the order exists. Then, it checks whether the order is within the **30-day return window**.

If the order is older than 30 days, the return request is rejected. Otherwise, the return is approved, and the refund amount is calculated.

The function docstring:

```python
"""Evaluates if an order is within the 30-day return window and calculates refund."""
```

acts as the tool description that helps Gemini understand when it should use this tool.

```python
@tool
def process_return(order_id: str) -> str:
    """Evaluates if an order is within the 30-day return window and calculates refund."""
    order = ORDERS_DB.get(order_id.upper())
    if not order:
        return f"Order {order_id} not found."
    
    if order["days_old"] > 30:
        return f"Return rejected for {order_id}. Order is {order['days_old']} days old (limit: 30 days)."
    
    return f"Return approved for {order_id}. Refund of ${order['price']} initiated."
```

Now our AI agent can not only retrieve information but also apply business rules and make decisions based on real data.


### 5. Constructing the Gemini ReAct Engine

Now, we move to **`main.py`** where we connect everything together and create our AI agent.

First, we import the required libraries.

We import **`load_dotenv`** to load our environment variables, **`ChatGoogleGenerativeAI`** to connect with Gemini, and **`create_agent`** to create our ReAct agent.

We also import HumanMessage, which helps us format user messages before sending them to the agent.

Finally, we import the tools we created earlier: **`get_order_status`** and **`process_return`**.

```python
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from tools import get_order_status, process_return
```

Next, we load our environment variables using:

```python
load_dotenv()
```

This allows our application to securely access values like the Gemini API key from our `.env` file.

Now our application has everything it needs to connect Gemini with our custom tools.

`Next, we create a list of our available tools.

```python id="k5q4ns"
tools = [get_order_status, process_return]
```

This list tells our agent which tools it can use when responding to user requests.

Then, we initialize **Gemini 3.6 Flash** using LangChain’s **`ChatGoogleGenerativeAI`**.

```python id="h4v8k2"
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=None
)
```

The **temperature** value controls how creative or random the model responses are.

The temperature value controls how creative or random the model responses are.

The temperature value controls how creative or random the model responses are.

By setting it to **None**, we allow Gemini to use its default behavior. For an agent that needs to select tools and follow instructions reliably, keeping the model behavior consistent is important

Next, we define the **system prompt** for our AI agent.

```python id="4k9v3h"
system_prompt = """You are ShopMind AI, an autonomous customer support agent for TechStore.
Always verify order details using tools before answering customer questions.
Be concise, accurate, and polite."""
```

This prompt defines the role and behavior of our agent.

It tells Gemini that it is **ShopMind AI**, a customer support assistant, and instructs it to use tools whenever it needs order information.

Finally, we create our agent executor using **`create_agent`**.

```python id="9d2x7m"
agent_executor = create_agent(
    model=llm,
    tools=tools,
    system_prompt=system_prompt
)
```

Here, we connect Gemini with our custom tools and system instructions.

The agent can now understand user requests, decide when a tool is needed, execute the tool, and generate a final response.

Next, we create a helper function called get_clean_text().
```python
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
```

This function handles different response formats from the AI model.

Sometimes the response content is returned as a simple string, and sometimes it may come as a list of content blocks.

This helper converts those formats into clean text that we can display to the user.


### 6. Live Agent Terminal Demonstration

Now, we are ready to test our AI agent in the terminal.

We create a simple loop that allows us to have a conversation with ShopMind AI.

```python id="4p7m2k"
if __name__ == "__main__":
    while True:
        user_input = input("Customer: ")
```

The agent waits for a customer message. When the user types **exit**, we break the loop and stop the application.

```python id="9x2v7d"
if user_input.lower() == "exit":
    break
```

Next, we send the user message to our agent executor.

```python id="j4m8qw"
response = agent_executor.invoke({"messages": [HumanMessage(content=user_input)]})
```

Here, we pass the user's input to the agent using HumanMessage.

The agent receives the request, reasons about the task, decides whether it needs to use one of our tools, and then generates the final response.

This is where the complete agent workflow happens.


Finally, we display the AI response in the terminal.

```python
# Extract and print only the final AI text response
last_msg = response["messages"][-1]

if last_msg.type == "ai" and last_msg.content:
    clean_text = get_clean_text(last_msg.content)
    print(f"\n[ShopMind AI]: {clean_text}\n")
```

Here, we get the last message from the agent response, which contains the final AI answer.

Before displaying it, we use our **`get_clean_text()`** helper function to make sure the response is converted into clean, readable text.

Finally, we print the message with the **ShopMind AI** label.

Now we have a fully working AI customer support agent that can understand user requests, use tools, apply business rules, and respond intelligently to customers.



Now, let's run our agent and see it in action.

We start the application:

```bash
python main.py
```

First, we ask about an order status:

```text
Customer: Can you check where my order ORD-10002 is?
```

ShopMind AI understands that this request requires order information, so it automatically uses the **`get_order_status`** tool.

The agent returns:

```text
[ShopMind AI]: Your order ORD-10002 for Wireless Headphones is currently Shipped.
```

Next, we test a return request:

```text
Customer: I want to return ORD-10008. Can I get a refund?
```

This time, the agent recognizes that it needs to check the return policy, so it uses the **`process_return`** tool.

The response:

```text
[ShopMind AI]: I checked order ORD-10008. Unfortunately, this order was placed 40 days ago, which exceeds our 30-day return policy limit.
```

Notice how the agent automatically selects the right tool based on the user's request.

It uses **`get_order_status`** for tracking questions and **`process_return`** for return validation.

This is the power of AI agents. They do not just generate text. They can reason about a task, use external tools, and apply business rules automatically.


### 7. Summary & Conclusion

`And that is how we built **ShopMind AI**, a complete e-commerce support agent powered by **Gemini 3.6 Flash, LangChain, and LangGraph**.

In this project, we connected Gemini with real tools, added clear tool descriptions, and created a ReAct workflow that allows the agent to understand requests, choose the right tool, and complete tasks automatically.

Instead of building a simple chatbot that only generates text, we created an AI agent that can access data, apply business rules, and take useful actions.

If you enjoyed building ShopMind AI, make sure to like this video, subscribe to the channel, and turn on notifications for more AI development tutorials.

Let me know in the comments what features you would like to add to ShopMind AI next.

Thanks for watching, and happy coding!
