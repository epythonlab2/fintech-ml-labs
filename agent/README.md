## Create Your First AI Agent Without LangChain

### 1. Introduction

One of the biggest misconceptions in AI today is that you need massive frameworks like LangChain or CrewAI to build an AI agent.

In reality, you can build using pure Python.

In this video, I will show you how to build a fully functional AI agent from scratch using pure Python. We're not going to rely on **LangChain, CrewAI**, or any other heavy frameworks. Instead, we'll build it step by step so you can understand exactly what's happening at each stage. By the end, you'll have a working AI agent and a solid understanding of how modern AI agents make decisions and use tools.

### 2. What is an AI Agent?
```bash
              ┌────────────────────────┐
              │     OBSERVE STATE      │
              └───────────┬────────────┘
                          ▼
              ┌────────────────────────┐
              │      REASON & THINK    │
              └───────────┬────────────┘
                          ▼
              ┌────────────────────────┐
              │     SELECT ACTION      │
              └───────────┬────────────┘
                          ▼
              ┌────────────────────────┐
              │    EXECUTE / RETURN    │
              └───────────┬────────────┘
                          │
                          └─── (REPEAT LOOP)
```
To design effective systems, we first have to demystify them.

Most engineers imagine AI agents as mysterious digital workers. In reality, every single agent architecture executes four distinct, sequential steps:

1. **Observe:** Capture input from the user or the environment.

2. **Reason:** Evaluate that input against context and objectives.

3. **Select Action:** Determine whether to respond directly or invoke external capabilities.

4. **Execute:** Trigger the corresponding function and ingest the result back into memory.

This cycle is known as the **Agent Execution Loop**.

While traditional large language models are limited to stateless text generation, an agent acts as an operator. It bridges the model with external infrastructure:- **querying databases, executing Python code, calling APIs, and managing local file systems**.

### 3. Agent Architecture
 ```bash
                  ┌──────────────────────┐
                  │     User Request     │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │    Reasoning Engine  │
                  └──────────┬───────────┘
                             │
               /────── Is a tool required? ──────\
              /                                   \
             YES                                  NO
              │                                    │
              ▼                                    ▼
   ┌────────────────────┐                ┌───────────────────┐
   │ Execute Capability │                │  Direct Response  │
   └──────────┬─────────┘                └─────────┬─────────┘
              │                                    │
              └───────────────► ┌──────────────────┘
                                │
                                ▼
                     ┌────────────────────┐
                     │   Return State     │
                     └────────────────────┘
```
Let's break down the control flow.

When a user initiates a request, it enters the Reasoning Engine. The engine evaluates the request and asks a fundamental routing question: Does fulfilling this objective require an external capability?

If **Yes**, control branches out to an external execution module—whether that’s a REST API, a shell script, or a mathematical solver. The result is captured and fed back into the reasoning context.

If **No**, the engine immediately generates a direct response.

That's the entire core architecture. Everything else—memory, planning, multi-agent orchestration—is built right on top of this decision tree.

### 4. Project Setup

Now, let's set up our project environment. First, create a new project and name it **agent**. Once you've created it, right-click on the project folder and select **Integrated Terminal** to open a terminal inside the project directory.

Next, let's create a virtual environment. Run the following command:

```python
python3 -m venv .venv
```
Here, `python3 -m venv` tells Python to create a virtual environment, and `.venv` is simply the name of that environment. You're free to choose any name you like, but .venv is a common convention and helps keep your projects organized.

Now that w've created our virtual environment, the next step is to install the packages we'll need for this project. Open your terminal and run the following command:
```python
pip install openai python-dotenv
```

This installs two libraries. The **OpenAI** package allows our Python application to communicate with OpenAI models, and **python-dotenv** lets us securely load environment variables, such as our API key, from a **.env** file instead of hardcoding them into our code. This is a simple but important best practice for keeping sensitive information secure.

Now, let's create the files we'll need for our project.

Inside the **agent** folder, create a new file named `.env`. This file is where we'll store our OpenAI API key securely.

Open the `.env` file and add the following line:
```python
OPENAI_API_KEY=your_actual_api_key_here
```

Next, sign in to your OpenAI account, create or copy your API key from the API dashboard, and replace your_actual_api_key_here with your own key.

**A quick security tip:** Never share your API key or commit your `.env` file to GitHub. Treat it like a password, because anyone with access to it can use your OpenAI account and incur charges.

Next, create another file named `requirements.txt`. This file keeps track of all the Python packages that our project depends on. It's especially useful when you want to share your project or set it up on another computer, because you can install all the required packages with a single command.

Inside the requirements.txt file, add the following:
```python
openai
python-dotenv
```

Later, anyone can install these dependencies by running:
```python
pip install -r requirements.txt
```

Using a `requirements.txt` file is a standard Python practice and helps make your projects easier to reproduce and maintain.

Finally, create the three Python files we need to use in this project:
```python
main.py
agent.py
tools.py
```

`main.py` is the entry point of our application. `agent.py` contains the AI agent's logic, and `tools.py` contains the tools the agent can use. This simple structure keeps our project organized and easy to maintain.

### 5. Build Our First Tool

Alright, we've finished setting up our project environment. Now it's time to start building our AI agent.

First, open the `tools.py` file and import Python's built-in `datetime` module. We'll use it to create our first tool, which returns the current date and time.
```python
from datetime import datetime
```
Next, let's define the `get_current_time()` function.

Notice how simple it is. A tool doesn't have to be anything complicated. In its simplest form, a tool is just a Python function that performs a specific task and returns the result. In this case, our tool returns the current date and time as a string.

```python
def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
```

Now, let's create our second tool: a simple calculator.

This function accepts a mathematical expression as text, evaluates it, and returns the result. If the expression isn't valid, it catches the error and returns a friendly error message instead.

Before we continue, there's one important thing to know. In real-world applications, you should never use Python's `eval()` function with user input because it can execute malicious code. We're only using it here to keep the example simple and focus on understanding how AI agents work. Later, you can replace it with a safer expression parser for production use.

```python
def calculator(expression):
    try:
        return str(eval(expression))
    except Exception:
        return "Error: Invalid mathematical expression"
```

### 6. Build the Decision Logic

Now let's move on to the core of our AI agent: the decision logic.

Open the `agent.py` file. This is where we'll define how our agent receives a request, analyzes it, and decides which tool to use.

We begin by importing the tool functions from our `tools.py` file.

These are the capabilities our agent can use when it needs to perform specific tasks. In this case, we're giving our agent access to the time tool and the calculator tool.
```python
from tools import get_current_time, calculator
```

Next, we create our `SimpleAgent` class.

Inside this class, we define the `think()` method, which is the main decision-making function of our agent.

First, we normalize the user's input by converting it to lowercase. This makes the agent's routing logic more consistent, so it can recognize requests regardless of how the user types them.
```python
class SimpleAgent:
    def think(self, prompt: str) -> str:
        normalized_prompt = prompt.lower()
```

Now we build the first part of our decision logic.

If the user's prompt contains the word `"time"`, the agent recognizes that it needs the time tool and calls the `get_current_time()` function.

This is the basic idea behind tool selection in an AI agent: the agent receives a request, decides what capability it needs, and then executes the appropriate tool.


