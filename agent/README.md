## Create Your First AI Agent Without LangChain

### 1. Introduction

One of the biggest misconceptions in AI today is that you need massive frameworks like LangChain or CrewAI to build an AI agent.

In reality, you can build one using pure Python.

In this video, I'll show you how to build a fully functional AI agent from scratch using pure Python. We're not going to rely on LangChain, CrewAI, or any other heavy frameworks. We'll also use Ollama to run a local language model, so you don't need an API key or paid credits to follow along.

We'll build everything step by step, so you can understand exactly what's happening at each stage. By the end of this tutorial, you'll have a working AI agent and a solid understanding of how modern AI agents make decisions and use tools.

Before we begin, please like this video, subscribe to the channel, and leave a comment below. Your support really helps. Let's get started!

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

Once the virtual environment is activated, we're ready to install the packages for this project.

In the terminal, run:
```python
pip install ollama
```

This installs the Ollama Python library, which allows our application to communicate with a local large language model running on our machine.


Now, let's create the files we'll need for our project.

First, create a file named `requirements.txt`. This file keeps track of all the Python packages that our project depends on. It's especially useful when you want to share your project or set it up on another computer, because you can install all the required packages with a single command.

Inside the requirements.txt file, add the following:
```python
ollama
```

Later, anyone can install these dependencies by running:
```python
pip install -r requirements.txt
```
Using a `requirements.txt` file is a standard Python practice and helps make your projects easier to reproduce and maintain.

And then, create the three Python files we need to use in this project:
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

```python
if "time" in normalized_prompt:
    return get_current_time()
```


Next, we handle calculation requests.

If the user's prompt contains the word "calculate", the agent extracts the mathematical expression from the input and sends it to our calculator tool.

The calculator then processes the expression and returns the result back to the agent.

```python
elif "calculate" in normalized_prompt:
    expression = normalized_prompt.replace("calculate", "").strip()
    return calculator(expression)
```
If no tool signatures are detected, the router falls back to standard execution.

```python
else:
    return "No external tools required. Reverting to direct response."
```
Pause and inspect this. Without importing a single external framework, we have designed a functioning agent core. It ingests state, evaluates criteria, routes execution, and yields output.


### 7. Run the Agent

Now let's move to `main.py`.

Here, we import our SimpleAgent class and create an instance of the agent. This gives us an object we can use to send requests and interact with our AI agent.

```python
from agent import SimpleAgent

agent = SimpleAgent()
```

Next, we create a simple execution loop that allows us to interact with our agent from the terminal.

The loop keeps asking for user input and continues running until the user types "exit". This gives us a simple way to test our agent in real time.

```python
while True:
    user_input = input("User: ")

    if user_input.strip().lower() == "exit":
        break
```

Finally, inside the loop, we send the user's input to the agent using the `think()` method.

The agent processes the request, decides what action to take, and returns a response. We then display that response in the terminal.

```python
response = agent.think(user_input)
print(f"Agent: {response}\n")
```

Now let's run our agent.

Open the terminal and execute the main.py file:
```python
python3 main.py
```
Once the application starts, we can interact with our agent directly from the terminal. Let's test it by sending different requests and see how the agent decides which tool to use.

First, we ask a time-related question. The agent recognizes the keyword "time", selects the correct tool, and returns the current time.

Next, we provide a calculation request. The agent extracts the expression, sends it to the calculator tool, and returns the result.

Finally, we send a request that doesn't require any tool. Since the agent doesn't have a matching tool for this type of request yet, it falls back to a direct response.

This simple example demonstrates the core agent workflow: receive input, make a decision, choose an action, and return a result.

```python
$ python main.py

User: what time is it right now
Agent: 2026-07-22 13:45:10

User: calculate 45 * 12
Agent: 540

User: tell me a joke
Agent: No external tools required. Reverting to direct response.
```

Notice how our simple decision logic cleanly routes each request to the right place.

When the agent recognizes a specific task, it calls the appropriate tool. When no tool is needed, it simply returns a direct response.

This is the basic foundation of an AI agent: understanding the request, deciding what action to take, and executing that action.

### 8. Add an LLM Brain

Now that we've built the basic agent loop, let's make our agent smarter by adding a Large Language Model.

So far, our agent relies on simple if statements to decide what to do. That works for straightforward commands, but real users don't always ask questions the same way.

For example, someone might say, "What time is it?", "Can you tell me the current time?", or "What's the time right now?" They all mean the same thing, but a rule-based system may not recognize every variation.

This is where an LLM becomes useful. Instead of relying only on hardcoded rules, it can understand the user's intent and generate a natural response.

Now let's integrate **Ollama** into our agent.

First, import the ollama library. Then, inside the think() method, remove the final else block. If none of our tools match the user's request, we'll send the prompt to the local language model instead.
```python
import ollama

# Fall back to the LLM
        response = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response["message"]["content"]
```

Now our agent follows a simple workflow. It first checks whether a request can be handled by one of its tools. If not, it forwards the request to the LLM for reasoning and response generation.

This approach gives us the best of both worlds: deterministic tools for specific tasks and a language model for flexible, natural conversations.

Let's run the application and see our AI agent in action.

Before running our AI agent, we need to make sure Ollama is ready.

First, start the Ollama server by running:
```python
ollama serve
```

This starts the local server that allows our Python application to communicate with the language model.

Next, download the model we'll use in this project. You only need to do this once:
```python
ollama pull llama3.2
```

This downloads the Llama 3.2 model and stores it locally on your machine.

Finally, run the agent application and try different types of questions. This time, instead of matching specific keywords, the model will interpret the meaning of the request and generate a response based on the context.

For example:
```python
$ python main.py

User: Explain what an AI agent is
Agent: An AI agent is a system that can perceive information, make decisions, and take actions to achieve a goal...

User: What is the difference between machine learning and deep learning?
Agent: Machine learning is a broader field where systems learn from data, while deep learning uses neural networks with multiple layers...

User: How does tool calling work in AI agents?
Agent: Tool calling allows an AI model to interact with external functions and systems to complete tasks beyond generating text.
```
Notice the difference. Our previous version relied on predefined rules, but now the agent can understand a wide range of natural language requests.

This is the foundation of modern AI agents: combining an LLM's reasoning ability with external tools that allow the system to take action.

### 9. Understanding the Agent Loop

Step back and examine the full picture. This exact lifecycle is what frameworks like AutoGen, LangChain, or CrewAI execute behind their abstraction layers.

They automate prompt engineering, schema definitions, and state serialization. But when you build this loop yourself from scratch, you eliminate mystery. You gain complete visibility over token management, routing logic, and execution safety.

```bash
┌─────────────────────────────────────────────────────────┐
│                    THE AGENT LIFECYCLE                  │
├─────────────────────────────────────────────────────────┤
│  1. Capture Input State                                 │
│  2. Evaluate Intent via Reasoning Engine (LLM)          │
│  3. Branch Decision: Tool Call required?                │
│     ├── YES ──► Invoke Function ──► Append to Context   │
│     └── NO  ──► Generate Final Output                   │
│  4. Yield Execution Control Back to System              │
└─────────────────────────────────────────────────────────┘
```
### 10. How LangChain Fits In

At this point, you might be wondering: Where do frameworks like LangChain and CrewAI fit in?

The answer is simple. Frameworks don't replace the fundamentals—they build on top of them.

Everything we've built from scratch has an equivalent abstraction in a framework. The main difference is that the framework automates much of the boilerplate code.

Here's a quick comparison:
```bash
| Core Concept     | Raw Python Implementation                 | Framework Abstraction                    |
| ---------------- | ----------------------------------------- | ---------------------------------------- |
| Control Loop     | `while` loop                              | Agent Executor                           |
| Tool Interface   | Python functions (`def`)                  | Tool schemas or decorators               |
| Model Runtime    | Direct OpenAI API calls                   | Provider-agnostic model wrappers         |
| State Management | Python variables, lists, and dictionaries | Conversation memory and state management |
```
Frameworks become valuable as your applications grow. They offer built-in integrations for vector databases, memory, multiple model providers, tool management, and workflow orchestration.

But there's an important lesson here. If you rely on a framework before understanding how an AI agent actually works, debugging becomes much harder because many of the details are hidden behind abstractions.

That's why I recommend learning the fundamentals first. Once you understand the core architecture, frameworks become productivity tools instead of black boxes. You'll know exactly what they're doing behind the scenes and when they're worth using.

### 11. Next Improvements

Now that you hold the core blueprint, here is your roadmap for scaling this architecture:

- First: Refactor tool execution to use native JSON schemas for automated argument generation.

- Second: Add persistent conversation memory to preserve multi-turn state.

- Third: Implement task-decomposition patterns to enable multi-step autonomous planning.

```bash
   [Phase 1] Pure Python Agent Core (Done)
      │
      ▼
   [Phase 2] Native Tool Calling & JSON Schemas
      │
      ▼
   [Phase 3] Conversational Memory & Vector Store State
      │
      ▼
   [Phase 4] Multi-Step Planning & Self-Correction Loops
```
### 12. Conclusion

AI agents aren't magic. They're built on a simple loop: observe, reason, decide, and act.
