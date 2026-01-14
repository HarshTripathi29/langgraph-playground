from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    raise RuntimeError(
        "GROQ_API_KEY not set. Create a .env with GROQ_API_KEY=... or set the environment variable."
    )
os.environ["GROQ_API_KEY"] = groq_key
model = init_chat_model("groq:llama-3.1-8b-instant")

# 1. Define the state
class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]

# 2. Define the node, llm init, functions
def chat_node(state:ChatState)->ChatState:
    # read messages and validate
    messages = state.get('messages')

    # Validate that 'messages' is a non-empty list.
    # The model API requires at least one message; raising early gives a clear error instead of a remote 400.
    if not isinstance(messages, list) or len(messages) == 0:
        raise ValueError("state['messages'] must be a non-empty list of BaseMessage (e.g., HumanMessage)")

    # ensure elements are BaseMessage
    # Validate that each item is a BaseMessage (HumanMessage/AIMessage/etc.).
    # This prevents serialization/type errors when the LLM client processes the message list.
    for m in messages:
        if not isinstance(m, BaseMessage):
            raise ValueError("All items in state['messages'] must be instances of BaseMessage (e.g., HumanMessage)")

    # send to the LLM and append reply to history
    response = model.invoke(messages)
    return {'messages': messages + [response]}


# 3. Define the graph
graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node',END)

chatbot = graph.compile()

# 4. Execute the graph
initial_state = {
    'messages':[HumanMessage(content='What is the capital of India')]
}

final_state = chatbot.invoke(initial_state)
print(final_state['messages'][-1].content)

# Goal

# Implement a simple, single-node chatbot using **LangGraph** that:

# - Maintains a running **message history** in state,
# - Sends the history to an **LLM** (e.g., `ChatOpenAI`),
# - Appends the **LLM’s reply** back into the state,
# - And returns the bot’s final response.

# You’ll model state using a `TypedDict` with `messages` and wire the flow using `StateGraph(START → chat_node → END)`.
# ## ✅ Learning Objectives

# 1. Define a LangGraph **state schema** with `TypedDict` and `Annotated` + `add_messages`.
# 2. Implement a node function that:
#     - Reads messages from state,
#     - Invokes an LLM with the message history,
#     - Returns the model’s reply appended as a new message.
# 3. Build and compile a **graph** with `START`, a single `chat_node`, and `END`.
# 4. Invoke the compiled graph with an initial `HumanMessage` and read the bot’s reply.

