from typing import TypedDict, Annotated
from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langgraph.prebuilt import ToolNode, tools_condition
import random 
import requests

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    raise RuntimeError(
        "GROQ_API_KEY not set. Create a .env with GROQ_API_KEY=... or set the environment variable."
    )
os.environ["GROQ_API_KEY"] = groq_key
model = init_chat_model("groq:llama-3.1-8b-instant")


# 1. Define a chat state 

class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages] # defines an annotated list of base messages

# 2. Define the nodes 

def chat_node(state:ChatState):

    decision = interrupt({
        "type":"approval",
        "reason":"Model is about to answer a user question",
        "question":state["messages"][-1].content,
        "instruction":"Approve this question ? Yes/No"
    })

    if decision["approved"]=='n':
        return {"messages":[AIMessage(content="Not approved")]}
    
    else:
        response = model.invoke(state["messages"])
        return {"messages":[response]}
    

# 3. Define the graph structure 

builder = StateGraph(ChatState)

builder.add_node("chat", chat_node)
builder.add_edge(START,"chat")
builder.add_edge("chat", END)

checkpointer = MemorySaver()

app = builder.compile(checkpointer=checkpointer)

app

# create a new thread for this conversation 

config = {"configurable":{"thread_id":'1234'}}

# step 1. user asks a question 
initial_input = {
    "messages":[
        ("user","Explain gradient descent in very simple terms.")
    ]
}


# invoke the graph for the first time 
result = app.invoke(initial_input, config=config)
print(result)

message = result['__interrupt__'][0].value
print(message)

user_input = input(f"\n Backend message - {message} \n Approve this question? (y/n):")


# resume the graph with approval decision 
final_result = app.invoke(
    Command(resume={"approved":user_input}),
    config = config,
)

print(final_result["messages"][-1].content)