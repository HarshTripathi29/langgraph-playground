from typing import TypedDict, Annotated
from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
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

# 1. define the tools 

# search tool : inbuilt langchain tool 
search_tool = DuckDuckGoSearchRun(region="us-en")

# user defined tool and make a list of tools and the bind the llm model with the tools
@tool
def calculator(first_num:float, second_num:float, operation:str)->dict:
    """
    Instructions for calculator
    perform a basic arithmetic operation on 2 numbers.
    Supported operations : addition, subtraction, multiplication, diviion 
    """
    try:
        if operation=="addition":
            result=first_num+second_num
        elif operation=="subtraction":
            result=first_num-second_num
        elif operation=="multiplication":
            result=first_num*second_num
        elif operation=="division":
            if second_num==0:
                return {"error":"Division by 0 is not allowed"}
            result=first_num/second_num
        else:
            return {"error":f"Unsupported operation '{operation}' "}
        
        return {"first_num":first_num, "second_num":second_num, "operation":operation, "result":result}
    
    except Exception as e:
        return {"error":str(e)}
    
@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    r = requests.get(url)
    return r.json()


# make the list of tools 
tools = [get_stock_price, search_tool, calculator]

# make the model tool aware 
model_with_tools = model.bind_tools(tools)


# 2. Define the state 
class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage], add_messages]

# 3. Define the graph nodes
def chat_node(state:ChatState)->ChatState:
    """LLM node that may answer or request a tool call"""
    messages = state['messages']
    response = model_with_tools.invoke(messages)
    return {"messages":[response]}

tool_node = ToolNode(tools)

# 4. Define the graph structure
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tool_node", tool_node)

graph.add_edge(START, "chat_node")
# if the llm asked for a tool then go for tool otherwise finish 
graph.add_conditional_edges("chat_node", tools_condition, {"tools": "tool_node", "__end__": END})
graph.add_edge("tool_node", "chat_node")

chatbot = graph.compile()

# 5. Execute the graph
out = chatbot.invoke({"messages":[HumanMessage(content="What is 2+2")]})
print(out["messages"][-1].content)

out = chatbot.invoke({"messages":[HumanMessage(content="What is the stock price of apple")]})
print(out["messages"][-1].content)

out = chatbot.invoke({"messages":[HumanMessage(content="First find out the stock price of Apple using get stock price tool then use the calculator tool to find out how much will it take to purchase 50 shares?")]})
print(out["messages"][-1].content)