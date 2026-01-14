from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import START,END,StateGraph
from langchain.chat_models import init_chat_model
import os

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
model = init_chat_model("groq:llama-3.1-8b-instant")

# 1. Define the state

class BlogState(TypedDict):
    title:str
    outline:str
    blog:str

# 2. Define the nodes : functions and the llm inits 

def create_outline(state:BlogState)->BlogState:

    # fetch title from the state 
    title = state['title']

    # call llm gen outline
    outline = model.invoke(f'create a blog outlinr for the given title {title} in 20 words')

    # update state
    state['outline'] = outline

    return state

def create_blog(state:BlogState)->BlogState:

    # get the outline from the state
    outline = state['outline']

    # create a blog following the outline 
    blog = model.invoke(f'create a blog following the given outline {outline} in 200 words')

    # update the state
    state['blog'] = blog

    return state 


# 3. Define the graph 

graph = StateGraph(BlogState)

graph.add_node('create_outline', create_outline)
graph.add_node('create_blog', create_blog)

graph.add_edge(START, 'create_outline')
graph.add_edge('create_outline', 'create_blog')
graph.add_edge('create_blog', END)

workflow = graph.compile()


# 4. Execute the graph

initial_state = {'title':'Rise of AI in IT Sector'}
final_state = workflow.invoke(initial_state)
print(final_state)


