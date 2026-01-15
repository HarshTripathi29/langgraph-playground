from typing import TypedDict
from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    raise RuntimeError(
        "GROQ_API_KEY not set. Create a .env with GROQ_API_KEY=... or set the environment variable."
    )
os.environ["GROQ_API_KEY"] = groq_key
model = init_chat_model("groq:llama-3.1-8b-instant")

# 1. Define the state
class JokeState(TypedDict):
    joke: str
    topic: str
    explanation: str 


# 2. Define the nodes 
def generate_jokes(state: JokeState) -> JokeState:
    # extract the topic from the state 
    topic = state['topic']

    # give the topic to the llm and generate a joke
    prompt = f'generate a joke for this topic : {topic} in 30-40 words'
    generated_joke = model.invoke(prompt)
    state['joke'] = generated_joke.content

    return state

def generate_explanation(state: JokeState) -> JokeState:
    # get the joke from the state
    joke = state['joke']
    
    # give the joke to the llm and generate explanation of the joke
    prompt = f'create a short explanation of the joke : {joke}'
    generated_explanation = model.invoke(prompt)
    state['explanation'] = generated_explanation.content

    return state


# 3. Define the graph, define a checkpointer before compiling 
graph = StateGraph(JokeState)

graph.add_node('generate_jokes', generate_jokes)
graph.add_node('generate_explanation', generate_explanation)

graph.add_edge(START, 'generate_jokes')
graph.add_edge('generate_jokes', 'generate_explanation')
graph.add_edge('generate_explanation', END)

checkpointer = InMemorySaver()

chatbot = graph.compile(checkpointer=checkpointer)

# 4. Execute the graph with a config dict having the thread_id
config1 = {"configurable": {"thread_id": "1"}}

initial_state = {'topic': 'burger', 'joke': '', 'explanation': ''}

final_state = chatbot.invoke(initial_state, config=config1)

print(final_state)

# get the state history
chatbot.get_state(config1)

#get a particular checkpoint