from dotenv import load_dotenv
load_dotenv()

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver 

search = GoogleSerperAPIWrapper()
model = ChatGroq(model="openai/gpt-oss-20b", streaming=True)

agent = create_agent(model, tools=[search.run], system_prompt="you are a google search agent you have to use the tool for searching", checkpointer=InMemorySaver())

while True:
    query = input("User:")
    if query == "quiet" or query == "exit":
        break
    response = agent.invoke({"messages": [{"role": "user", "content": query}]}, {"configurable":{"thread_id":"Lokesh"}})
    print("AI:", response["messages"][-1].content)