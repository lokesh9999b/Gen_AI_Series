from dotenv import load_dotenv
load_dotenv()

from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver 
import streamlit as st


if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()
    st.session_state.history = []



search = GoogleSerperAPIWrapper()
model = ChatGroq(model="openai/gpt-oss-20b", temperature= 0.5, streaming=True)
tools = [search.run]
memory = st.session_state.memory

agent = create_agent(
                model = model, 
                tools = tools, 
                system_prompt="you are a google search agent you have to use the tool for searching", 
                checkpointer=memory
                    )
st.title("Google Search Agent")

for message in st.session_state.history:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)

query = st.chat_input("Ask Anything")

if query:
    st.chat_message("user").markdown(query)
    st.session_state.history.append({"role": "user", "content": query})

    response = agent.stream(
                {"messages": [{"role": "user", "content": query}]}, 
                {"configurable":{"thread_id":"Lokesh"}},
                stream_mode="messages"
                       )
    
    ai_container = st.chat_message("ai")
    with ai_container:
        space = st.empty()

        message =""
        for chunk in response:
            message = message + chunk[0].content
            space.write(message)
        st.session_state.history.append({"role":"AI","content": message})



    # 



    