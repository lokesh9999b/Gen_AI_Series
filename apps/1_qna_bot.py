from dotenv import load_dotenv
load_dotenv()
import streamlit as st

from langchain_groq import ChatGroq

llm = ChatGroq(model="openai/gpt-oss-120b")
st.title("QnA Bot")
st.markdown("This is a simple QnA bot using LangChain and Groq. Type your question below and press Enter to get an answer.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)

query = st.chat_input("Ask a question:")
history = st.session_state.messages
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").markdown(query)
    res = llm.invoke(st.session_state.messages)
    st.chat_message("assistant").markdown(res.content)
    st.session_state.messages.append({"role": "assistant", "content": res.content})
