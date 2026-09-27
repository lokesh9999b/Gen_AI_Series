# GENAI-Series — Learn Generative AI with LangChain, Groq, OpenAI, Ollama

A hands-on learning series. Each notebook teaches one idea, each app shows the same idea in a runnable program. Written so a beginner can follow in order, and anyone else can jump to a single file as a reference.

If you are new: read this file top to bottom, then run the notebooks in the order given in [Learning path](#3-learning-path-follow-this-order).

## Table of contents

1. [What you will learn](#1-what-you-will-learn)
2. [Big picture: GenAI, LLMs, LangChain](#2-big-picture-genai-llms-langchain)
3. [Learning path](#3-learning-path-follow-this-order)
4. [Setup](#4-setup)
5. [Project layout](#5-project-layout)
6. [Notebooks explained](#6-notebooks-explained)
7. [Apps explained](#7-apps-explained)
8. [Core patterns with examples](#8-core-patterns-with-examples)
9. [Models used](#9-models-used)
10. [Troubleshooting](#10-troubleshooting-errors-i-hit-and-fixes)
11. [Git and GitHub](#11-git-and-github)
12. [Security](#12-security)
13. [Next steps](#13-next-steps)

## 1. What you will learn

- Call chat models from OpenAI, Google, Anthropic, Groq, and local Ollama.
- Difference between `llm.invoke("text")`, message lists, static prompts, and `ChatPromptTemplate`.
- LCEL chains: `prompt | llm | parser | function`.
- `StrOutputParser` and why `print(res)` vs `print(res.content)` matters.
- Structured output with Pydantic (`with_structured_output`).
- Custom tools with `@tool` and math agents with `create_agent`.
- Web-search agents with Serper + Groq.
- Local models with Ollama.
- Streamlit chat app with history.
- Common errors: wrong model name 404, missing `SERPER_API_KEY`, `langchain-community` deprecation, empty output.

## 2. Big picture: GenAI, LLMs, LangChain

**Generative AI** creates new content (text, code) instead of only classifying. **LLMs** (GPT, Claude, Gemini, Llama-family, gpt-oss) are transformer models trained on large text corpora that predict the next token.

**LangChain** is a helper layer:

- `ChatGroq`, `ChatOpenAI`, `ChatAnthropic`, `ChatGoogleGenerativeAI`, `ChatOllama` — same `.invoke()` interface, different backends.
- Messages: `{"role": "system"|"user"|"assistant", "content": "..."}`.
- Prompts: templates with variables like `{language}`, `{query}`.
- Output parsers: convert `AIMessage` to `str` / JSON / Pydantic object.
- LCEL (`|`): pipe steps into a chain.
- Tools + agents: model decides when to call your Python function or search API.
- LangGraph `create_agent`: runs the think → act → observe loop.

Flow is always: `input dict → prompt → messages → model → AIMessage → parser → Python value`.

## 3. Learning path: follow this order

1. `notebooks/dynamic.ipynb` — simplest Groq call + translator LCEL chain. Start here.
2. `notebooks/1_basic_langchain_wih_openai.ipynb` — same `.invoke()` across OpenAI / Google / Anthropic / Groq.
3. `notebooks/3_structured_output.ipynb` — force JSON-shaped answers with Pydantic.
4. `notebooks/4_ollama.ipynb` — same code, local model, no API key.
5. `notebooks/5_basic_agents.ipynb` — `@tool` + `create_agent` for math.
6. `notebooks/6_googleSearch_agent.ipynb` — search agent with Serper.
7. `apps/1_qna_bot.py` — Streamlit chat memory.
8. `apps/2_googleSearch_agent.py` — Streamlit search agent with streaming + conversation memory.

## 4. Setup

Requirements: Python 3.11, Git, Jupyter. Ollama only for notebook 4.

```powershell
cd GENAI-Series
python -m venv env
.\env\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` installs: `langchain`, `langchain-openai`, `langchain-anthropic`, `langchain-google-genai`, `langchain-groq`, `langchain-ollama`, `langchain-community`, `langchain-diffbot`, `langgraph`, `streamlit`, `dotenv` (use `python-dotenv` if `dotenv` fails).

Env keys — copy and fill only what you use:

```powershell
Copy-Item .env.example apps\.env
Copy-Item .env.example notebooks\.env
```

`.env.example`:

```text
OPENAI_API_KEY=
GOOGLE_API_KEY=
ANTROPIC_API_KEY=
GROQ_API_KEY=
HUGGINGFACEHUB_API_KEY=
SERPER_API_KEY=
```

Run:

```powershell
jupyter notebook notebooks
streamlit run apps\1_qna_bot.py
streamlit run apps\2_googleSearch_agent.py
```

For Ollama:

```powershell
ollama pull gemma3
ollama serve
```

## 5. Project layout

```text
GENAI-Series/
├── apps/
│   ├── 1_qna_bot.py            # Streamlit QnA, ChatGroq openai/gpt-oss-120b, chat history
│   └── 2_googleSearch_agent.py # Streamlit Serper + Groq agent, MemorySaver, token streaming
├── notebooks/
│   ├── dynamic.ipynb                       # Groq invoke + ChatPromptTemplate + LCEL translator
│   ├── 1_basic_langchain_wih_openai.ipynb  # OpenAI/Groq/Google/Anthropic invoke, static prompts
│   ├── 3_structured_output.ipynb           # Pydantic structured output, openai/gpt-oss-120b
│   ├── 4_ollama.ipynb                      # Local ChatOllama gemma3
│   ├── 5_basic_agents.ipynb                # @tool math functions + create_agent
│   └── 6_googleSearch_agent.ipynb          # GoogleSerperAPIWrapper + Groq search agent
├── requirements.txt
├── .env.example
├── .gitignore   # ignores env/, .env, __pycache__, checkpoints
└── .gitattributes
```

`env/` is the local virtualenv and is never committed. `apps/.env` and `notebooks/.env` hold real keys and are never committed.

## 6. Notebooks explained

### `dynamic.ipynb` — prompts and LCEL

- `load_dotenv()` loads keys.
- `ChatGroq(model="openai/gpt-oss-20b")`.
- Direct call: `llm.invoke([{"role":"system",...},{"role":"user",...}])` returns `AIMessage`, print with `res.content`.
- Templated call: `ChatPromptTemplate.from_messages([system with {language}, user with {query}])`.
- Chain: `chains = prompts | llm | StrOutputParser() | transform_case`, then `chains.invoke({"language":"Telugu","query":"..."})` returns plain `str`, print with `print(res)`.
- Key lesson: after a parser the result has no `.content`.

### `1_basic_langchain_wih_openai.ipynb` — one interface, many vendors

- `ChatOpenAI(model="gpt-4o")`, `ChatGoogleGenerativeAI`, `ChatAnthropic(model="claude-3.0")`, `ChatGroq(model="openai/gpt-oss-20b")` all share `.invoke()`.
- Static prompt form: `[("system","You are a python fullstack developer."),("user","write the code...")]`.
- Shows raw `AIMessage` vs parsed text.

### `3_structured_output.ipynb` — JSON you can code against

- Model: `ChatGroq(model="openai/gpt-oss-120b")`.
- `class ResponseStructure(BaseModel): name: str; age: int; email: str` + `llm.with_structured_output(ResponseStructure, method="json_schema")`.
- `class MovieList(BaseModel): movies: list[Movies]` for lists.
- Call returns Pydantic object, use `.model_dump()`.
- Lesson: use when you need fields, not prose.

### `4_ollama.ipynb` — local inference

- `ChatOllama(model="gemma3")`, `llm.invoke("give me the latest 3 movies...")`.
- No cloud key needed, but Ollama must be running and model pulled. Quality/speed depend on your machine.

### `5_basic_agents.ipynb` — tools and agents

- `@tool def add_numbers(a:int,b:int)`, plus `mul_numbers`, `substarction`, `division`, `reminder`. Docstring becomes the tool description the model reads.
- `create_agent(llm, tools=[...], system_prompt="you are a math teacher every time use tools...")`.
- `agent.invoke({"messages":[{"role":"user","content":"what is division of 100000000 with 5..."}]})`, answer in `response["messages"][-1].content`.
- Example output in notebook renders LaTeX fractions. In plain `print()` that looks like `\[ \frac{...}{...} \]`. To render nicely in Jupyter use `from IPython.display import display, Markdown; display(Markdown(text))`. For comma numbers use `f"{x:,}"`.

### `6_googleSearch_agent.ipynb` — search agent

- `GoogleSerperAPIWrapper()` needs `SERPER_API_KEY`.
- `create_agent(model, tools=[...], system_prompt="you are a google search agent...")`.
- Same invoke shape as math agent. Good reference for grounding answers in live search.

## 7. Apps explained

### `apps/1_qna_bot.py` — Streamlit chat

- `ChatGroq(model="openai/gpt-oss-120b")`.
- `st.session_state.messages` stores history, rendered with `st.chat_message(role).markdown(content)`.
- `st.chat_input` gets query, appends user msg, calls `llm.invoke(messages)`, appends assistant msg.
- Lesson: memory here is just passing the full message list each turn.

### `apps/2_googleSearch_agent.py` — Streamlit search agent with streaming

- `search = GoogleSerperAPIWrapper()`, `model = ChatGroq(model="openai/gpt-oss-20b", temperature=0.5, streaming=True)`.
- `create_agent(model, tools=[search.run], system_prompt=..., checkpointer=MemorySaver())` with `MemorySaver` kept in `st.session_state` so memory survives Streamlit reruns.
- Same `thread_id` (`"Lokesh"`) passed as `{"configurable": {"thread_id": "Lokesh"}}` so follow-ups share conversation state.
- Streams tokens with `agent.stream({...}, {...}, stream_mode="messages")`, accumulating `chunk[0].content` into a `st.empty()` placeholder for live output, then appends the full answer to `st.session_state.history`.
- Run with `streamlit run apps\2_googleSearch_agent.py` (no longer a `input()` CLI loop).

## 8. Core patterns with examples

Direct message call:

```python
from langchain_groq import ChatGroq
llm = ChatGroq(model="openai/gpt-oss-20b")
res = llm.invoke([{"role":"system","content":"You are a translator."},
                  {"role":"user","content":"hello"}])
print(res.content)
```

LCEL chain:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
prompts = ChatPromptTemplate.from_messages([
  ("system","You are a translator to {language}."),
  ("user","{query}")])
chain = prompts | llm | StrOutputParser()
print(chain.invoke({"language":"Telugu","query":"hello"}))
```

Structured output:

```python
from pydantic import BaseModel, Field
class Person(BaseModel):
    name: str = Field(description="name")
    age: int = Field(description="age")
s = llm.with_structured_output(Person, method="json_schema")
print(s.invoke("John is 30").model_dump())
```

Tool + agent:

```python
from langchain_core.tools import tool
from langchain.agents import create_agent
@tool
def add_numbers(a:int,b:int)->int:
    """Add two numbers."""
    return a+b
agent = create_agent(llm, tools=[add_numbers],
  system_prompt="always use tools for calculations")
r = agent.invoke({"messages":[{"role":"user","content":"2+3?"}]})
print(r["messages"][-1].content)
```

## 9. Models used

| Where | Model string | Provider | Notes |
|---|---|---|---|
| Groq default | `openai/gpt-oss-20b` | Groq | Fast, used in dynamic/5/6/2 |
| Groq large | `openai/gpt-oss-120b` | Groq | Structured output, QnA app |
| OpenAI | `gpt-4o` | OpenAI | Needs `OPENAI_API_KEY` |
| Google | `chat-bison-001` | Google | Legacy name, may need update |
| Anthropic | `claude-3.0` | Anthropic | Needs key |
| Local | `gemma3` | Ollama | Needs `ollama serve` |

Use exact strings. `openai/gpt-oss-20bopenai/gpt-oss-20b` (duplicated) gives 404.

## 10. Troubleshooting: errors I hit and fixes

1. `NotFoundError 404: model openai/gpt-oss-20bopenai/gpt-oss-20b does not exist` — model string was pasted twice. Fix in `notebooks/dynamic.ipynb`: `ChatGroq(model="openai/gpt-oss-20b")`.
2. `ValidationError: Did not find serper_api_key` at `GoogleSerperAPIWrapper()` — `SERPER_API_KEY` not in env. Fix: set it in `apps/.env` / `notebooks/.env`, ensure `load_dotenv()` runs first.
3. `DeprecationWarning: langchain-community is being sunset` on `from langchain_community.utilities import GoogleSerperAPIWrapper` — warning only. Migrate to standalone integration package when ready.
4. Empty output (`"\n"`) — model returned little, or system prompt too strict. Simplify prompt, check `res.content`.
5. `print(res.content)` fails on chains — chains ending in `StrOutputParser` return `str`. Use `print(res)`.
6. Ollama errors — run `ollama serve`, `ollama pull gemma3`.
7. Math/LaTeX looks raw — expected with `print()`. Use `display(Markdown(...))` in Jupyter.

## 11. Git and GitHub

Repo: `https://github.com/lokesh9999b/Gen_AI_Series.git`. This folder is its own repo. Do not run git from home directory.

```powershell
cd GENAI-Series
git status --short
git add <files>
git commit -m "message"
git push -u origin main
```

Never commit `env/`, `apps/.env`, `notebooks/.env`.

## 12. Security

Keys in `.env` are secrets. If a key was committed, pasted in a screenshot, or printed in logs, rotate it at the provider dashboard. Use `.env.example` (names only) for sharing setup.

## 13. Next steps

- Add streaming to Streamlit (`st.write_stream`).
- Add `thread_id` per user in QnA bot instead of one global list.
- Try `openai/gpt-oss-120b` vs `20b` on structured tasks and note accuracy/latency.
- Replace Serper import with current standalone package and compare.
- Add a RAG notebook (loader → splitter → vector store → retriever → chain).
