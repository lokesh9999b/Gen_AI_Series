# GENAI-Series

Hands-on GenAI + LangChain learning series: notebooks + small Streamlit/agent apps.

## Layout

```text
GENAI-Series/
├── apps/
│   ├── 1_qna_bot.py           # Streamlit QnA bot (ChatGroq)
│   └── 2_googleSearch_agent.py # Groq + Serper search agent
├── notebooks/
│   ├── 1_basic_langchain_wih_openai.ipynb
│   ├── 3_structured_output.ipynb
│   ├── 4_ollama.ipynb
│   ├── 5_basic_agents.ipynb
│   ├── 6_googleSearch_agent.ipynb
│   └── dynamic.ipynb
├── requirements.txt
├── .env.example
├── .gitignore
└── .gitattributes
```

## Setup

```powershell
cd GENAI-Series
python -m venv env
.\env\Scripts\activate
pip install -r requirements.txt
Copy-Item .env.example apps\.env
# edit apps\.env with real keys
```

Required keys (see `.env.example`): `GROQ_API_KEY`, `SERPER_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ANTROPIC_API_KEY`, `HUGGINGFACEHUB_API_KEY`. Only set what you use.

## Run

```powershell
# notebooks
jupyter notebook notebooks

# apps
streamlit run apps\1_qna_bot.py
python apps\2_googleSearch_agent.py
```

## Git / GitHub

This folder is meant to be its own repo. Do **not** commit from `C:\Users\91897\` (home dir has its own `.git`).

```powershell
cd C:\Users\91897\Desktop\GENAI-Series
git init
git branch -M main
git add .gitignore .gitattributes .env.example README.md requirements.txt apps\*.py notebooks\*.ipynb
git commit -m "Initial GENAI-Series commit"
gh repo create GENAI-Series --private --source=. --push
# or: git remote add origin https://github.com/<you>/GENAI-Series.git; git push -u origin main
```

Never commit `env/`, `apps/.env`, `notebooks/.env`.

## Security

If API keys were ever committed or pasted in logs, rotate them immediately at OpenAI / Google / Anthropic / Groq / HuggingFace / Serper dashboards.
