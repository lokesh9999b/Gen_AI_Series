# SQL Task Agent

A conversational SQL agent built with LangChain, LangGraph, Groq, SQLAlchemy, and Streamlit. Ask questions about your task data in natural language, and the agent can look up tasks or propose changes through a chat interface.

This project is part of the [Gen AI Series](https://github.com/lokesh9999b/Gen_AI_Series). The instructions below assume you have cloned that repository and run commands from its root directory.

## Features

- **Natural-language task management:** ask to list, create, update, or delete tasks in the demo database.
- **Streamlit chat interface:** interactive chat with conversation history and streamed responses.
- **Database access through SQLAlchemy:** SQLite is configured by default; other supported SQLAlchemy databases can be configured with a suitable database driver.
- **SQL guardrails:** restrict queries to configured tables, cap reads, block selected dangerous statements, and require explicit confirmation before supported writes.
- **Demo database setup:** create the sample `tasks` table automatically on startup.

> **Important:** The guardrails are application-level safeguards, not a substitute for database permissions or a security review. Start with the demo SQLite database. Do not connect this demo to sensitive or production data without reviewing the code, configuring least-privilege database credentials, and testing the guardrails for your use case.

## How it works

1. The Streamlit app loads settings from environment variables and connects to the configured database.
2. A LangChain SQL toolkit provides database tools to the LangGraph agent, which uses a Groq chat model to interpret a user's request.
3. The query tool applies the configured table scope and read limit. Supported `INSERT`, `UPDATE`, and `DELETE` operations are staged until the user confirms; the app's **Confirm** and **Cancel** controls handle that decision.
4. The app streams the assistant's user-facing response and formats task results for display.

## Requirements

- Python 3.11 (recommended; this is the version documented by the parent project)
- Git
- A [Groq API key](https://console.groq.com/keys)

## Run locally

### 1. Clone the repository and enter its root

```bash
git clone https://github.com/lokesh9999b/Gen_AI_Series.git
cd Gen_AI_Series
```

All commands below are run from `Gen_AI_Series/`, the directory containing `requirements.txt`.

### 2. Create and activate a virtual environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

The project keeps dependencies in the repository-root `requirements.txt`:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Configure your Groq API key

Copy the example environment file into `apps/` and edit it to add your key.

**macOS / Linux**

```bash
cp .env.example apps/.env
```

**Windows PowerShell**

```powershell
Copy-Item .env.example apps\.env
```

Open `apps/.env` and set:

```dotenv
GROQ_API_KEY=your_groq_api_key
```

Keep the key private. Do not commit `apps/.env` or paste a real API key into source control. The agent's default model is `openai/gpt-oss-120b` through Groq.

### 5. Start the app

From the repository root, run:

```bash
streamlit run apps/sql_agent/app.py
```

Streamlit will print a local URL (typically <http://localhost:8501>) in the terminal and usually open it in your browser. The app creates the demo `tasks` table automatically and stores the SQLite database at `apps/my_tasks.db`.

## Try it out

Example prompts:

- “Show me the latest tasks.”
- “Create a task to review the project proposal.”
- “Mark the project proposal task as completed.”
- “Delete the task about the project proposal.”

Read requests run within the configured limit. For a supported write, review the agent's request and explicitly confirm it in the app before the database is changed. Cancel to leave the database unchanged.

## Configuration

The app has safe demo defaults; these optional variables can be added to `apps/.env` to customize them:

| Variable | Default | Description |
| --- | --- | --- |
| `GROQ_API_KEY` | — | Required credential for Groq. |
| `SQL_AGENT_DB_URI` | SQLite database at `apps/my_tasks.db` | SQLAlchemy connection URI. |
| `SQL_AGENT_MODEL` | `openai/gpt-oss-120b` | Groq model name. |
| `SQL_AGENT_TABLES` | `tasks` | Comma-separated table allow-list. |
| `SQL_AGENT_COLUMNS` | `id,title,description,status,created_at` | Columns used when formatting task results. |
| `SQL_AGENT_READ_LIMIT` | `10` | Maximum number of rows for reads without an explicit limit. |
| `SQL_AGENT_INIT_DEMO` | `1` | Set to `0` to skip creating the demo table. |
| `SQL_AGENT_THREAD_ID` | `1` | Conversation thread identifier. |
| `SQL_AGENT_TEMPERATURE` | `0.0` | Model temperature. |
| `SQL_AGENT_TITLE` | `SQL Task Agent` | Streamlit page title. |

Example configuration for the default SQLite demo:

```dotenv
SQL_AGENT_DB_URI=sqlite:///apps/my_tasks.db
SQL_AGENT_INIT_DEMO=1
SQL_AGENT_TABLES=tasks
SQL_AGENT_COLUMNS=id,title,description,status,created_at
SQL_AGENT_READ_LIMIT=10
```

The configured tables and columns should match the database schema and the agent's intended use. For a non-SQLite database, set an appropriate SQLAlchemy URI, install its required Python database driver, and set `SQL_AGENT_INIT_DEMO=0` unless you specifically intend to use the demo schema.

## Run the tests

The SQL agent's test suite covers guardrails and response formatting and does not require a Groq API key:

```bash
python -m pytest apps/sql_agent/tests -q
```

## Project structure

```text
apps/sql_agent/
├── app.py          # Streamlit UI and confirmation flow
├── agent.py        # Model, SQL toolkit, prompt, and agent assembly
├── config.py       # Environment-driven configuration and defaults
├── db.py           # Database connection and demo schema
├── guardrails.py   # SQL policy checks and read limits
├── tools.py        # Guarded SQL tool wrapper
├── display.py      # Streamed answer filtering and formatting
└── tests/          # Guardrail and display tests
```

## Troubleshooting

- **Missing Groq key:** check that `apps/.env` exists and includes `GROQ_API_KEY=...`; restart Streamlit after changing environment settings.
- **App cannot find its entry point:** make sure the terminal is at the repository root before running the Streamlit command.
- **Database connection error:** verify `SQL_AGENT_DB_URI` and, for databases other than SQLite, install the matching SQLAlchemy driver.
- **Unexpected read scope or results:** verify `SQL_AGENT_TABLES`, `SQL_AGENT_COLUMNS`, and `SQL_AGENT_READ_LIMIT` match your database and intended configuration.

## Limitations and security notes

- This is a learning/demo application, not a production-ready database administration tool.
- The default configuration is for a task table; changing the table allow-list does not automatically redesign the agent prompt or schema expectations. Review and adapt the agent before using a different schema.
- Guardrails use application-level SQL checks and are not a complete SQL parser or a security boundary. Use a dedicated database account with only the permissions the app needs, and avoid granting schema or administrative privileges.
- The SQLite demo data is local to your machine. The LangGraph conversation checkpoint uses in-memory storage and is not durable across app restarts.

## Related files

- [Parent project README](../../README.md)
- [Environment template](../../.env.example)
- [SQL agent source](.)
