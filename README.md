# AI-HR-Telegram-Bot
# AI-HR-Telegram-Bot 🚀

An autonomous, production-ready Telegram Assistant Bot built using a state-machine architecture. The bot handles corporate information lookup natively and features an intelligent fallback loop to query the live internet for answers when local documentation lacks context.

## ✨ Features

- **State-Machine Architecture:** Powered by **LangGraph** to manage predictable, multi-step conversation logic and edge routing.
- **Local RAG Lookup:** Automatically scans `handbook.md` to extract internal company policies.
- **Autonomous Web Search Falling:** Integrates **Tavily AI Search** to look up current live web documentation or global facts dynamically if local context is missing.
- **Multi-User Authentication:** Built-in secure user registration (`/register`) and session tracking (`/login`) to guard private company data.
- **Isolated Session Memory:** Uses LangGraph's checkpointer to keep conversation history distinct for every single unique Telegram user.
- **Observability & Tracing:** Outfitted with **AgentSpan (OpenTelemetry)** hooks to emit execution traces of model calls and tool operations.

## 🛠️ The Tech Stack

- **Framework:** [LangGraph](https://github.com)
- **AI Brain Endpoints:** [OpenRouter API](https://openrouter.ai) (Gemini 1.5 Flash via `openrouter/free`)
- **Agentic Search Engine:** [Tavily API](https://tavily.com)
- **Messaging Interface:** [python-telegram-bot](https://github.com)
- **Telemetry:** [AgentSpan Framework](https://agentspan.com)

---

## 🚀 Local Installation & Setup

Ensure you have Python 3.10+ installed on your system.

### 1. Clone the Repository
```bash
git clone https://github.com
cd AI-HR-Telegram-Bot
```

### 2. Configure Your Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Set Up Your Environment Keys
Create a `.env` file in the root directory:
```ini
TELEGRAM_TOKEN="your_telegram_bot_token"
OPENROUTER_API_KEY="your_openrouter_api_key"
TAVILY_API_KEY="your_tavily_api_key"
LLM_MODEL="openrouter/free"
```

### 4. Run the Application Natively
```bash
python agent.py
```

---

## 🤖 How to Interact with the Bot

1. Locate your bot on Telegram and press **Start**.
2. **Register an Account:** Send `/register <username> <password>` (e.g., `/register abel ABEL`).
3. **Log In:** Send `/login <username> <password>`.
4. Ask a local question: *"What is the vacation policy?"* (Bot reads `handbook.md`).
5. Ask a global or current question: *"What is the latest version of LangGraph?"* (Bot triggers Tavily Search).

---

## ☁️ Continuous Cloud Deployment

This repository includes a native **GitHub Actions** daemon script configured in `.github/workflows/bot-runner.yml`. 

To leave your bot running in the cloud 24/7 completely for free:
1. Push your code files up to your GitHub `main` branch.
2. Go to your repository **Settings** ➔ **Secrets and variables** ➔ **Actions**.
3. Create three Repository Secrets matching your tokens: `TELEGRAM_TOKEN`, `OPENROUTER_API_KEY`, and `TAVILY_API_KEY`.
4. The workflow runner will instantly initialize a server instance and maintain your polling worker background loop indefinitely!
