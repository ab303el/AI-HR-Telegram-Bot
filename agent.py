"""Telegram RAG worker: handbook first, Tavily fallback, AgentSpan traces.

Required env (via .env or shell):
  TELEGRAM_TOKEN
  OPENROUTER_API_KEY   (or OPENAI_API_KEY)
  TAVILY_API_KEY

Optional:
  LLM_MODEL
  OPENAI_BASE_URL
  AGENTSPAN_OTLP_ENDPOINT  (default: http://localhost:8088/v1/traces)
"""

from __future__ import annotations

import operator
import os
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from openai import OpenAI
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from tavily import TavilyClient

from agentspan.agents.tracing import trace_agent_run, trace_llm_call, trace_tool_call

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RAG_DIR = Path(__file__).resolve().parent
load_dotenv(RAG_DIR / ".env")  # always load the RAG folder .env
load_dotenv()  # also allow a CWD .env to override


def resolve_handbook_path() -> Path:
    """Find handbook.md next to agent.py (RAG folder) or in the current working dir."""
    candidates = [
        RAG_DIR / "handbook.md",
        Path.cwd() / "handbook.md",
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    return (RAG_DIR / "handbook.md").resolve()


HANDBOOK_PATH = resolve_handbook_path()
if HANDBOOK_PATH.is_file():
    print(f"Handbook detected: {HANDBOOK_PATH}")
else:
    print(f"WARNING: handbook.md not found (expected at {RAG_DIR / 'handbook.md'})")

MODEL_NAME = os.getenv("LLM_MODEL", "google/gemini-1.5-flash")
AGENT_NAME = "telegram-rag-agent"
TELEGRAM_MAX_CHARS = 4000  # under Telegram's 4096 hard limit
OTLP_ENDPOINT = os.getenv(
    "AGENTSPAN_OTLP_ENDPOINT",
)

EMPTY_HANDBOOK_MARKERS = (
    "",
    "No explicit rules found in handbook.",
    "Error: handbook.md file is missing from the project folder.",
)

_client: OpenAI | None = None
_tavily: TavilyClient | None = None

# Simple in-memory auth store (demo only — not for production)
USER_ACCOUNTS: dict[str, str] = {
    "admin": "password123",
}
# chat_id -> authenticated username
ACTIVE_SESSIONS: dict[int, str] = {}

AUTH_REQUIRED_MSG = (
    "You are not logged in.\n"
    "Register: /register <username> <password>\n"
    "Login: /login <username> <password>"
)


def get_llm_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "Set OPENROUTER_API_KEY or OPENAI_API_KEY before generating answers."
            )
        _client = OpenAI(
            base_url=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=api_key,
        )
    return _client


def get_tavily_client() -> TavilyClient:
    global _tavily
    if _tavily is None:
        api_key = os.getenv("TAVILY_API_KEY", "")
        if not api_key:
            raise RuntimeError("Set TAVILY_API_KEY before using web search fallback.")
        _tavily = TavilyClient(api_key=api_key)
    return _tavily


def setup_agentspan_tracing() -> None:
    """Send AgentSpan OTel spans to the local collector (or console)."""
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    provider = TracerProvider(
        resource=Resource.create({"service.name": AGENT_NAME})
    )
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=OTLP_ENDPOINT))
        )
        print(f"AgentSpan tracing → {OTLP_ENDPOINT}")
    except Exception as exc:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        print(f"AgentSpan OTLP unavailable ({exc}); using console exporter")

    trace.set_tracer_provider(provider)


def chunk_text(text: str, limit: int = TELEGRAM_MAX_CHARS) -> list[str]:
    """Split long replies so Telegram does not truncate them."""
    text = text or ""
    if len(text) <= limit:
        return [text] if text else ["No answer."]

    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break
        cut = remaining.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = remaining.rfind(" ", 0, limit)
        if cut < limit // 2:
            cut = limit
        chunks.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    return chunks


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------
class BotState(TypedDict):
    user_message: str
    context_data: str
    source: str  # "handbook" | "tavily" | "none"
    final_answer: str
    # Accumulates across turns per thread_id via MemorySaver
    messages: Annotated[list[dict], operator.add]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def retrieve_handbook_context(state: BotState) -> dict:
    """Local handbook.md reader node."""
    query = state["user_message"]
    with trace_tool_call(AGENT_NAME, "handbook_reader", {"query": query}) as span:
        try:
            content = HANDBOOK_PATH.read_text(encoding="utf-8")
            lines = content.splitlines()
            words = [w for w in query.lower().split() if len(w) > 2]
            matches = [
                line
                for line in lines
                if any(word in line.lower() for word in words)
            ]
            if matches:
                context = "\n".join(matches[:6])
                source = "handbook"
            else:
                context = "No explicit rules found in handbook."
                source = "none"
        except FileNotFoundError:
            context = "Error: handbook.md file is missing from the project folder."
            source = "none"

        if span:
            span.set_attribute("tool.source", source)
            span.set_attribute("tool.context_length", len(context))

    return {"context_data": context, "source": source}


def search_tavily(state: BotState) -> dict:
    """TavilyClient web search node (fallback)."""
    query = state["user_message"]
    with trace_tool_call(AGENT_NAME, "tavily_search", {"query": query}) as span:
        try:
            result = get_tavily_client().search(
                query=query, max_results=4, include_answer=True
            )
            bits: list[str] = []
            if result.get("answer"):
                bits.append(f"Answer: {result['answer']}")
            for hit in result.get("results", []):
                title = hit.get("title", "")
                content = hit.get("content", "")
                url = hit.get("url", "")
                bits.append(f"- {title}: {content} ({url})")
            context = "\n".join(bits) if bits else "Tavily returned no results."
            source = "tavily" if bits else "none"
        except Exception as exc:
            context = f"Tavily search failed: {exc}"
            source = "none"

        if span:
            span.set_attribute("tool.source", source)
            span.set_attribute("tool.context_length", len(context))

    return {"context_data": context, "source": source}


def generate_bot_response(state: BotState) -> dict:
    """LLM answer using handbook/web context plus this chat's memory history."""
    system_instruction = (
        "You are an automated HR Assistant. Prefer company handbook context. "
        "If context came from the web, say so briefly and be careful with accuracy. "
        "Use prior conversation turns when helpful.\n"
        f"--- CONTEXT SOURCE: {state.get('source', 'unknown')} ---\n"
        f"{state['context_data']}"
    )
    history = list(state.get("messages") or [])
    # Keep prompt size bounded while preserving recent turns
    recent = history[-12:]
    llm_messages = [{"role": "system", "content": system_instruction}, *recent]
    if not recent or recent[-1].get("role") != "user":
        llm_messages.append({"role": "user", "content": state["user_message"]})

    with trace_llm_call(AGENT_NAME, MODEL_NAME) as span:
        response = get_llm_client().chat.completions.create(
            model=MODEL_NAME,
            messages=llm_messages,
        )
        answer = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        if span and usage:
            span.set_attribute("llm.prompt_tokens", getattr(usage, "prompt_tokens", 0) or 0)
            span.set_attribute(
                "llm.completion_tokens",
                getattr(usage, "completion_tokens", 0) or 0,
            )
        if span:
            span.set_attribute("llm.answer_length", len(answer))
            span.set_attribute("llm.history_turns", len(recent))

    return {
        "final_answer": answer,
        "messages": [{"role": "assistant", "content": answer}],
    }


# ---------------------------------------------------------------------------
# Conditional router: handbook first, else Tavily
# ---------------------------------------------------------------------------
def route_after_handbook(state: BotState) -> Literal["generate", "tavily"]:
    context = (state.get("context_data") or "").strip()
    source = state.get("source", "none")
    if source == "handbook" and context not in EMPTY_HANDBOOK_MARKERS:
        return "generate"
    return "tavily"


def build_graph():
    builder = StateGraph(BotState)
    builder.add_node("retrieve", retrieve_handbook_context)
    builder.add_node("tavily", search_tavily)
    builder.add_node("generate", generate_bot_response)

    builder.set_entry_point("retrieve")
    builder.add_conditional_edges(
        "retrieve",
        route_after_handbook,
        {"generate": "generate", "tavily": "tavily"},
    )
    builder.add_edge("tavily", "generate")
    builder.add_edge("generate", END)

    # One isolated conversation history per Telegram chat_id (thread_id)
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


rag_agent = build_graph()


def thread_config(chat_id: int | str) -> dict:
    """LangGraph config that isolates MemorySaver state per chat."""
    return {"configurable": {"thread_id": str(chat_id)}}


def is_authenticated(chat_id: int | None) -> bool:
    return chat_id is not None and chat_id in ACTIVE_SESSIONS


# ---------------------------------------------------------------------------
# Telegram auth + worker
# ---------------------------------------------------------------------------
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.effective_chat:
        return
    chat_id = update.effective_chat.id
    if is_authenticated(chat_id):
        user = ACTIVE_SESSIONS[chat_id]
        await update.message.reply_text(
            f"Welcome back, {user}. Ask me about the corporate handbook.\n"
            "Commands: /logout"
        )
        return
    await update.message.reply_text(
        "Welcome to the HR assistant.\n"
        "You must authenticate before chatting.\n\n"
        + AUTH_REQUIRED_MSG
    )


async def register_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message:
        return
    args = context.args or []
    if len(args) != 2:
        await update.message.reply_text("Usage: /register <username> <password>")
        return

    username, password = args[0].strip(), args[1]
    if not username or not password:
        await update.message.reply_text("Username and password cannot be empty.")
        return
    if username in USER_ACCOUNTS:
        await update.message.reply_text(
            f"Username '{username}' already exists. Try /login {username} <password>"
        )
        return

    USER_ACCOUNTS[username] = password
    await update.message.reply_text(
        f"Registered '{username}'. Now run: /login {username} <password>"
    )


async def login_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.effective_chat:
        return
    args = context.args or []
    if len(args) != 2:
        await update.message.reply_text("Usage: /login <username> <password>")
        return

    username, password = args[0].strip(), args[1]
    expected = USER_ACCOUNTS.get(username)
    if expected is None or expected != password:
        await update.message.reply_text("Invalid username or password.")
        return

    chat_id = update.effective_chat.id
    ACTIVE_SESSIONS[chat_id] = username
    await update.message.reply_text(
        f"Logged in as {username}. Your chat memory is isolated to this chat.\n"
        "Ask me anything about the handbook (or /logout)."
    )


async def logout_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.effective_chat:
        return
    chat_id = update.effective_chat.id
    if chat_id in ACTIVE_SESSIONS:
        user = ACTIVE_SESSIONS.pop(chat_id)
        await update.message.reply_text(
            f"Logged out ({user}). Conversation memory for this chat remains stored, "
            "but you must /login again to continue."
        )
        return
    await update.message.reply_text("You were not logged in.\n" + AUTH_REQUIRED_MSG)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.effective_chat:
        return

    chat_id = update.effective_chat.id
    if not is_authenticated(chat_id):
        await update.message.reply_text(AUTH_REQUIRED_MSG)
        return

    user_query = update.message.text or ""
    username = ACTIVE_SESSIONS[chat_id]
    print(f"[Telegram] chat={chat_id} user={username}: {user_query}")

    turn_input: BotState = {
        "user_message": user_query,
        "context_data": "",
        "source": "none",
        "final_answer": "",
        "messages": [{"role": "user", "content": user_query}],
    }

    with trace_agent_run(AGENT_NAME, user_query, model=MODEL_NAME) as span:
        resulting_state = rag_agent.invoke(turn_input, config=thread_config(chat_id))
        if span:
            span.set_attribute("agent.source", resulting_state.get("source", ""))
            span.set_attribute("agent.chat_id", str(chat_id))
            span.set_attribute("agent.username", username)
            span.set_attribute(
                "agent.output_length",
                len(resulting_state.get("final_answer") or ""),
            )

    for part in chunk_text(resulting_state.get("final_answer") or ""):
        await update.message.reply_text(part)


def main():
    setup_agentspan_tracing()

    token = os.getenv("TELEGRAM_TOKEN", "").strip()
    if not token:
        print("No TELEGRAM_TOKEN — terminal simulation:")
        demo_thread = "demo-terminal"
        with trace_agent_run(
            AGENT_NAME, "What is the vacation rule?", model=MODEL_NAME
        ):
            demo = rag_agent.invoke(
                {
                    "user_message": "What is the vacation rule?",
                    "context_data": "",
                    "source": "none",
                    "final_answer": "",
                    "messages": [
                        {"role": "user", "content": "What is the vacation rule?"}
                    ],
                },
                config=thread_config(demo_thread),
            )
        print("Source:", demo["source"])
        print("Context:", demo["context_data"])
        print("Answer:", demo["final_answer"])
        print("History turns:", len(demo.get("messages") or []))
        return

    print("Telegram worker running…")
    print(f"Seeded accounts: {', '.join(USER_ACCOUNTS)} (use /login)")
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("register", register_command))
    app.add_handler(CommandHandler("login", login_command))
    app.add_handler(CommandHandler("logout", logout_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.run_polling()


if __name__ == "__main__":
    main()

