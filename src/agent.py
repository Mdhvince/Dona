import re
from collections import Counter
from datetime import date
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import (HumanInTheLoopMiddleware, ModelRequest,
                                         dynamic_prompt, wrap_model_call)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from src.prompt import CONVERSATION_PROMPT, ROUTER_PROMPT, SYSTEM_PROMPT
from src.tools import TOOL_ERROR


CITATION_MARKER = re.compile(r" ?\[([0-9a-f]{4})\]")


@dynamic_prompt
def system_prompt(request: ModelRequest) -> str:
    """Resolved at every model call so the date never goes stale in a
    long-lived process."""
    return SYSTEM_PROMPT.format(date=date.today().strftime("%d/%m/%Y"))


def conversation_only(messages):
    """Past turns keep their conversation (questions and answers) but lose
    their tool calls and retrieved excerpts; the current turn is untouched.
    Left in place, stale excerpts make the model answer from them instead of
    searching again - including "not found" on documents it never looked
    for. Tool calls are dropped along with their results: a dangling call
    without its result is rejected by some providers."""
    turn_start = len(messages)
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            turn_start = i
            break

    kept = []
    for message in messages[:turn_start]:
        if isinstance(message, ToolMessage):
            continue
        if isinstance(message, AIMessage):
            if not message.content:
                continue
            message = AIMessage(content=message.content)
        kept.append(message)
    return kept + messages[turn_start:]


@wrap_model_call
def fresh_retrieval(request, handler):
    """Every question triggers its own search: the model cannot lean on the
    excerpts of a previous turn."""
    return handler(request.override(messages=conversation_only(request.messages)))


def route(router_llm, message):
    """CONVERSATION only when the message clearly needs no data: anything
    else, including any doubt or an unreadable answer, goes to the tool
    agent. Misrouting a real question would answer it without tools."""
    try:
        verdict = router_llm.invoke(ROUTER_PROMPT.format(message=message)).content
    except Exception:
        return "agent"
    return "conversation" if "CONVERSATION" in str(verdict).upper() else "agent"


def build_graph(llm, router_llm, tools, checkpointer=None, tools_needing_confirmation=()):
    """Router in front of two branches sharing one message history: small
    talk answered directly by a no-thinking call, everything else by the
    tool agent (citations, confirmations, fresh retrieval). The checkpointer
    sits on the parent graph; interrupts and resumes propagate through it."""
    agent_with_tools = build_agent(llm, tools, tools_needing_confirmation=tools_needing_confirmation)

    def conversation(state):
        prompt = CONVERSATION_PROMPT.format(date=date.today().strftime("%d/%m/%Y"))
        messages = [SystemMessage(content=prompt), *conversation_only(state["messages"])]
        return {"messages": [router_llm.invoke(messages)]}

    graph = StateGraph(MessagesState)
    graph.add_node("conversation", conversation)
    graph.add_node("agent", agent_with_tools)
    graph.add_conditional_edges(
        START,
        lambda state: route(router_llm, state["messages"][-1].content),
        {"conversation": "conversation", "agent": "agent"}
    )
    graph.add_edge("conversation", END)
    graph.add_edge("agent", END)
    return graph.compile(checkpointer=checkpointer)


def build_agent(llm, tools, checkpointer=None, tools_needing_confirmation=()):
    """Agent over the tools it is given: the documents RAG and the MCP
    servers (calendar, qonto...) are all built by the caller. Passing a
    checkpointer enables multi-turn conversations (one thread_id per
    conversation). No response_format on purpose: a constrained output
    grammar competes with tool calling on some models; the agent cites
    inline instead (see parse_citations). Tools are listed as needing a
    confirmation when they have side effects: the graph interrupts before
    running them and only resumes on an explicit approval."""
    middleware = [system_prompt, fresh_retrieval]
    if tools_needing_confirmation:
        middleware.append(HumanInTheLoopMiddleware(
            interrupt_on={name: True for name in tools_needing_confirmation}))
    return create_agent(
        model=llm,
        tools=list(tools),
        middleware=middleware,
        checkpointer=checkpointer)


def parse_citations(answer_text, retrieved):
    """Resolve the [id] markers the agent wrote into the chunks they point
    to, and strip them from the displayed answer. Fully deterministic: the
    citing model is the one that read the excerpts, and an id matching no
    retrieved chunk is dropped. Returns (clean text, cited sources)."""
    by_id = {info["id"]: info for info in retrieved if info.get("id")}
    cited, seen = [], set()
    for marker in CITATION_MARKER.findall(answer_text or ""):
        info = by_id.get(marker)
        if info and marker not in seen:
            seen.add(marker)
            cited.append(info)
    return CITATION_MARKER.sub("", answer_text or ""), cited


def current_turn(messages):
    """Messages of the ongoing exchange: everything from the last human
    message onward (the checkpointer brings the whole history back)."""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return messages[i:]
    return messages


def source_label(info):
    return info.get("label") or Path(info["path"]).name


def _add_labels(sources):
    """Give each source a label that is unique among them: the file name,
    plus its parent folder when several paths share that name (accounting
    documents repeat identically across fiscal years). Without it, citation
    matching and the UI would confuse two different files."""
    names = Counter(Path(info["path"]).name for info in sources)
    for info in sources:
        name = Path(info["path"]).name
        info["label"] = (name if names[name] == 1
                         else f"{name} ({Path(info['path']).parent.name})")
    return sources


def collect_sources(messages):
    """Deduplicated {path, page, label} of every document retrieved during
    the run, read from the tool message artifacts (never from the LLM)."""
    sources, seen = [], set()
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        for info in message.artifact or []:
            key = (info.get("path"), info.get("page"))
            if info.get("path") and key not in seen:
                seen.add(key)
                sources.append(dict(info))
    return _add_labels(sources)


def tool_failures(messages):
    """(failed tool names, successful call count) for the run, detected in
    code (TOOL_ERROR sentinel emitted by the tool layer, or error status set
    by the framework) so the UI can report failures deterministically: the
    LLM never narrates them."""
    call_names = {call["id"]: call["name"]
                  for message in messages
                  for call in (getattr(message, "tool_calls", None) or [])}
    failed, succeeded = [], 0
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        errored = (getattr(message, "status", None) == "error"
                   or TOOL_ERROR in str(message.content))
        if errored:
            failed.append(call_names.get(message.tool_call_id, "outil inconnu"))
        else:
            succeeded += 1
    return failed, succeeded


def collect_queries(messages):
    """Search queries the agent actually sent to the RAG tool, in order."""
    queries = []
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            queries.extend(call.get("args", {}).get("queries") or [])
    return queries
