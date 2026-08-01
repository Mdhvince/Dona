import re
from collections import Counter
from datetime import date
from pathlib import Path
from typing import NamedTuple

from langchain.agents import create_agent
from langchain.agents.middleware import (HumanInTheLoopMiddleware, ModelRequest,
                                         dynamic_prompt, wrap_model_call)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph

from src.prompt import CONVERSATION_PROMPT, ROUTER_PROMPT
from src.tools import TOOL_ERROR


CITATION_MARKER = re.compile(r" ?\[([0-9a-f]{4})\]")


class AgentProfile(NamedTuple):
    """Everything one agent branch is made of: its model, its tools, the
    rules it plays by, and the tools it may only run after confirmation."""
    llm: object
    tools: tuple
    system_prompt: str
    tools_needing_confirmation: tuple = ()


def dated_system_prompt(template):
    """Middleware resolving the system prompt at every model call, so the
    date never goes stale in a long-lived process."""
    @dynamic_prompt
    def resolve(request: ModelRequest) -> str:
        return template.format(date=date.today().strftime("%d/%m/%Y"))
    return resolve


def index_of_last_question(messages):
    """Where the turn being answered right now begins.
    :param messages: The message history.
    :return: The index of the last question, or None when the history holds none.
    """
    for index in reversed(range(len(messages))):
        if isinstance(messages[index], HumanMessage):
            return index
    return None


def is_question_or_answer(message):
    """Tell apart what was said from the tool machinery that produced it.
    :param message: The message to classify.
    :return: True for a question or an answer carrying text, False for a tool
             result and for the empty answer that only carried a tool call.
    """
    if isinstance(message, ToolMessage):
        return False
    if isinstance(message, AIMessage):
        return bool(message.content)
    return True


def strip_tool_calls(message):
    """Rebuild an answer from its text alone: a tool call kept without its result
    is rejected by some providers.
    :param message: The message to strip.
    :return: The same message, minus the tool calls attached to it.
    """
    if isinstance(message, AIMessage):
        return AIMessage(content=message.content)
    return message


def conversation_only(messages):
    """Past turns keep their conversation (questions and answers) but lose their
    tool calls and retrieved excerpts; the current turn is untouched. Left in
    place, stale excerpts make the model answer from them instead of searching
    again - including "not found" on documents it never looked for.
    :param messages: The message history.
    :return: The history with the tool traces of the past turns removed.
    """
    turn_start = index_of_last_question(messages)
    if turn_start is None:
        turn_start = len(messages)
    past_turns = [strip_tool_calls(message)
                  for message in messages[:turn_start] if is_question_or_answer(message)]
    return past_turns + messages[turn_start:]


@wrap_model_call
def fresh_retrieval(request, handler):
    """Every question triggers its own search: the model cannot lean on the
    excerpts of a previous turn."""
    return handler(request.override(messages=conversation_only(request.messages)))


def router_history(messages, kept_exchanges=6, excerpt_length=300):
    """The past conversation as short plain-text lines, so the router can
    resolve a follow-up ("et le mois d'avant ?") from what was being
    discussed. Words only, bounded: never the tool traces, and never more
    than the router needs to decide.
    :param messages: The message history, current question included.
    :return: One line per past question or answer, empty for a first question.
    """
    past = [message for message in messages[:-1] if is_question_or_answer(message)]
    return "\n".join(
        f"{'Moi' if isinstance(message, HumanMessage) else 'Assistant'} : "
        f"{str(message.content)[:excerpt_length]}"
        for message in past[-kept_exchanges:])


def choose_route(router_llm, messages):
    """CONVERSATION only when the message clearly needs no data, LOCAL for
    what the local model handles (documents, calendars), and CRITIQUE -
    banking, emails - for everything else, unreadable verdicts and doubt
    included: the critical agent can do all the local one can, the reverse
    is false. A router failure falls back on the local branch instead, so
    the assistant keeps working when the hosted router is unreachable."""
    prompt = ROUTER_PROMPT.format(history=router_history(messages),
                                  message=messages[-1].content)
    try:
        verdict = str(router_llm.invoke(prompt).content).upper()
    except Exception:
        return "agent_local"
    if "CONVERSATION" in verdict:
        return "conversation"
    if "LOCAL" in verdict:
        return "agent_local"
    return "agent_critical"


def build_graph(router_llm, conversation_llm, local_profile, critical_profile, checkpointer=None):
    """Router in front of three branches sharing one message history: small
    talk answered directly by a local no-thinking call, documents and
    calendars by the local agent, banking and emails by the critical agent
    (stronger hosted model). The checkpointer sits on the parent graph;
    interrupts and resumes propagate through it."""
    def conversation(state):
        prompt = CONVERSATION_PROMPT.format(date=date.today().strftime("%d/%m/%Y"))
        messages = [SystemMessage(content=prompt), *conversation_only(state["messages"])]
        return {"messages": [conversation_llm.invoke(messages)]}

    graph = StateGraph(MessagesState)
    graph.add_node("conversation", conversation)
    graph.add_node("agent_local", build_agent(local_profile))
    graph.add_node("agent_critical", build_agent(critical_profile))
    graph.add_conditional_edges(
        START,
        lambda state: choose_route(router_llm, state["messages"]),
        {"conversation": "conversation",
         "agent_local": "agent_local",
         "agent_critical": "agent_critical"}
    )
    graph.add_edge("conversation", END)
    graph.add_edge("agent_local", END)
    graph.add_edge("agent_critical", END)
    return graph.compile(checkpointer=checkpointer)


def build_agent(profile, checkpointer=None):
    """Agent over one profile: the model, tools and system prompt of a
    branch, all built by the caller. Passing a checkpointer enables
    multi-turn conversations (one thread_id per conversation). No
    response_format on purpose: a constrained output grammar competes with
    tool calling on some models; the agent cites inline instead (see
    parse_citations). Tools are listed as needing a confirmation when they
    have side effects: the graph interrupts before running them and only
    resumes on an explicit approval."""
    middleware = [dated_system_prompt(profile.system_prompt), fresh_retrieval]
    if profile.tools_needing_confirmation:
        middleware.append(HumanInTheLoopMiddleware(
            interrupt_on={name: True for name in profile.tools_needing_confirmation}))
    return create_agent(
        model=profile.llm,
        tools=list(profile.tools),
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
    """Messages of the ongoing exchange: everything from the last question
    onward (the checkpointer brings the whole history back).
    :param messages: The message history.
    :return: The messages of the current turn, or the whole history when it holds no question.
    """
    turn_start = index_of_last_question(messages)
    return messages if turn_start is None else messages[turn_start:]


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


def tool_outcomes(messages):
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


def models_of_turn(messages):
    """Names of the models that generated the turn's answers, read from the
    response metadata the framework stamps ("model" for Ollama, "model_name"
    for OpenAI-compatible providers) - never from the model's own words.
    :param messages: The messages of the current turn.
    :return: Distinct model names, in order of first answer.
    """
    names = []
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        metadata = getattr(message, "response_metadata", None) or {}
        name = metadata.get("model_name") or metadata.get("model")
        if name and name not in names:
            names.append(name)
    return names


def collect_queries(messages):
    """Search queries the agent actually sent to the RAG tool, in order."""
    queries = []
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            queries.extend(call.get("args", {}).get("queries") or [])
    return queries
