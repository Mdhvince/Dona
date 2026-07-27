from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agent import (collect_queries, collect_sources, conversation_only,
                       current_turn, parse_citations, choose_route, tool_outcomes)
from src.tools import TOOL_ERROR


A24 = {"path": "/tmp/avis_2024.pdf", "page": "2"}
RIB = {"path": "/tmp/rib.pdf", "page": None}


def tool_message(artifact):
    return ToolMessage(content="extraits", tool_call_id="1", artifact=artifact)


def test_collect_sources_aggregates_and_deduplicates():
    messages = [HumanMessage(content="q"),
                tool_message([A24, RIB]),
                tool_message([A24]),
                AIMessage(content="réponse")]
    collected = collect_sources(messages)
    assert [(s["path"], s["page"]) for s in collected] == [
        (A24["path"], A24["page"]), (RIB["path"], RIB["page"])]


def test_collect_sources_labels_disambiguate_same_file_names():
    same_name_2024 = {"path": "/drive/Exercice_2024/Compte de résultat.pdf", "page": "1"}
    same_name_2025 = {"path": "/drive/Exercice_2025/Compte de résultat.pdf", "page": "1"}
    sources = collect_sources([tool_message([same_name_2024, same_name_2025, A24])])
    assert [s["label"] for s in sources] == [
        "Compte de résultat.pdf (Exercice_2024)",
        "Compte de résultat.pdf (Exercice_2025)",
        "avis_2024.pdf"]


def test_collect_sources_skips_missing_artifacts_and_paths():
    messages = [tool_message(None), tool_message([{"path": None, "page": "1"}])]
    assert collect_sources(messages) == []


CHUNKS = [{"id": "3f2a", "path": "/tmp/avis_2024.pdf", "page": "2"},
          {"id": "b7c1", "path": "/tmp/rib.pdf", "page": None}]


def test_parse_citations_resolves_markers_and_strips_them():
    text, cited = parse_citations("Le solde est de 9570 euros [3f2a].", CHUNKS)
    assert text == "Le solde est de 9570 euros."
    assert cited == [CHUNKS[0]]


def test_parse_citations_keeps_order_and_deduplicates():
    text, cited = parse_citations("A [b7c1] B [3f2a] C [b7c1]", CHUNKS)
    assert cited == [CHUNKS[1], CHUNKS[0]]
    assert "[" not in text


def test_parse_citations_drops_unknown_markers():
    text, cited = parse_citations("Inventé [dead] et vrai [3f2a]", CHUNKS)
    assert cited == [CHUNKS[0]]
    assert "dead" not in text


def test_parse_citations_without_markers():
    assert parse_citations("Aucune citation ici", CHUNKS) == ("Aucune citation ici", [])


def test_collect_queries_reads_tool_calls_in_order():
    messages = [
        AIMessage(content="", tool_calls=[{"name": "rag_medhys_files", "id": "1",
                                           "args": {"queries": ["a", "b"]}}]),
        tool_message([]),
        AIMessage(content="", tool_calls=[{"name": "rag_medhys_files", "id": "2",
                                           "args": {"queries": ["c"]}}]),
    ]
    assert collect_queries(messages) == ["a", "b", "c"]


def test_collect_queries_strip_tool_calls():
    assert collect_queries([HumanMessage(content="q"), AIMessage(content="r")]) == []


def test_tool_outcomes_maps_names_and_counts_successes():
    messages = [
        AIMessage(content="", tool_calls=[
            {"name": "rag_medhys_files", "id": "1", "args": {}},
            {"name": "calendar_pro_list_events", "id": "2", "args": {}}]),
        ToolMessage(content="extraits ok", tool_call_id="1"),
        ToolMessage(content=f"{TOOL_ERROR} permission", tool_call_id="2"),
    ]
    assert tool_outcomes(messages) == (["calendar_pro_list_events"], 1)


def test_tool_outcomes_detects_framework_error_status():
    messages = [
        AIMessage(content="", tool_calls=[{"name": "outil", "id": "1", "args": {}}]),
        ToolMessage(content="boom", tool_call_id="1", status="error"),
    ]
    assert tool_outcomes(messages) == (["outil"], 0)


def test_tool_outcomes_clean_run():
    assert tool_outcomes([HumanMessage(content="q"), tool_message([A24]),
                          AIMessage(content="r")]) == ([], 1)


def test_current_turn_slices_from_last_human_message():
    history = [HumanMessage(content="q1"), AIMessage(content="r1"),
               HumanMessage(content="q2"), tool_message([A24]), AIMessage(content="r2")]
    assert current_turn(history) == history[2:]


def test_current_turn_without_human_message():
    messages = [AIMessage(content="r")]
    assert current_turn(messages) == messages


def test_conversation_only_strips_past_excerpts_and_keeps_the_current_turn():
    history = [
        HumanMessage(content="q1"),
        AIMessage(content="", tool_calls=[{"name": "rag", "id": "1", "args": {}}]),
        tool_message([A24]),
        AIMessage(content="r1"),
        HumanMessage(content="q2"),
        AIMessage(content="", tool_calls=[{"name": "rag", "id": "2", "args": {}}]),
        tool_message([RIB]),
    ]
    kept = conversation_only(history)
    assert [type(m).__name__ for m in kept] == [
        "HumanMessage", "AIMessage", "HumanMessage", "AIMessage", "ToolMessage"]
    # The past answer survives without its tool call, the current turn is intact
    assert kept[1].content == "r1" and not kept[1].tool_calls
    assert kept[3].tool_calls and kept[4] is history[6]


class FakeRouterLLM:
    def __init__(self, verdict=None, error=False):
        self.verdict = verdict
        self.error = error

    def invoke(self, prompt):
        if self.error:
            raise RuntimeError("router down")
        return AIMessage(content=self.verdict)


def test_route_sends_small_talk_to_the_conversation_branch():
    assert choose_route(FakeRouterLLM("CONVERSATION"), "merci") == "conversation"


def test_route_defaults_to_the_tool_agent():
    # Anything but an explicit CONVERSATION verdict, including junk, a
    # refusal or a router failure: answering without tools would hallucinate
    assert choose_route(FakeRouterLLM("OUTILS"), "mon SIRET ?") == "agent"
    assert choose_route(FakeRouterLLM("je ne sais pas"), "mon SIRET ?") == "agent"
    assert choose_route(FakeRouterLLM(""), "mon SIRET ?") == "agent"
    assert choose_route(FakeRouterLLM(error=True), "mon SIRET ?") == "agent"


def test_conversation_only_leaves_a_single_turn_untouched():
    messages = [HumanMessage(content="q"), tool_message([A24]), AIMessage(content="r")]
    assert conversation_only(messages) == messages
