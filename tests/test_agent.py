from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agent import (CitedSource, CitedSources, collect_queries, collect_sources,
                       current_turn, extract_citations, tool_failures,
                       validate_citations)
from src.tools import TOOL_ERROR


class FakeExtractorLLM:
    def __init__(self, sources=None, error=False):
        self.sources = sources or []
        self.error = error
        self.prompts = []

    def with_structured_output(self, schema):
        return self

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if self.error:
            raise RuntimeError("llm down")
        return CitedSources(sources=self.sources)

A24 = {"path": "/tmp/avis_2024.pdf", "page": "2"}
RIB = {"path": "/tmp/rib.pdf", "page": None}


def tool_message(artifact):
    return ToolMessage(content="extraits", tool_call_id="1", artifact=artifact)


def test_collect_sources_aggregates_and_deduplicates():
    messages = [HumanMessage(content="q"),
                tool_message([A24, RIB]),
                tool_message([A24]),
                AIMessage(content="réponse")]
    assert collect_sources(messages) == [A24, RIB]


def test_collect_sources_skips_missing_artifacts_and_paths():
    messages = [tool_message(None), tool_message([{"path": None, "page": "1"}])]
    assert collect_sources(messages) == []


def test_collect_queries_reads_tool_calls_in_order():
    messages = [
        AIMessage(content="", tool_calls=[{"name": "rag_medhys_files", "id": "1",
                                           "args": {"queries": ["a", "b"]}}]),
        tool_message([]),
        AIMessage(content="", tool_calls=[{"name": "rag_medhys_files", "id": "2",
                                           "args": {"queries": ["c"]}}]),
    ]
    assert collect_queries(messages) == ["a", "b", "c"]


def test_collect_queries_without_tool_calls():
    assert collect_queries([HumanMessage(content="q"), AIMessage(content="r")]) == []


def test_validate_citations_matches_on_file_and_page():
    cited = [CitedSource(file="avis_2024.pdf", page=2)]
    assert validate_citations(cited, [A24, RIB]) == [A24]


def test_validate_citations_drops_unknown_documents():
    cited = [CitedSource(file="invente.pdf", page=1),
             CitedSource(file="avis_2024.pdf", page=99)]
    assert validate_citations(cited, [A24, RIB]) == []


def test_validate_citations_handles_pageless_and_dedup():
    cited = [CitedSource(file="rib.pdf"), CitedSource(file="rib.pdf")]
    assert validate_citations(cited, [A24, RIB]) == [RIB]


def test_validate_citations_skips_plain_string_entries():
    cited = ["calendar_perso_list_events", CitedSource(file="rib.pdf")]
    assert validate_citations(cited, [A24, RIB]) == [RIB]


def test_extract_citations_validates_against_retrieved():
    llm = FakeExtractorLLM([CitedSource(file="avis_2024.pdf", page=2),
                            CitedSource(file="invente.pdf", page=1)])
    assert extract_citations(llm, "réponse", [A24, RIB]) == [A24]
    assert "avis_2024.pdf" in llm.prompts[0]


def test_extract_citations_skips_the_call_without_candidates():
    llm = FakeExtractorLLM()
    assert extract_citations(llm, "réponse", []) == []
    assert llm.prompts == []


def test_extract_citations_survives_llm_failure():
    assert extract_citations(FakeExtractorLLM(error=True), "réponse", [A24]) == []


def test_tool_failures_maps_names_and_counts_successes():
    messages = [
        AIMessage(content="", tool_calls=[
            {"name": "rag_medhys_files", "id": "1", "args": {}},
            {"name": "calendar_pro_list_events", "id": "2", "args": {}}]),
        ToolMessage(content="extraits ok", tool_call_id="1"),
        ToolMessage(content=f"{TOOL_ERROR} permission", tool_call_id="2"),
    ]
    assert tool_failures(messages) == (["calendar_pro_list_events"], 1)


def test_tool_failures_detects_framework_error_status():
    messages = [
        AIMessage(content="", tool_calls=[{"name": "outil", "id": "1", "args": {}}]),
        ToolMessage(content="boom", tool_call_id="1", status="error"),
    ]
    assert tool_failures(messages) == (["outil"], 0)


def test_tool_failures_clean_run():
    assert tool_failures([HumanMessage(content="q"), tool_message([A24]),
                          AIMessage(content="r")]) == ([], 1)


def test_current_turn_slices_from_last_human_message():
    history = [HumanMessage(content="q1"), AIMessage(content="r1"),
               HumanMessage(content="q2"), tool_message([A24]), AIMessage(content="r2")]
    assert current_turn(history) == history[2:]


def test_current_turn_without_human_message():
    messages = [AIMessage(content="r")]
    assert current_turn(messages) == messages
