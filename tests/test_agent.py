from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agent import (CitedSource, collect_queries, collect_sources,
                       current_turn, validate_citations)

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


def test_current_turn_slices_from_last_human_message():
    history = [HumanMessage(content="q1"), AIMessage(content="r1"),
               HumanMessage(content="q2"), tool_message([A24]), AIMessage(content="r2")]
    assert current_turn(history) == history[2:]


def test_current_turn_without_human_message():
    messages = [AIMessage(content="r")]
    assert current_turn(messages) == messages
