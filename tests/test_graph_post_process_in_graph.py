"""closure guard: every post-synthesis enrichment of the brief happens inside a graph node
(post_process), never on the returned object after invoke. The finding named build_brief's
post-hoc Pydantic mutation (docs/design_review/02_remediation_plan.md:399); build_brief is gone and
invoke_pipeline returns a deepcopy. This test fails if post_process leaves the graph."""

from __future__ import annotations


def test_post_process_is_a_node_between_synthesis_and_ml_prediction():
    import aer.graph as g

    graph = g._build_graph().get_graph()
    edges = {(e.source, e.target) for e in graph.edges}
    assert ("synthesize", "post_process") in edges
    assert ("post_process", "ml_prediction") in edges


def test_build_brief_no_longer_exists():
    import aer.graph as g

    assert not hasattr(g, "build_brief")
