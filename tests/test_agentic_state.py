from __future__ import annotations

from clinicalclaw.agentic import create_initial_state


def test_graph_state_initialization() -> None:
    state = create_initial_state("Does aspirin reduce platelet aggregation?")

    assert state == {"question": "Does aspirin reduce platelet aggregation?"}
