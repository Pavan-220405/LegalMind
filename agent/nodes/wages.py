"""Wages-law analysis node."""

from typing import Any

from agent.nodes.domain_agent import RetrieveFn, build_domain_agent


def build_wages_agent(chat_model: Any, retrieve_fn: RetrieveFn) -> Any:
    return build_domain_agent("wages", "wages", chat_model, retrieve_fn)
