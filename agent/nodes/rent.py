"""Rent and tenancy-law analysis node."""

from typing import Any

from agent.nodes.domain_agent import RetrieveFn, build_domain_agent
from agent.schemas.state import BaseState


def build_rent_agent(chat_model: Any, retrieve_fn: RetrieveFn) -> Any:
    return build_domain_agent("rent and tenancy", "rent", chat_model, retrieve_fn)
