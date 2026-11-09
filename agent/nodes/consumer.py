"""Consumer-law analysis node."""

from typing import Any

from agent.nodes.domain_agent import RetrieveFn, build_domain_agent


def build_consumer_agent(chat_model: Any, retrieve_fn: RetrieveFn) -> Any:
    return build_domain_agent("consumer protection", "consumer", chat_model, retrieve_fn)
