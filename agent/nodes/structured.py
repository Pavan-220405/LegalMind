"""Provider-aware structured-output binding for workflow nodes."""

from typing import Any

from config import settings

_GROQ_JSON_SCHEMA_MODELS = {
    "openai/gpt-oss-20b",
}


def bind_structured_output(chat_model: Any, schema: type[Any]) -> Any:
    """Bind a schema using Groq's JSON Schema mode where supported.

    Groq's function-calling mode can fail when a model replies in prose instead
    of emitting the forced tool call. GPT-OSS supports Groq's JSON Schema output
    mode. Use best-effort schema mode so fields with Pydantic defaults stay
    optional, while Pydantic continues to validate the required fields.
    """

    if (
        settings.legal_rag_llm_provider == "groq"
        and settings.legal_model in _GROQ_JSON_SCHEMA_MODELS
    ):
        return chat_model.with_structured_output(
            schema,
            method="json_schema",
            strict=False,
        )
    return chat_model.with_structured_output(schema)
