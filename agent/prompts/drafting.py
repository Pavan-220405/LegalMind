"""Prompt for careful first drafts of legal communications."""

DRAFTING_PROMPT = """
Prepare a clearly labeled draft for the user's requested action. This is a draft
for review, not a filed or sent document. Use only the facts provided by the
user or explicitly identified from the uploaded PDF; keep those sources
distinct. Do not add legal sections, allegations, dates, amounts, recipients,
jurisdiction, or procedural claims unless supported by supplied facts or
verified legal information. Use bracketed placeholders for missing details.
Keep the tone factual and civil. Do not imply that the document has been sent.
""".strip()
