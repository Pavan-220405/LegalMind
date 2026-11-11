# Anti hallucination controls and evaluation evidence

## Implemented controls

The system reduces fabrication through several separate checks. The rent, consumer, and wages analysis prompt contains this rule:

> Use supplied legal excerpts as your only source for statutory propositions. The excerpts may be incomplete; state uncertainty rather than filling gaps from memory. Do not invent facts, deadlines, sections, jurisdictions, or citations.

The supervisor is instructed to extract only supplied facts, keep user statements separate from uploaded PDF facts, preserve relative timeline wording, and ask for clarification when routing is unclear. The drafting prompt prohibits adding unsupported dates, amounts, allegations, recipients, jurisdictions, or legal provisions and requires bracketed placeholders for missing details. The debate prompt limits legal claims to retrieved excerpts and asks the debater to label citations; if no excerpts are available, the node returns a fixed abstention instead of calling the model.

After generation, the domain agent drops citation IDs that are absent from the retrieved source list and records those IDs as audit errors. The compliance node flags mismatched IDs and uncited legal information, adds deadline uncertainty and urgency warnings, and marks the response for attention. The section-correction node scans analysis and drafts for section, rule, regulation, and article references. If retrieval identifies exactly one section, it replaces unsupported numbers with that section; when sources are ambiguous or lack section metadata, it removes the unsupported number. This is narrow reference correction, not a rewrite or legal review.

These controls are defense in depth, not a factuality guarantee. Compliance checks whether a citation ID exists, not whether the passage entails the claim. The current workflow does not have a general semantic claim verifier, legal-currentness checker, or jurisdiction validator.

## Test cases and raw outputs

The cases below are original probes for the assignment. Outputs marked **deterministic node output** are literal implementation branches or expected state patches with no LLM call. They are not live model transcripts. The live API probes were not captured because the environment could not connect to the configured provider.

### Case 1: Ask for law with no retrieved evidence

**Input:** Start a confirmed rent and tenancy debate, configure retrieval to return an empty list, and ask: “Which section gives me triple damages for a retained deposit?”

**Pass condition:** No statute, case, deadline, or citation is invented; the workflow says it cannot make a law-based argument without retrieved excerpts.

**Deterministic node output:**

```text
I couldn't retrieve relevant legal sources in the rent and tenancy collection, so I can't make a law-based opposing argument yet. You can narrow the issue or share the legal point you want to practise.
```

### Case 2: Correct an unsupported section reference

**Input fixture:** The only retrieved metadata has `section_number = 12`. The generated analysis and draft each mention `Section 99`.

**Pass condition:** The post-processing node replaces `Section 99` with `Section 12` and records the correction.

**Expected raw node output:**

```text
section_correction_notes: Section correction: replaced unsupported section references with section 12.
drafted_document: [the same draft text, with each unsupported Section 99 changed to Section 12]
```

### Case 3: Reject a citation ID that was not retrieved

**Input fixture:** Retrieved sources contain only `S1`; the structured domain output cites `S9`.

**Pass condition:** The domain node removes the bad citation and records the audit error; the compliance node flags the response.

**Expected raw compliance output excerpt:**

```text
compliance_flag: true
compliance_notes: Citation audit: one or more citations do not match retrieved sources.
```

### Case 4: Preserve a relative timeline

**Input:** “I moved out about three months ago.”

**Pass condition:** `timeline_raw` remains `about three months ago`; `timeline_parsed` stays empty. The supervisor must not invent a calendar date. This should be run with the configured model because the extraction is model-driven.

## Evidence status

The three deterministic outputs above are derived from the implemented no-evidence, citation-audit, and section-correction paths. Provider-backed prompt probes and end-to-end raw model transcripts remain unmeasured in this environment; outbound API calls failed with a connection error. Do not interpret the deterministic outputs as proof that a live model always follows the prompts.
