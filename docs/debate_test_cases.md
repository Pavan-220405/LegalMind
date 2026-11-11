# Manual debater test case

Use this case in the app started with `streamlit run frontend/app.py`. Select
**Debate practice**. Ingest the legal PDFs first if the rent collection is empty.
This scenario checks intake, rent routing, source-grounded opposition, turn
continuity, and coaching. Exact wording and source ranking can vary.

## Case: Tenant seeking return of a security deposit

**Setup**

- Area: Rent and tenancy
- Issue: A landlord has kept the tenant's security deposit after the tenancy
  ended.
- User's position: The deposit should be returned because the tenant says they
  paid rent and left the property in good condition.
- Jurisdiction: India; do not assume a particular state or local tenancy law.

**Conversation to enter**

1. `I rented a flat and my landlord kept my security deposit after I moved out. I paid the rent and left the flat in good condition. I want to argue that the deposit should be returned.`
2. Read the setup summary. If it accurately reflects the issue and position,
   reply: `Yes, that's right. Start the debate.` If not, correct the summary and
   confirm the corrected version.
3. After the debater responds, reply: `The landlord says there were repair costs, but gave me no itemized list or receipts. Why should the deposit still be withheld?`
4. Then reply: `Coach me on my argument.`

## Expected checks

- The first assistant response asks a concise clarifying question or confirms
  the setup; it does not begin the debate or retrieve sources yet.
- The assistant uses the rent and tenancy collection after the user confirms
  the setup. Its opposing argument responds to the user's position and cites
  retrieved excerpts with labels such as `[S1]`.
- Each legal claim attached to a citation is supported by that cited excerpt.
  If the retrieved material does not support a claim, the debater says so or
  frames the point as hypothetical instead of inventing a rule.
- The debater invites a response and addresses the user's next point in the
  same practice, rather than restarting intake.
- The coaching response steps out of the opposing role, identifies a persuasive
  point and a gap needing support, and gives one concrete improvement. Any legal
  comments remain grounded in the retrieved excerpts.
- If retrieval returns no relevant excerpts, the assistant says it cannot make
  a law-based opposing argument yet; it does not fabricate citations or law.

This is a behavior check, not a legal assessment. Do not require a specific
legal outcome or a fixed citation rank; check the actual excerpt shown for each
citation.