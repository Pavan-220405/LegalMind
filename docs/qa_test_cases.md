# AI Lawyer test cases: manual and RAGAS

This suite has 15 medium-to-hard scenarios for the supported Indian-law sources
in this repository: tenancy/property, consumer protection, and wages/social
security. Use each scenario in two ways:

1. **Normal testing:** enter the prompt in a fresh conversation unless the case
   says it is multi-turn. Check routing, source provenance, answer behavior, and
   the listed pass criteria. The answer need not use the wording in this file.
2. **RAGAS evaluation:** capture the actual retrieved chunks and final answer,
   then compare them with the reference answer and expected context below. The
   expected contexts are relevance targets, not text that must match verbatim.

These are evaluation fixtures, not legal advice or an exhaustive statement of
the law. A response should qualify jurisdiction- or fact-dependent conclusions
and should not invent deadlines, remedies, sections, or citations. Check legal
references against the source PDFs before treating a score as a legal-quality
judgment.

## Shared case set

| ID / level | Prompt | Normal test: expected behavior and pass criteria | RAGAS reference answer | Expected context (reference context) |
|---|---|---|---|---|
| N01 / Medium | **Tenancy:** “I moved out three weeks ago. My landlord says they can keep my whole deposit because the walls need repainting, but the agreement only mentions deductions for damage. What should I do?” | Routes to tenancy. Asks for location, agreement terms, handover/inspection evidence, and itemized deductions if needed. Gives practical evidence-preservation steps; avoids guaranteeing a refund or asserting a universal deposit deadline. | The outcome depends on the agreement, applicable state rules, and evidence of damage beyond ordinary wear. Ask for an itemized deduction and preserve the agreement, payment proof, move-out photos, and messages. Verify local remedies before stating a deadline. | Model Tenancy Act, 2021 material on tenancy agreements, security deposits, and landlord/tenant obligations; retrieve only from `rent_law`. |
| N02 / Medium | **Tenancy:** “My landlord and I agreed to rent by WhatsApp and exchanged the deposit, but never signed a paper agreement. Is the tenancy automatically invalid?” | Routes to tenancy and distinguishes evidence of an agreement from statutory form/registration requirements. Does not declare the tenancy automatically valid or void; asks state, term, and relevant messages/payment records. | The available sources should be checked for the applicable agreement and registration requirements. The facts that there was an oral or message-based arrangement and deposit payment may be relevant evidence, but they do not alone settle validity or compliance. State and term matter. | Model Tenancy Act, 2021 provisions about written tenancy agreements; Registration Act, 1908 provisions relevant to registration of leases. |
| N03 / Hard | **Tenancy:** “My lease is for 14 months. The landlord says notarization is enough, so registration is unnecessary. Is that right?” | Retrieves both tenancy and registration material in `rent_law`. Identifies the term as material, explains that notarization and registration are different concepts, and tells the user to confirm current applicable law/state rules. No unsupported categorical answer. | Registration requirements for a lease can depend on the lease term and applicable law. Notarization is not itself proof that a document has been registered. Check the Registration Act source and relevant state rules before deciding whether this lease needs registration. | Registration Act, 1908 lease-registration provisions, alongside Model Tenancy Act, 2021 agreement provisions. |
| N04 / Hard | **Tenancy, multi-turn:** Turn 1: “My landlord changed the locks after I complained about repairs.” Turn 2: “I am in [state]; I have rent receipts and the messages. Can I break in and take my belongings?” | Turn 1 routes to tenancy and treats lockout/access as urgent and fact-sensitive. Turn 2 preserves prior facts, discourages breaking in or confrontation, advises documenting events and seeking prompt local legal/authority help, and avoids inventing a universal remedy. | A lockout and access to belongings require prompt, jurisdiction-specific advice. Preserve rent receipts and messages, document the lock change, and seek help from a local lawyer or appropriate authority. Do not use force or create a safety risk. | Tenancy agreement and landlord/tenant obligation material in the Model Tenancy Act, 2021. Expected behavior also includes retaining multi-turn facts; retrieved law alone may not answer the procedural remedy. |
| N05 / Medium | **Consumer:** “A shop sold me a phone for my small business. It stopped working after two weeks. Can I file as a consumer?” | Routes to consumer. Recognizes that use/purpose may affect consumer status; asks whether it was bought for livelihood/self-employment or a larger commercial purpose and requests invoice/warranty/complaint records. Does not decide status solely from “small business.” | Consumer status can depend on the purpose for which goods were bought and the statutory definition and exceptions. Clarify the actual use and circumstances before concluding. Preserve the invoice, warranty, fault evidence, and seller communications. | Consumer Protection Act, 2019 definition of “consumer” and its commercial-purpose/livelihood qualification. |
| N06 / Medium | **Consumer:** “The washing machine is still under warranty, but the service center has visited four times and it still leaks. What remedy can I ask for?” | Routes to consumer, identifies defect/service evidence, and describes possible statutory relief only at a high level subject to forum and facts. Does not promise replacement/refund or assume warranty resolution blocks a complaint. | Repeated unsuccessful repairs may be relevant to a consumer complaint about a defective product or deficient service. Preserve the invoice, warranty, repair/job sheets, photos, and written complaints. The available statutory remedies and appropriate forum depend on the facts and applicable rules. | Consumer Protection Act, 2019 provisions on consumer complaints and relief for defective goods/deficiency; relevant consumer regulations/rules in the corpus. |
| N07 / Hard | **Consumer:** “The seller offered a replacement, but it is a refurbished unit. I asked for a refund and they say their ‘no refund’ sign ends the matter. What should I document?” | Routes to consumer. Separates seller policy from statutory rights without asserting that the policy is automatically void; asks about purchase terms, defect, offer, and communications; gives an evidence checklist. | A store policy does not by itself resolve whether statutory relief is available. The defect, disclosures, purchase terms, and seller response matter. Keep the receipt, product/serial details, listing or policy, repair/return communications, and evidence of the defect; verify the applicable remedy and forum. | Consumer Protection Act, 2019 provisions on complaints, product defect, unfair trade practices, and relief, as relevant to retrieved text. |
| N08 / Hard | **Consumer:** “I first noticed the defect 26 months ago, but the seller kept promising to fix it. Is it too late to complain?” | Routes to consumer. Notices the explicit date and limitation issue; mentions the section 69 two-year general period and possible condonation for sufficient cause, but labels the calculation preliminary and asks when the cause of action arose and what evidence supports the promises. | Section 69 of the Consumer Protection Act, 2019 generally provides a two-year period from the cause of action, subject to possible condonation for sufficient cause. The stated 26 months is not enough to decide the deadline: identify the legally relevant cause-of-action date and obtain prompt legal advice. | Consumer Protection Act, 2019 section 69 limitation provision, including delay-condonation language. |
| N09 / Medium | **Consumer:** “The courier marked my laptop delivered, but I never received it. The marketplace and seller each blame the other. Who can I complain against?” | Routes to consumer and recognizes multiple actors and missing delivery facts. Requests order, tracking, payment, and complaint records; does not declare one party automatically liable or invent a forum threshold. | Responsibility depends on the transaction, seller/platform role, delivery records, and applicable law. Preserve the order confirmation, payment proof, tracking status, delivery confirmation, and messages with both seller and marketplace. The complaint target and forum should be checked against the facts and current rules. | Consumer Protection Act, 2019 complaint provisions and applicable consumer rules/regulations in the corpus; retrieval should stay in `consumer_law`. |
| N10 / Medium | **Wages:** “I work for a small private shop and receive less than the minimum wage rate I found online. Does the Code on Wages cover me?” | Routes to wages. Identifies location, job category, actual duties, wage period, and applicable rate/date as necessary. Does not assume the online rate is current or that coverage/rate is identical everywhere. | Coverage and the applicable minimum-wage rate require the worker’s location, category, duties, and the current applicable notification. The Code on Wages source is relevant, but the online figure should be verified with the responsible authority or a lawyer. | Code on Wages, 2019 provisions concerning minimum wages and applicable rates/notifications. |
| N11 / Hard | **Wages:** “My employer deducted ₹3,000 for a broken tool. I did not agree, and there was no inquiry. Can they deduct whatever they want?” | Routes to wages. Checks deduction, authorization, notice/process, wage records, and applicable rules; avoids giving a categorical ruling without facts. Advises preserving payslips and written explanation. | Wage deductions are governed by statutory limits and conditions; an employer cannot be assumed to have unlimited discretion. Whether this deduction is permitted depends on the statutory provision and facts, including the alleged loss, process, and records. Preserve payslips and request the basis in writing. | Code on Wages, 2019 provisions about deductions from wages and limits/conditions. |
| N12 / Medium | **Wages:** “My salary is 20 days late. HR says cash flow problems mean they can pay whenever they want. What records should I keep, and where do I start?” | Routes to wages. Gives a practical evidence checklist (contract/offer, attendance, payslips, bank statement, messages), points to prompt local labour-authority/legal help, and does not invent a limitation period or claim process. | Cash-flow problems do not by themselves establish that delayed wages are lawful. Keep the employment terms, attendance and wage records, bank statements, and written requests. The applicable payment deadline and complaint route should be checked for the worker’s situation and current law. | Code on Wages, 2019 provisions on payment of wages, wage period, and claims/complaint mechanisms as present in the retrieved text. |
| N13 / Hard | **Wages:** “Two employees do substantially the same work. One is paid less because the manager says women are ‘less available.’ Is that allowed?” | Routes to wages. Recognizes possible sex-based wage discrimination; asks for role/duties, pay records, and comparator evidence. Explains relevant statutory protection at a high level and recommends documenting facts; does not decide the claim. | The Code on Wages, 2019 contains a prohibition on gender-based discrimination in wages for the same work or work of a similar nature, subject to the statutory text and facts. Preserve job descriptions, duties, payslips, schedules, and communications and seek appropriate local advice. | Code on Wages, 2019 provision prohibiting gender discrimination in wages for same/similar work. |
| N14 / Hard | **Wages/social security:** “I deliver food through an app. The company calls me an independent partner and says social security laws can never apply. Is the label decisive?” | Routes to wages collection per current architecture. Uses the Code on Social Security, 2020 source if retrieved; distinguishes statutory classification/coverage from a contract label, asks state/platform/work details, and does not promise eligibility or benefits. | The contract label alone may not settle statutory classification or coverage. The Code on Social Security source addresses categories including gig/platform workers, but eligibility, schemes, and benefits depend on the law in force and the worker’s facts. Verify current implementation and applicable scheme details. | Code on Social Security, 2020 definitions/provisions on gig workers, platform workers, and social-security schemes. |
| N15 / Hard | **Clarification / scope:** “My landlord is also my employer and has withheld both my deposit and my salary. Can I file one case and recover both?” | Tests issue decomposition and routing. The assistant should clarify that tenancy and wage claims may involve different laws/processes, identify both domains or ask focused questions, avoid promising a single forum, and not drop either issue. If the current workflow can route only one domain, it should say so and ask which issue to address first. | A rental deposit dispute and withheld wages raise distinct legal issues that may be governed by different statutes and complaint processes. More facts and location are needed; do not assume one combined filing or forum. Preserve separate tenancy and employment records and seek local advice. | Relevant Model Tenancy Act, 2021 agreement/deposit material and Code on Wages, 2019 payment provisions. This is also a routing/scope test; one retrieved collection may not support both parts. |

## Running normal tests

- Start a new conversation for N01–N03, N05–N14. Use the two turns specified
  for N04. Use N15 to check clarification or the documented one-domain scope.
- For domain cases, verify the expected collection (`rent_law`, `consumer_law`,
  or `wages_law`), source title/path, section and page metadata when available,
  and that citations refer to returned source IDs. No fabricated section or
  page reference is a pass.
- Check the final response as well as retrieval: it should answer the question,
  name material unknowns, preserve user-provided facts, and avoid overconfident
  legal conclusions. For N08, inspect the limitation warning. For N04, test
  urgency and conversation history.
- Record the app/model version, retrieval settings, selected collection,
  retrieved chunk IDs, final answer, and pass/fail notes for each run.

## Preparing a RAGAS evaluation

Create one evaluation row per case, retaining these fields (RAGAS single-turn
sample names in parentheses):

- `case_id` (local tracking field)
- `user_input` (prompt; for N04, include the previous turn in conversation
  context and evaluate the second turn)
- `retrieved_contexts` (actual chunk text returned by the run)
- `response` (actual final answer)
- `reference` (the RAGAS reference answer above)
- `reference_contexts` (the expected-context cell, expanded to actual source
  excerpts/chunks from the named project PDFs before evaluation)

Recommended metrics: **Faithfulness** and **AnswerRelevancy** on all answerable
cases; **ContextPrecision** and **ContextRecall** where expected contexts are
available; **FactualCorrectness** or answer correctness where the installed
RAGAS version supports it. Keep N04, N15, and other clarification/scope cases
as behavior checks; a low context-recall score does not by itself mean the
clarifying response is wrong. N08 is the best limitation-warning check. N14
needs a current-law review before using its reference as a gold legal answer.

Do not use the prose expected-context labels directly as `reference_contexts`:
resolve them to source excerpts from the named PDFs and retain the source,
section, and page metadata. Score results by case and inspect failures; avoid
relying on a single aggregate score across legal, clarification, and safety
behaviors.
