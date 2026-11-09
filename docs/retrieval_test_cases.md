# Manual retrieval test cases

Run the UI with `streamlit run scripts/retrieval_playground.py`. For each case,
select the indicated collection and retriever. These cases check retrieval and
provenance; they do not grade legal advice or expect one exact rank order.
For baseline dense, BM25, and hybrid comparisons, turn off query expansion and
reranking. Turn both on specifically for case 11 and the toggle checks below.

| # | Collection | Retriever | Query | What to check |
|---|---|---|---|---|
| 1 | Rent / tenancy | Hybrid | My landlord is refusing to return my security deposit | Results come from `rent_law`; source, section, and page metadata appear when present. |
| 2 | Rent / tenancy | BM25 only | Model Tenancy Act section 12 tenancy agreement | Exact statutory terms find a chunk from the Model Tenancy Act PDF. |
| 3 | Rent / tenancy | Dense only | What document must a landlord and tenant keep for a tenancy? | A semantically related tenancy-agreement chunk appears even without an exact query match. |
| 4 | Rent / tenancy | Hybrid | Registration Act, 1908 registration of lease | Results cite the Registration Act source in the rent collection. |
| 5 | Consumer law | BM25 only | Consumer Protection Act 2019 section 2 complaint definition | A definition chunk from the Consumer Protection Act is retrieved. |
| 6 | Consumer law | Hybrid | Where can a consumer file a complaint against a seller? | Consumer law sources are returned with document and statutory metadata. |
| 7 | Consumer law | Dense only | What remedies exist when a purchased product is defective? | Product-defect or consumer-remedy chunks appear from the consumer collection. |
| 8 | Wages / labour | BM25 only | Code on Wages 2019 section 2 wages | A wages-code source is returned and its statutory reference is visible. |
| 9 | Wages / labour | Hybrid | Are minimum wages applicable to all employees? | Results come from the wages collection and retain source provenance. |
| 10 | Wages / labour | Hybrid | Code on Social Security 2020 contribution obligations | The Social Security source appears under `wages_law`, despite its PDF being stored in the consumer folder. |
| 11 | Rent / tenancy | Hybrid + multi-query | What does Section 12 of the Model Tenancy Act, 2021 require? | Expanded retrieval returns unique chunk IDs and keeps the section/year in the original query used for reranking. |
| 12 | Each collection | Hybrid | `section 1` | Every result stays within the selected collection; inspect chunk IDs and citation metadata for duplicates. |

## Repeatability checks

- Run cases 1, 5, and 8 twice. The same query and settings should return the
  same collection scope and generally similar top results.
- Compare dense, BM25, and hybrid for cases 1, 6, and 9. Hybrid should be able
  to surface results from both retrieval signals; rank order may differ.
- Toggle multi-query and reranking independently. With reranking on, results
  should be capped at the selected display count and show `reranker_score`.
- Confirm every displayed result has a `chunk_id`, source path or filename,
  and the section/page fields available for that source.
