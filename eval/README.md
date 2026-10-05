# Evaluation Modes

Run the default STUB mode on every commit and in CI. It is deterministic, makes no provider calls, and is only a harness check; its scores are not judge-quality measurements.

Run LIVE calibration deliberately and periodically, after preparing 20-30 reviewed cases and completing the matching 1-5 human ratings. Each repeat calls the Groq judge once per case, so the default three repeats estimate `case_count x 3` calls before retries. The CLI prints this estimate and requires interactive confirmation unless `--yes` is supplied explicitly. Groq 429 responses receive bounded exponential backoff; exhausted retries fail visibly.

```text
./.venv/bin/python -m eval.production.judge_calibration \
  --cases eval/data/calibration_cases.local.json \
  --ratings eval/data/human_ratings.local.json \
  --live --runs 3
```

Use `--yes` only for intentional automation. The answer-evaluation CLI is also STUB by default; use `--live` to make one judge call per golden case, with the same confirmation guard. Never put private calibration inputs or reports under tracked fixture files.

## Deterministic retrieval evaluation

`python -m eval.retrieval_eval` computes precision@k, recall@k, MRR, and nDCG@k independently for vector candidates and post-rerank passages, plus paired rerank lift. These metrics use no LLM judge and make no network calls. Unanswerable cases are excluded from retrieval-quality means and remain available to the answer/abstention evaluator.

```text
./.venv/bin/python -m eval.retrieval_eval
./.venv/bin/python -m eval.retrieval_eval \
  --dataset eval/data/retrieval_cases_v1.json \
  --results path/to/ranked_passages.json \
  --k 5 --json-out eval/reports/retrieval_candidate.json
```

The versioned dataset is `data/retrieval_cases_v1.json`; it contains a small synthetic corpus and 12 manually authored starter cases spanning single-fact, multi-part, comparison, global-summary, follow-up, and unanswerable questions. This is a reproducible evaluator fixture, not a measurement of production quality. `data/retrieval_results_v1.example.json` contains hand-authored rankings solely to exercise the CLI offline; it was not captured from a live RAG run.

Case schema:

- `id`, `type`, `question`, optional `history` and `expected_standalone_question`
- `expected_route` (`local` or `broad`) and `answerable`
- `expected_answer` for later answer-quality scoring
- `evidence`: zero or more `{filename, page, anchor}` labels

Ranked results use one record per `case_id`, with `vector_candidates` and `post_rerank` arrays. Each candidate carries `filename`, `page`, and `text` (or `page_content`); any passage/chunk ID is informational and ignored by matching. A gold evidence anchor is matched after Unicode/case/whitespace/punctuation normalization, scoped to the same source filename and page. Thus labels do not depend on splitter-specific chunk IDs and remain valid when the same answer-bearing span appears in newly chunked passage text. Multi-part cases list multiple anchors.

For an answerable case, precision@k counts matching returned passages over `min(k, returned count)`, recall@k counts distinct evidence anchors found over all gold anchors, MRR is the reciprocal rank of the first matching passage within k, and nDCG@k uses binary passage relevance. Rerank lift is post-rerank minus vector-candidate score, paired per case and then averaged. Results are also summarized by question type and expected route.

The fixed corpus is suitable for deterministic evaluation and label-review practice. It does not itself run a live `ask_stream()` or measure the production embedder/reranker; live ranked outputs must be captured as a results artifact in a later approved phase. Keep corpus facts, answer anchors, and expected answers reviewed together when editing a case.