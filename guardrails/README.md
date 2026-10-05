# Output Guardrails

Groundedness, memory-leakage, and abstention checks are enabled by default. They reuse `eval.answer_eval.score_answer` with a live Groq judge at runtime; a judge/provider error fails closed to the existing no-information abstention response.

```text
RAG_GUARDRAIL_GROUNDEDNESS_ENABLED=1
RAG_GUARDRAIL_MEMORY_LEAKAGE_ENABLED=1
RAG_GUARDRAIL_ABSTENTION_ENABLED=1
RAG_GUARDRAIL_GROUNDEDNESS_ACTION=abstain
RAG_GUARDRAIL_MIN_FAITHFULNESS=1.0
RAG_GUARDRAIL_MIN_CITATION_PRECISION=1.0
RAG_GUARDRAIL_MIN_CITATION_RECALL=1.0
```

Set `RAG_GUARDRAIL_GROUNDEDNESS_ACTION=regenerate_once` to permit one correction pass before abstention. Every response is buffered until its judge result is available; this intentionally increases time-to-first-token and adds at least one Groq judge call to a grounded answer. Failed/unsupported outputs never stream candidate tokens. The existing `token`, `answer`, and `done` events are reused without schema changes.

For offline CI tests, inject a stub validator or disable the two scoring checks in a test fixture. Do not disable them for production requests. Pattern/config unit tests remain deterministic and require no model calls.

The judge is not a formal proof system: it can miss entailment errors or misread citations. Citation IDs are checked against retrieved passages, but confidence and false-positive/negative rates depend on the judge and the document's citation format.

## Structural and Content Checks

`RAG_GUARDRAIL_OUTPUT_SCHEMA_ENABLED=1` validates rewritten queries, emitted RAG trace entries, and recognized answer citation references. A malformed rewrite is retried once, then the previous query is retained. `RAG_GUARDRAIL_HARMFUL_CONTENT_ENABLED=1` blocks a small set of high-confidence harmful operational instructions on both retrieved-answer and direct-answer routes. It is enabled by default and has no added dependency; the implementation uses standard-library pattern checks. Pattern hits fail closed to the existing no-information response and are recorded in the guardrail trace. Direct streaming output is buffered until the check completes, so the first token is delayed; the SSE event names and payloads are unchanged.

The harmful-content patterns are deliberately narrow for technical/business RAG. They can miss unfamiliar or indirect phrasing, and a benign answer discussing harmful conduct could be blocked if it matches a pattern. Review `harmful_content` trace categories when tuning this behavior.

Sensitive-data masking is opt-in and applies only to values copied into persistence, not to the live answer or SSE response:

```text
RAG_GUARDRAIL_SENSITIVE_CONTENT_ENABLED=1
```

The masker uses Python standard-library patterns for email, US phone, SSN, and Luhn-valid payment-card numbers. It has no external dependency, may miss other identifiers/formats, and can occasionally mask a number that resembles a phone number. Confirm the resulting persisted data policy before enabling it broadly.

Run the focused Phase 3 tests with `python -m pytest -q tests/test_guardrails_phase3.py tests/test_guardrails_output.py tests/test_api.py`. For a manual harmful-output check, enable `RAG_GUARDRAIL_HARMFUL_CONTENT_ENABLED=1`, ask a question routed to direct answering that elicits one of the narrow detector patterns, and confirm that only the existing abstention response is emitted and the trace reports `harmful_content: blocked`. A benign response that does not match a pattern passes unchanged.