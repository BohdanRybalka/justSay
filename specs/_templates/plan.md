# Spec NNN — <Title>

> **Status:** planned
> **Target release:** vX.Y.Z

## Context / Why

<!--
Business justification. What prompted this (user request, backlog item, QA/reviewer
finding, broken assumption). The intended outcome in one sentence. Keep this honest:
if the change is cosmetic or speculative, say so here.
-->

## Acceptance Criteria

<!--
Verifiable, testable bullets — what "done" means. Each one should be checkable by
the reviewer against the actual code, not against a claim.
-->

## Recommended approach

<!--
Single chosen direction described in prose. Alternatives that were considered and
rejected belong in "Cuts deferred" below — do NOT present multiple options as if
equivalent. Be concrete about names, signatures, and file paths so the reviewer can
sanity-check without re-imagining. Link an ADR here if one was written
(docs/adr/NNN-title.md).
-->

## Files to modify

<!--
One bullet per file or new file, with a one-line purpose. Example:
- `backend/app/stt/local.py` — add `audio_duration` kwarg + four new transcribe kwargs
- `backend/tests/test_stt.py` — assert each new kwarg via `mock.call_args.kwargs`
-->

## Risks

<!--
Residual risk after the chosen approach + chosen cuts. What we accept and why.
Don't list "could break the build" — that's true for every change. List the
specific failure modes that survive merge.
-->

## Cuts deferred to a future spec

<!--
Table: what / why / trigger to reinstate. Non-empty unless the task is genuinely
one-dimensional — premature universality is a smell.
-->

## Deviations

<!--
Appended by the implementer if the actual change diverges from the plan above.
One bullet per deviation, with the reason. Left empty until implementation happens.
-->

## Review history

<!--
Appended after each reviewer pass. One block per iteration:

### Review — iteration 1 (yyyy-mm-dd)
- reviewer verdict: approved | needs-work
- Triage (architect-coordinator, only if needs-work): plan-flaw fixes applied /
  issues routed to implementer
- Issues for implementer: <numbered list, or "none">
-->
