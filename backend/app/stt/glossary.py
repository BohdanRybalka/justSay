"""The stored glossary turned into what one engine family may receive (ADR 081).

The stored string is never truncated. `glossary_text` hands it whole to a model
that reads natural language, and `whisper_glossary` assembles whole terms up to
`WHISPER_PROMPT_CHAR_BUDGET` for a Whisper decoder's prompt window.
`glossary_summary` is the only shape a glossary may take in a log line.
"""

import re

WHISPER_PROMPT_CHAR_BUDGET = 487

_TERM_SEPARATORS = re.compile(r"[,\n]")
_TERM_JOIN = ", "


def glossary_text(raw: str) -> str | None:
    """The stored glossary stripped, or `None` when nothing is left.

    An empty string is never sent: Groq answers `400` to `prompt=""`, and a
    Whisper decoder conditions on nothing either way.
    """
    return raw.strip() or None


def whisper_glossary(raw: str) -> tuple[str | None, int]:
    """Whole terms within the budget, and how many terms were dropped.

    Terms split on commas and newlines keep the user's order; the first term
    that would cross `WHISPER_PROMPT_CHAR_BUDGET` is dropped together with
    every term after it. A first term already over the budget yields `None`
    rather than a fragment of it.
    """
    text = glossary_text(raw)
    if text is None:
        return None, 0

    terms = [term for term in (part.strip() for part in _TERM_SEPARATORS.split(text)) if term]
    kept: list[str] = []
    length = 0
    for index, term in enumerate(terms):
        addition = len(term) + (len(_TERM_JOIN) if kept else 0)
        if length + addition > WHISPER_PROMPT_CHAR_BUDGET:
            return (_TERM_JOIN.join(kept) or None), len(terms) - index
        kept.append(term)
        length += addition

    return (_TERM_JOIN.join(kept) or None), 0


def glossary_summary(text: str | None, dropped: int) -> str:
    """A glossary's length and how many terms were dropped, never its content.

    Reads `none` when nothing is sent, and gains ` -Nterms` only when the
    budget dropped some.
    """
    summary = f"{len(text)}chars" if text else "none"
    return f"{summary} -{dropped}terms" if dropped else summary
