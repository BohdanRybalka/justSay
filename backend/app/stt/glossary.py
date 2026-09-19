"""The stored glossary turned into what one engine family may receive (ADR 081).

The stored string is never truncated. `glossary_text` hands it whole to a model
that reads natural language, and `whisper_glossary` fits it into
`WHISPER_PROMPT_CHAR_BUDGET` for a Whisper decoder's prompt window, never
answering nothing for a glossary that has text in it. `glossary_summary` is the
only shape a glossary may take in a log line.
"""

import re

WHISPER_PROMPT_CHAR_BUDGET = 487

_TERM_SEPARATORS = re.compile(r"[,\n、，；]")
_TERM_JOIN = ", "
_WORD_JOIN = " "


def glossary_text(raw: str) -> str | None:
    """The stored glossary stripped, or `None` when nothing is left.

    An empty string is never sent: Groq answers `400` to `prompt=""`, and a
    Whisper decoder conditions on nothing either way.
    """
    return raw.strip() or None


def _fitting(pieces: list[str], join: str) -> list[str]:
    """The pieces that fit the budget once joined, in the order given.

    A piece that would cross `WHISPER_PROMPT_CHAR_BUDGET` is skipped and the
    walk continues, so one over-long piece keeps the shorter ones behind it.
    """
    kept: list[str] = []
    length = 0
    for piece in pieces:
        addition = len(piece) + (len(join) if kept else 0)
        if length + addition <= WHISPER_PROMPT_CHAR_BUDGET:
            kept.append(piece)
            length += addition
    return kept


def whisper_glossary(raw: str) -> tuple[str | None, int]:
    """Whole terms within the budget, and how many terms were dropped.

    Whole terms, then whole words, then characters; each pass runs only when the
    one before it kept nothing, and the character cut is reachable only inside a
    run the user wrote no boundary into. A non-empty stored glossary always
    yields a non-empty value of at most `WHISPER_PROMPT_CHAR_BUDGET` characters.
    """
    text = glossary_text(raw)
    if text is None:
        return None, 0

    terms = [term for term in (part.strip() for part in _TERM_SEPARATORS.split(text)) if term]
    kept_terms = _fitting(terms, _TERM_JOIN)
    dropped = len(terms) - len(kept_terms)
    if kept_terms:
        return _TERM_JOIN.join(kept_terms), dropped

    kept_words = _fitting(text.split(), _WORD_JOIN)
    if kept_words:
        return _WORD_JOIN.join(kept_words), dropped

    return text[:WHISPER_PROMPT_CHAR_BUDGET], dropped


def glossary_summary(text: str | None, dropped: int) -> str:
    """A glossary's length and how many terms were dropped, never its content.

    Reads `none` when nothing is sent, and gains ` -Nterms` only when the
    budget dropped some.
    """
    summary = f"{len(text)}chars" if text else "none"
    return f"{summary} -{dropped}terms" if dropped else summary
