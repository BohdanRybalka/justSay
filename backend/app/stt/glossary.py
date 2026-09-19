"""The stored glossary turned into what one engine family may receive (ADR 081).

The stored string is never truncated. `glossary_text` hands it whole to a model
that reads natural language, and `whisper_glossary` fits it into
`WHISPER_PROMPT_CHAR_BUDGET` characters, never answering nothing for a glossary
that has text in it and never rewriting one that already fits. The budget is a
character count standing in for a token window, so a dense script can still
overflow it. `glossary_summary` is the only shape a glossary may take in a log
line.
"""

import re

WHISPER_PROMPT_CHAR_BUDGET = 487

_TERM_SEPARATORS = re.compile(r"[,;\n、，；]")
_TERM_JOIN = ", "
_WORD_JOIN = " "


def glossary_text(raw: str) -> str | None:
    """The stored glossary stripped, or `None` when nothing is left.

    `None` rather than `""`, which every caller must treat as no glossary.
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
    """What a Whisper-family engine receives, and how many characters were cut.

    A glossary already within the budget is passed through unchanged. Otherwise
    whole terms, then whole words, then characters; each pass runs only when the
    one before it kept nothing. A non-empty stored glossary always yields a
    non-empty value of at most `WHISPER_PROMPT_CHAR_BUDGET` characters.
    """
    text = glossary_text(raw)
    if text is None:
        return None, 0
    if len(text) <= WHISPER_PROMPT_CHAR_BUDGET:
        return text, 0

    terms = [term for term in (part.strip() for part in _TERM_SEPARATORS.split(text)) if term]
    kept_terms = _fitting(terms, _TERM_JOIN)
    if kept_terms:
        sent = _TERM_JOIN.join(kept_terms)
    else:
        kept_words = _fitting(text.split(), _WORD_JOIN)
        sent = (
            _WORD_JOIN.join(kept_words)
            if kept_words
            else text[:WHISPER_PROMPT_CHAR_BUDGET]
        )
    return sent, len(text) - len(sent)


def glossary_summary(text: str | None, cut: int = 0) -> str:
    """A glossary's length and how many characters were cut, never its content.

    Reads `none` when nothing is sent, and gains ` -Ncut` only when the budget
    shortened the stored value.
    """
    summary = f"{len(text)}chars" if text else "none"
    return f"{summary} -{cut}cut" if cut else summary
