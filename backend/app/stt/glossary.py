"""The stored glossary turned into what one engine family may receive (ADR 081).

The stored string is never truncated. `glossary_text` reads a blank one as no
glossary; `whisper_glossary` fits it into a budget of characters, never answering
nothing for a glossary with text in it and never rewriting one that fits. The budget
counts characters for a token window, so a dense script can still overflow it.
`dictation_prompt` adds a sentence in the dictation language, which `without_prompt_echo`
takes back out of a transcript. `glossary_summary` is a glossary's only shape in a log line.
"""

import re

from app.stt.languages import LANGUAGE_SENTENCES, UNSPACED_LANGUAGES

WHISPER_PROMPT_CHAR_BUDGET = 487

_TERM_SEPARATORS = re.compile(r"[,;\n、，；]")
_TERM_JOIN = ", "
_WORD_JOIN = " "
_SENTENCE_JOIN = "\n"


def glossary_text(raw: str) -> str | None:
    """The stored glossary stripped, or `None` when nothing is left.

    `None` rather than `""`, which every caller must treat as no glossary.
    """
    return raw.strip() or None


def _fitting(pieces: list[str], join: str, budget: int) -> list[str]:
    """The pieces that fit ``budget`` characters once joined, in the order given.

    A piece that would cross the budget is skipped and the walk continues, so
    one over-long piece keeps the shorter ones behind it.
    """
    kept: list[str] = []
    length = 0
    for piece in pieces:
        addition = len(piece) + (len(join) if kept else 0)
        if length + addition <= budget:
            kept.append(piece)
            length += addition
    return kept


def whisper_glossary(
    raw: str, budget: int = WHISPER_PROMPT_CHAR_BUDGET
) -> tuple[str | None, int]:
    """What a Whisper-family engine receives, and how many characters were cut.

    A glossary within ``budget`` characters is passed through unchanged. Otherwise whole
    terms, then whole words, then characters; each pass runs only when the one before it
    kept nothing. A non-empty stored glossary always yields a non-empty value within it.
    """
    text = glossary_text(raw)
    if text is None:
        return None, 0
    if len(text) <= budget:
        return text, 0

    terms = [term for term in (part.strip() for part in _TERM_SEPARATORS.split(text)) if term]
    kept_terms = _fitting(terms, _TERM_JOIN, budget)
    if kept_terms:
        sent = _TERM_JOIN.join(kept_terms)
    else:
        kept_words = _fitting(text.split(), _WORD_JOIN, budget)
        sent = _WORD_JOIN.join(kept_words) if kept_words else text[:budget]
    return sent, len(text) - len(sent)


def dictation_prompt(raw: str, language: str) -> tuple[str | None, int]:
    """The glossary, then one sentence in ``language``, within `WHISPER_PROMPT_CHAR_BUDGET`.

    The sentence comes last and whole; the glossary is trimmed to make room for it. A
    language with no sentence gets the glossary alone. Returns the prompt and the cut.
    """
    sentence = LANGUAGE_SENTENCES.get(language)
    if sentence is None:
        return whisper_glossary(raw)
    room = WHISPER_PROMPT_CHAR_BUDGET - len(sentence) - len(_SENTENCE_JOIN)
    glossary, cut = whisper_glossary(raw, room)
    return (sentence if glossary is None else f"{glossary}{_SENTENCE_JOIN}{sentence}"), cut


def without_prompt_echo(text: str, language: str) -> str:
    """``text`` without the dictation sentence of ``language`` an engine wrote at its start or end.

    The sentence is matched as whole words, ignoring case and the punctuation and spaces around
    it; one anywhere else stays, and the rest of ``text`` is left exactly as it was.
    """
    sentence = LANGUAGE_SENTENCES.get(language)
    if sentence is None:
        return text
    unspaced = language in UNSPACED_LANGUAGES
    gap = r"\W*" if unspaced else r"\W+"
    echo = gap.join(re.escape(word) for word in re.findall(r"\w+", sentence))
    before, after = ("", "") if unspaced else (r"(?<!\w)", r"(?!\w)")
    text = re.sub(rf"\A\W*{echo}{after}\W*", "", text, flags=re.IGNORECASE)
    return re.sub(rf"\s*{before}{echo}\W*\Z", "", text, flags=re.IGNORECASE)


def glossary_summary(text: str | None, cut: int = 0) -> str:
    """A glossary's length and how many characters were cut, never its content.

    Reads `none` when nothing is sent, and gains ` -Ncut` only when the budget
    shortened the stored value.
    """
    summary = f"{len(text)}chars" if text else "none"
    return f"{summary} -{cut}cut" if cut else summary
