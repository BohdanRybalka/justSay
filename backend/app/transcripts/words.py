"""Word frequency over stored dictations.

Derived from ``entries`` on demand and cached on ``history.derived_generation_locked``;
tokenisation runs in Python, outside the store lock. Files and meetings carry other
voices, so only rows whose ``source`` is ``dictation`` count. Both stop-word lists always
apply to the favourite words, because real dictations code-switch (ADR 016); filler words
are flagged instead, and stay even where a stop-word list holds them.
"""

from __future__ import annotations

import re
import threading
from collections import Counter
from collections.abc import Iterator
from typing import Literal, NamedTuple

from pydantic import BaseModel

from app.transcripts import history
from app.transcripts.fillers_en import FILLERS_EN
from app.transcripts.fillers_uk import FILLERS_UK
from app.transcripts.stopwords_en import STOPWORDS_EN
from app.transcripts.stopwords_uk import STOPWORDS_UK

TOP_LIMIT_MAX = 500

STOPWORDS_ALL: frozenset[str] = STOPWORDS_UK | STOPWORDS_EN
FILLERS: frozenset[str] = FILLERS_UK | FILLERS_EN

WordFilter = Literal["all", "fillers"]

_TOKEN_RE = re.compile(r"[\wЀ-ӿ]+(?:['’][\wЀ-ӿ]+)*", re.UNICODE)

_DICTATIONS_SQL = (
    "SELECT cleaned_text, audio_duration_seconds FROM entries WHERE source = 'dictation'"
)


class DictationTokens(NamedTuple):
    generation: int
    counts: Counter[str]
    said: Counter[str]
    speaking_seconds: float


_tokens_cache: DictationTokens | None = None
_scan_lock = threading.Lock()


def tokenize(text: str) -> list[str]:
    """Lowercase + extract content tokens. Stop-words NOT applied here —
    callers apply ``STOPWORDS_ALL`` after, so tests can inspect the raw
    token stream."""
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


_PHRASES: dict[tuple[str, ...], str] = {
    tuple(tokenize(filler)): filler for filler in FILLERS if len(tokenize(filler)) > 1
}
_LONGEST_PHRASE = max(len(phrase) for phrase in _PHRASES)


def said_words(tokens: list[str]) -> Iterator[str]:
    """``tokens`` as words said: a filler of several words becomes one word, as listed,
    matched longest first, and its tokens are not yielded again."""
    index = 0
    while index < len(tokens):
        for length in range(_LONGEST_PHRASE, 1, -1):
            phrase = _PHRASES.get(tuple(tokens[index : index + length]))
            if phrase is not None:
                yield phrase
                index += length
                break
        else:
            yield tokens[index]
            index += 1


def dictation_tokens() -> DictationTokens:
    """Every token and every word said in every dictation, counted, with their speaking
    time, as of ``generation``.

    A miss reads and tokenises every dictation, so run it off the event loop. Readers
    that miss together wait for one scan rather than each running their own.
    """
    global _tokens_cache
    with _scan_lock:
        with history._lock:
            generation = history.derived_generation_locked()
            if _tokens_cache is not None and _tokens_cache.generation == generation:
                return _tokens_cache
            rows = history._ensure_conn_locked().execute(_DICTATIONS_SQL).fetchall()
        counts: Counter[str] = Counter()
        said: Counter[str] = Counter()
        for row in rows:
            row_tokens = tokenize(row["cleaned_text"])
            counts.update(row_tokens)
            said.update(said_words(row_tokens))
        speaking = sum(row["audio_duration_seconds"] or 0.0 for row in rows)
        tokens = DictationTokens(generation, counts, said, speaking)
        with history._lock:
            if history.derived_generation_locked() == generation:
                _tokens_cache = tokens
        return tokens


class WordCount(BaseModel):
    word: str
    count: int
    is_filler: bool


class FillerNote(BaseModel):
    """The most said filler, the speaking minutes between two of it (``None`` without
    speaking time), and how many of the ``top_size`` most said words are fillers."""

    word: str
    count: int
    minutes_between: float | None
    fillers_in_top: int
    top_size: int


class TopWordsResponse(BaseModel):
    items: list[WordCount]
    note: FillerNote | None


def _listed(word: str) -> bool:
    return word in FILLERS or (len(word) >= 2 and word not in STOPWORDS_ALL)


def top_words(limit: int = 50, word_filter: WordFilter = "all") -> TopWordsResponse:
    """The ``limit`` most said dictation words, or fillers only, with the filler note.

    Stop-words and one-letter tokens are left out unless they are fillers.
    """
    scan = dictation_tokens()
    ranked = Counter({word: count for word, count in scan.said.items() if _listed(word)})
    clamped_limit = max(1, min(int(limit), TOP_LIMIT_MAX))
    top = ranked.most_common(clamped_limit)
    fillers = [(word, count) for word, count in ranked.most_common() if word in FILLERS]
    shown = fillers[:clamped_limit] if word_filter == "fillers" else top
    note = None
    if fillers:
        word, count = fillers[0]
        note = FillerNote(
            word=word,
            count=count,
            minutes_between=round(scan.speaking_seconds / 60 / count, 2)
            if scan.speaking_seconds > 0
            else None,
            fillers_in_top=sum(1 for listed, _ in top if listed in FILLERS),
            top_size=len(top),
        )
    return TopWordsResponse(
        items=[WordCount(word=w, count=c, is_filler=w in FILLERS) for w, c in shown], note=note
    )
