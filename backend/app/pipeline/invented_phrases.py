"""Phrases speech engines write over silence, and the cut that takes them out of a transcript.

The list holds the sign-offs and subtitle credits Whisper-family engines write where nobody spoke,
in English and Ukrainian. A phrase matches as whole words in any case with any punctuation between
them; a square-bracketed label such as ``[BLANK_AUDIO]`` and a music note always match. Marks
touching a cut go with it, and a text left without a letter or a digit is empty.
"""

import re

INVENTED_PHRASES = (
    "thanks for watching",
    "thank you for watching",
    "subtitles by the amara.org community",
    "subtitles by steamteamextra",
    "hello everyone welcome to my channel",
    "дякую за перегляд",
    "дякуємо за перегляд",
    "субтитри зроблені спільнотою amara.org",
    "субтитри створено спільнотою amara.org",
    "субтитры создавал dimatorzok",
    "субтитры сделал dimatorzok",
)

_WORD = re.compile(r"\w+")
_ALNUM = re.compile(r"[^\W_]")
_MARK = r"[^\w\s\[\]]"
_SOUND_LABEL = r"\[[^\]]*\]|[♪♫]+"


def _phrase_pattern(phrase: str) -> str:
    return r"\W+".join(re.escape(word) for word in _WORD.findall(phrase))


_PHRASES = "|".join(_phrase_pattern(phrase) for phrase in INVENTED_PHRASES)
_LEADING_MARKS = rf"(?:(?<!{_MARK}){_MARK}*)?"
_INVENTED = re.compile(
    rf"{_LEADING_MARKS}(?:(?<!\w)(?:{_PHRASES})(?!\w)|{_SOUND_LABEL}){_MARK}*",
    re.IGNORECASE,
)


def without_invented_phrases(text: str) -> str:
    """``text`` with every listed phrase and sound label cut out, and the marks touching it.

    The spaces close up. A text left without a letter or a digit is empty; text that holds
    none of the phrases is returned exactly as it came.
    """
    cleaned, cut = _INVENTED.subn(" ", text)
    if not cut:
        return text
    cleaned = " ".join(cleaned.split())
    return cleaned if _ALNUM.search(cleaned) else ""
