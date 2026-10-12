"""Phrases speech engines write over silence, and the cut that takes them out of a transcript.

The list holds the sign-offs and subtitle credits Whisper-family engines write where nobody spoke,
in English and Ukrainian. A phrase matches as whole words in any case with any punctuation between
them; a square-bracketed label such as ``[BLANK_AUDIO]`` and a music note always match. What is
left of the text keeps only the words and the punctuation that sits on them.
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
)

_WORD = re.compile(r"\w+")
_SOUND_LABEL = r"\[[^\]]*\]|[♪♫]+"


def _phrase_pattern(phrase: str) -> str:
    return r"\W+".join(re.escape(word) for word in _WORD.findall(phrase))


_INVENTED = re.compile(
    rf"(?:(?<!\w)(?:{'|'.join(_phrase_pattern(p) for p in INVENTED_PHRASES)})(?!\w)"
    rf"|{_SOUND_LABEL})",
    re.IGNORECASE,
)


def without_invented_phrases(text: str) -> str:
    """``text`` with every listed phrase and sound label cut out and the spaces closed up.

    Marks left standing alone where one was cut ("(", "...") go with it. Text that holds none
    is returned exactly as it came.
    """
    cleaned, cut = _INVENTED.subn(" ", text)
    if not cut:
        return text
    return " ".join(token for token in cleaned.split() if _WORD.search(token))
