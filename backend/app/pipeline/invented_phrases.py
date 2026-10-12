"""Phrases speech engines write over silence, and the cut that takes them out of a transcript.

The list holds the sign-offs and subtitle credits Whisper-family engines are documented to write
where nobody spoke, plus the Ukrainian wording of the same. A phrase matches as whole words in any
case with any punctuation between them; a square-bracketed label such as ``[BLANK_AUDIO]`` and a
music note always match.
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
_SOUND_LABEL = r"\[[^\]]*\]|♪+"


def _phrase_pattern(phrase: str) -> str:
    return r"\W+".join(re.escape(word) for word in _WORD.findall(phrase))


_INVENTED = re.compile(
    rf"(?:(?<!\w)(?:{'|'.join(_phrase_pattern(p) for p in INVENTED_PHRASES)})(?!\w)"
    rf"|{_SOUND_LABEL})[^\w\s]*",
    re.IGNORECASE,
)


def without_invented_phrases(text: str) -> str:
    """``text`` with every listed phrase and sound label cut out and the spaces closed up.

    Text that holds none is returned exactly as it came.
    """
    cleaned, cut = _INVENTED.subn(" ", text)
    return " ".join(cleaned.split()) if cut else text
