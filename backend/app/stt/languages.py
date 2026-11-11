"""Language names shared by the STT providers, the sentence that ends a dictation prompt, and
the languages written without spaces between words.
"""

LANGUAGE_NAMES = {
    "uk": "Ukrainian",
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pl": "Polish",
    "ja": "Japanese",
    "zh": "Chinese",
}

LANGUAGE_SENTENCES = {
    "uk": "Це запис українською мовою.",
    "en": "This is a recording in English.",
    "de": "Das ist eine Aufnahme auf Deutsch.",
    "fr": "Ceci est un enregistrement en français.",
    "es": "Esta es una grabación en español.",
    "pl": "To jest nagranie w języku polskim.",
    "ja": "これは日本語の録音です。",
    "zh": "这是一段中文录音。",
}

UNSPACED_LANGUAGES = frozenset({"ja", "zh"})
