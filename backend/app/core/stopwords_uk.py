"""Ukrainian stop-word list for the word-frequency feature.

Static module: no external download, frozenset for O(1) membership lookup.
Words are lowercased; the tokeniser also lowercases before lookup.
Curated from the typical Ukrainian stop-word set (function words, pronouns,
auxiliaries, common particles). Kept conservative — words that may carry
content meaning (verbs, nouns) are NOT included.
"""

from __future__ import annotations

STOPWORDS_UK: frozenset[str] = frozenset(
    {
        # articles / particles
        "а", "ай", "але", "б", "би", "бо", "будь", "більш", "більше",
        "в", "ва", "вам", "вами", "вас", "ваш", "ваша", "ваше", "ваші",
        "ввесь", "вгору", "вже", "взагалі", "вниз", "він", "вона", "воно", "вони",
        "все", "всі", "всім", "всіх", "всього", "всьому", "ви", "від",
        # prepositions
        "до", "для", "за", "з", "із", "зі", "над", "на", "не", "ні", "ну",
        "о", "об", "од", "ось", "от", "перед", "під", "при", "про",
        # conjunctions
        "та", "так", "також", "тільки", "те", "тебе", "теж", "тих", "тобі",
        "тобою", "той", "тому", "ту", "тут", "у", "хоч", "хоча", "чи", "що",
        "щоб", "як", "якби", "якщо",
        # pronouns
        "ми", "мене", "мені", "мною", "мій", "моя", "моє", "мої", "мого",
        "хто", "коли", "кого", "кому", "куди", "лише",
        "є", "був", "була", "було", "були",
        # common short tokens
        "вже", "ще", "це", "цей", "ця", "ці", "цього", "цьому",
        "той", "та", "те", "ті", "того", "тому",
        "сам", "сама", "само", "самі", "себе", "собі", "собою",
        "свій", "своя", "своє", "свої", "свого", "своєму",
        # adverbs / particles
        "тут", "там", "куди", "звідки", "колись", "ніколи", "завжди",
        "тепер", "вчора", "сьогодні", "завтра", "зараз",
        "будь-який", "інший", "інша", "інше", "інші",
    }
)
