"""Ukrainian filler words: speech habits flagged in the favourite words, never removed.

Separate from the stop-word list, which removes words from the statistics. Written as
they are shown; a filler of several words is matched on the token stream as one word.
"""

from __future__ import annotations

FILLERS_UK: frozenset[str] = frozenset(
    {
        "ну", "типу", "коротше", "значить", "власне", "от", "ось", "ніби",
        "як би", "взагалі", "так би мовити", "в принципі",
    }
)
