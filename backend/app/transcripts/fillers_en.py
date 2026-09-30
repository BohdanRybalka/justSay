"""English filler words: speech habits flagged in the favourite words, never removed.

Separate from the stop-word list, which removes words from the statistics. Written as
they are shown; a filler of several words is matched on the token stream as one word.
"""

from __future__ import annotations

FILLERS_EN: frozenset[str] = frozenset(
    {
        "like", "I mean", "you know", "so", "basically", "actually", "just",
        "kind of", "sort of", "right", "well", "literally",
    }
)
