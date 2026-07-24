"""English stop-word list for the word-frequency feature.

Static module: no external download, frozenset for O(1) membership lookup.
Words are lowercased; the tokeniser also lowercases before lookup.
Curated from the standard NLTK / sklearn English stop-word list.
"""

from __future__ import annotations

STOPWORDS_EN: frozenset[str] = frozenset(
    {
        "a", "about", "above", "after", "again", "against", "all", "am", "an",
        "and", "any", "are", "as", "at",
        "be", "because", "been", "before", "being", "below", "between", "both",
        "but", "by",
        "can", "could",
        "did", "do", "does", "doing", "don", "down", "during",
        "each",
        "few", "for", "from", "further",
        "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
        "him", "himself", "his", "how",
        "i", "if", "in", "into", "is", "it", "its", "itself",
        "just",
        "me", "more", "most", "my", "myself",
        "no", "nor", "not", "now",
        "of", "off", "on", "once", "only", "or", "other", "our", "ours",
        "ourselves", "out", "over", "own",
        "s", "same", "she", "should", "so", "some", "such",
        "t", "than", "that", "the", "their", "theirs", "them", "themselves",
        "then", "there", "these", "they", "this", "those", "through", "to",
        "too",
        "under", "until", "up",
        "very",
        "was", "we", "were", "what", "when", "where", "which", "while", "who",
        "whom", "why", "will", "with", "would",
        "you", "your", "yours", "yourself", "yourselves",
        "ll", "re", "ve", "m", "d",
    }
)
