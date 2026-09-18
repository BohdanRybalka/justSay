"""Stored transcripts — the SQLite entry store, its search lanes and word stats.

A closed sub-graph. `history` owns the connection, the lock and the reads and
writes over `entries`; `schema` holds that table's DDL, column lists and
migrations; `relocation` moves the file; `search` owns the keyword and semantic
lanes; `vector_store` adds the embeddings over the same connection and lock;
`words` owns the word statistics; `store_errors` maps a locked store onto HTTP.
Outside the package it reaches `app.embeddings` and `app.core`, nothing else.
"""
