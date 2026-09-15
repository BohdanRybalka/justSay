"""Stored transcripts — the SQLite entry store, its search lanes and word stats.

These twelve modules form a closed sub-graph. `history` owns the connection,
the lock and the reads and writes over `entries`; `schema` holds that table's
DDL, column lists and migrations; `relocation` moves the file when the output
directory changes; `search` owns the keyword and semantic lanes; `vector_store`
adds the embeddings over the same connection and lock; `words` owns the word
statistics; `store_errors` maps a locked store onto HTTP; the two `stopwords_*`
modules are data; and the two routers expose the rest. Their only dependency
outside this package is `app.embeddings`, which supplies the query vector.

They lived in `app.core` until spec 076. That placement made `core` both a leaf
layer and a consumer of the feature packages at once, so `core -> embeddings`
read as a cycle and several imports had to be deferred into function bodies to
break it. As a sibling package the edge is ordinary and the imports are plain.
"""
