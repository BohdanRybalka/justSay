"""Stored transcripts — the SQLite entry store, its search lanes and word stats.

These seven modules form a closed sub-graph: `history` owns the connection and
the `entries` table, `vector_store` adds the semantic lane over the same
connection and lock, `words` owns the keyword lane and the statistics, and the
two routers expose them. Their only dependency outside this package is
`app.embeddings`, which supplies the query vector.

They lived in `app.core` until spec 076. That placement made `core` both a leaf
layer and a consumer of the feature packages at once, so `core -> embeddings`
read as a cycle and several imports had to be deferred into function bodies to
break it. As a sibling package the edge is ordinary and the imports are plain.
"""
