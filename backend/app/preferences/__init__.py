"""User preferences — the persisted settings file and the endpoints over it.

`user_settings` owns `settings.json`: reading it, validating a partial update,
writing it back, and pushing the accepted values into the runtime `AppSettings`
singleton. That last step is why this is not a leaf — applying a change reaches
the STT and embedding provider caches to clear them, and the transcript store
to relocate the database when the output directory moves (ADR 076).
"""
