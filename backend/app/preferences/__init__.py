"""User preferences — the persisted settings file and the endpoints over it.

`user_settings` owns `settings.json`: reading it, validating a partial update,
writing it back, and pushing the accepted values into the runtime `AppSettings`
singleton. That last step is why this is not a leaf: applying a change has to
reach the STT and embedding provider caches to clear them, and the transcript
store to relocate the database when the output directory moves.

Those three edges are the reason these two modules left `app.core` in spec 076.
Sitting in `core` they made it import the feature packages it is supposed to sit
underneath, so the dependency direction could not be stated at all.
"""
