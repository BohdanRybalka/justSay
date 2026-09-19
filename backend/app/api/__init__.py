"""The application's HTTP boundary: routes, middleware and exception handlers.

Starlette composes all three into one ASGI stack, so they are one layer owned by
the module that builds the application — `app/main.py`, this package's only
consumer. Nothing here is a primitive and nothing here may be imported by
`app.core` (ADR 082).
"""
