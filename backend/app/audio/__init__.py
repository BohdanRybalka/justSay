"""Audio module — microphone and system audio capture.

Deliberately empty of imports: importing any module under this package must
not pull in the capture stack. Recorders come from ``app.audio.recorder`` and
``app.audio.meeting_recorder``, the ``Depends()`` accessors from
``app.audio.dependencies``.
"""
