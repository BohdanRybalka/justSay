"""Cloud STT provider — Groq Whisper (whisper-large-v3-turbo).

Used for short audio clips (<= cloud_routing_threshold seconds) in normal style.
Long audio and ai_prompt style go to Gemini.
"""

import asyncio
import logging
from pathlib import Path

from app.core.constants import GROQ_TIMEOUT_SECONDS
from app.stt.base import (
    STTProvider,
    TranscriptionResult,
    min_no_speech_prob,
    normalize_detected_language,
)
from app.stt.config import STTSettings

log = logging.getLogger(__name__)


class GroqWhisperSTTProvider(STTProvider):
    """Groq-hosted Whisper (whisper-large-v3-turbo) for fast short-audio transcription.

    Notes:
        - Free tier file size limit: 25 MB (enforced upstream by /transcribe route).
        - Accepted formats: WAV, MP3, FLAC, OGG. NOT .webm.
        - Timeout on the SDK call: GROQ_TIMEOUT_SECONDS (generous for short audio).
    """

    def __init__(self, settings: STTSettings):
        self._settings = settings
        self._client = None

    @property
    def model_name(self) -> str:
        return f"groq/{self._settings.groq_whisper_model}"

    def _get_client(self):
        if self._client is None:
            if not self._settings.groq_api_key:
                raise RuntimeError(
                    "Groq API key is missing. Go to Settings → Keys and add your key."
                )
            from groq import Groq

            self._client = Groq(api_key=self._settings.groq_api_key, timeout=GROQ_TIMEOUT_SECONDS)
        return self._client

    async def transcribe(self, audio_path: Path, language: str = "uk", **kwargs) -> TranscriptionResult:
        """Send audio file to Groq Whisper API. ``style`` kwarg is ignored (Groq can't structure)."""
        client = self._get_client()
        size_kb = audio_path.stat().st_size / 1024
        prompt = self._settings.initial_prompt.strip() or None
        log.info(
            "Groq Whisper: POST transcriptions model=%s file=%s size=%.1fKB lang=%s glossary=%s",
            self._settings.groq_whisper_model, audio_path.name, size_kb, language,
            f"{len(prompt)}chars" if prompt else "none",
        )

        try:
            text, detected_raw, no_speech_prob = await asyncio.to_thread(
                self._call_groq,
                client,
                self._settings.groq_whisper_model,
                audio_path,
                language,
                prompt,
            )
        except Exception:
            log.exception("Groq Whisper call failed")
            raise
        return TranscriptionResult(
            text=text.strip() if text else "",
            tokens_used=None,
            detected_language=normalize_detected_language(detected_raw),
            no_speech_prob=no_speech_prob,
        )

    def cleanup(self) -> None:
        """No persistent resources — HTTP client is stateless. Reset for consistency."""
        self._client = None

    @staticmethod
    def _call_groq(
        client,
        model: str,
        audio_path: Path,
        language: str,
        prompt: str | None,
    ) -> tuple[str, str | None, float | None]:
        """Isolated SDK call — mockable in tests without installing groq.

        Returns ``(text, detected_language_raw, no_speech_prob)``. ``response_format``
        escalates to ``"verbose_json"`` only when ``language == "auto"`` --
        that's the only path that needs a detected language back. The
        explicit-language hot path keeps its exact current wire format
        (``"text"``, a bare string with no metadata) unchanged (spec 029 /
        docs/adr/016-detected-language-on-stt-contract.md).
        """
        response_format = "verbose_json" if language == "auto" else "text"
        try:
            with audio_path.open("rb") as fh:
                kwargs: dict = {
                    "file": (audio_path.name, fh.read()),
                    "model": model,
                    "response_format": response_format,
                }
                if language and language != "auto":
                    kwargs["language"] = language
                if prompt:
                    kwargs["prompt"] = prompt
                response = client.audio.transcriptions.create(**kwargs)
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate_limit" in msg.lower():
                raise RuntimeError(
                    "Groq rate limit exceeded. Try again later or switch STT to Gemini."
                ) from e
            raise

        if isinstance(response, str):
            return response, None, None

        text = getattr(response, "text", None)
        if text is None:
            text = str(response)
        detected_language = getattr(response, "language", None)
        no_speech_prob = (
            min_no_speech_prob(getattr(response, "segments", None))
            if response_format == "verbose_json"
            else None
        )
        return text, detected_language, no_speech_prob
