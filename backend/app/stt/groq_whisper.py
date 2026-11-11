"""Cloud STT provider — Groq Whisper (whisper-large-v3-turbo).

Cloud mode's only transcription engine. A dictation piece is asked as ``verbose_json`` with the
dictation sentence in its prompt, and its answer is read for signs that it went wrong.
"""

import asyncio
import logging
from pathlib import Path

from app.core.constants import GROQ_TIMEOUT_SECONDS, RATE_LIMIT_RETRY_SECONDS
from app.core.errors import ConfigurationError, ResourceUnavailableError
from app.stt.base import (
    NO_ANSWER_MESSAGE,
    SINGLE_ATTEMPT,
    WHISPER_WINDOW_SECONDS,
    STTProvider,
    TranscriptionResult,
    clean_transcript_text,
    min_no_speech_prob,
    normalize_detected_language,
)
from app.stt.config import STTSettings
from app.stt.glossary import dictation_prompt, glossary_summary, whisper_glossary
from app.stt.whisper_doubts import read_doubts

log = logging.getLogger(__name__)


NO_ANSWER_RETRY_SECONDS = "2"


def _retry_after(error: Exception) -> str:
    """The seconds Groq's ``retry-after`` header asks for, or the app's default without one."""
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None) or {}
    return str(headers.get("retry-after") or RATE_LIMIT_RETRY_SECONDS)


def _no_answer(error: Exception) -> bool:
    """Whether Groq gave no answer: a dropped connection, a timeout, or a server error."""
    if getattr(error, "status_code", None) in (500, 502, 504):
        return True
    from groq import APIConnectionError

    return isinstance(error, APIConnectionError)


class GroqWhisperSTTProvider(STTProvider):
    """Groq-hosted Whisper (whisper-large-v3-turbo), Cloud mode's transcription engine.

    Free-tier file size limit 25 MB, enforced upstream. Accepts WAV, MP3, FLAC
    and OGG, not .webm. The SDK call times out at `GROQ_TIMEOUT_SECONDS`.
    """

    longest_piece_seconds = WHISPER_WINDOW_SECONDS

    def __init__(self, settings: STTSettings):
        self._settings = settings
        self._client = None

    @property
    def model_name(self) -> str:
        return f"groq/{self._settings.groq_whisper_model}"

    def _get_client(self):
        if self._client is None:
            if not self._settings.groq_api_key:
                raise ConfigurationError(
                    "Groq API key is missing. Go to Settings → Keys and add your key."
                )
            from groq import Groq

            self._client = Groq(api_key=self._settings.groq_api_key, timeout=GROQ_TIMEOUT_SECONDS)
        return self._client

    async def transcribe(
        self, audio_path: Path, language: str = "uk", **kwargs
    ) -> TranscriptionResult:
        """Send audio file to Groq Whisper API; a ``dictation`` piece is read for doubts."""
        client = self._get_client()
        if kwargs.get("dictation"):
            return await self._transcribe_dictation(
                client, audio_path, language, second_opinion=bool(kwargs.get("second_opinion"))
            )
        size_kb = audio_path.stat().st_size / 1024
        prompt, dropped_terms = whisper_glossary(self._settings.initial_prompt)
        log.info(
            "Groq Whisper: POST transcriptions model=%s file=%s size=%.1fKB lang=%s glossary=%s",
            self._settings.groq_whisper_model, audio_path.name, size_kb, language,
            glossary_summary(prompt, dropped_terms),
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
            text=clean_transcript_text(text),
            tokens_used=None,
            detected_language=normalize_detected_language(detected_raw),
            no_speech_prob=no_speech_prob,
        )

    async def _transcribe_dictation(
        self, client, audio_path: Path, language: str, *, second_opinion: bool
    ) -> TranscriptionResult:
        """One push-to-talk piece; ``second_opinion`` sends it to the stronger model.

        A piece whose every segment is non-speech comes back with no text.
        """
        model = (
            self._settings.groq_second_opinion_model
            if second_opinion
            else self._settings.groq_whisper_model
        )
        prompt, cut = dictation_prompt(self._settings.initial_prompt, language)
        log.info(
            "Groq Whisper: POST dictation piece model=%s file=%s lang=%s prompt=%s",
            model, audio_path.name, language, glossary_summary(prompt, cut),
        )
        try:
            response = await asyncio.to_thread(
                self._request, client, model, audio_path, language, prompt, "verbose_json"
            )
        except Exception:
            log.exception("Groq Whisper call failed")
            raise
        text = clean_transcript_text(
            response if isinstance(response, str) else getattr(response, "text", None)
        )
        segments = getattr(response, "segments", None)
        doubts = read_doubts(text, segments, language)
        if doubts.is_non_speech:
            log.info("Groq Whisper: a piece held no speech; dropped %d chars", len(text))
            text = ""
        return TranscriptionResult(
            text=text,
            detected_language=normalize_detected_language(getattr(response, "language", None)),
            no_speech_prob=min_no_speech_prob(segments),
            doubts=doubts.signals,
        )

    def cleanup(self) -> None:
        """No persistent resources — HTTP client is stateless. Reset for consistency."""
        self._client = None

    @staticmethod
    def _request(
        client,
        model: str,
        audio_path: Path,
        language: str,
        prompt: str | None,
        response_format: str,
    ):
        """One SDK call at temperature 0, its errors turned into the app's refusals.

        Under ``SINGLE_ATTEMPT`` the SDK makes one request; no answer is a refusal to wait out.
        """
        if SINGLE_ATTEMPT.get():
            client = client.with_options(max_retries=0)
        try:
            with audio_path.open("rb") as fh:
                kwargs: dict = {
                    "file": (audio_path.name, fh.read()),
                    "model": model,
                    "response_format": response_format,
                    "temperature": 0,
                }
                if language and language != "auto":
                    kwargs["language"] = language
                if prompt:
                    kwargs["prompt"] = prompt
                response = client.audio.transcriptions.create(**kwargs)
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate_limit" in msg.lower():
                raise ResourceUnavailableError(
                    "Groq rate limit exceeded. Try again later.",
                    headers={"Retry-After": _retry_after(e)},
                ) from e
            if getattr(e, "status_code", None) == 503:
                raise ResourceUnavailableError(
                    "Groq is overloaded right now. Try again later.",
                    headers={"Retry-After": _retry_after(e)},
                ) from e
            if SINGLE_ATTEMPT.get() and _no_answer(e):
                raise ResourceUnavailableError(
                    NO_ANSWER_MESSAGE,
                    headers={"Retry-After": NO_ANSWER_RETRY_SECONDS},
                ) from e
            raise
        return response

    @staticmethod
    def _call_groq(
        client,
        model: str,
        audio_path: Path,
        language: str,
        prompt: str | None,
    ) -> tuple[str, str | None, float | None]:
        """Isolated SDK call — mockable in tests without installing groq.

        Returns ``(text, detected_language_raw, no_speech_prob)``; only
        ``language == "auto"`` escalates ``response_format`` to ``"verbose_json"``.
        """
        response_format = "verbose_json" if language == "auto" else "text"
        response = GroqWhisperSTTProvider._request(
            client, model, audio_path, language, prompt, response_format
        )
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
