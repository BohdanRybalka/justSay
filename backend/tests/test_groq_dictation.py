"""A Groq dictation piece — its prompt, the verbose answer, the doubts read off it, the model asked.

The SDK request is replaced by a fake that records its arguments and answers a scripted
``verbose_json`` shape, as objects or as dicts.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from app.stt.config import STTSettings
from app.stt.glossary import WHISPER_PROMPT_CHAR_BUDGET, dictation_prompt
from app.stt.groq_whisper import GroqWhisperSTTProvider
from app.stt.languages import LANGUAGE_SENTENCES
from app.stt.whisper_doubts import read_doubts

SPEECH = {"no_speech_prob": 0.1, "avg_logprob": -0.2, "compression_ratio": 1.5}
NON_SPEECH = {"no_speech_prob": 0.9, "avg_logprob": -1.4, "compression_ratio": 1.2}
LOOPING = {"no_speech_prob": 0.1, "avg_logprob": -0.2, "compression_ratio": 2.6}


def _wav(tmp_path: Path) -> Path:
    path = tmp_path / "piece.flac"
    sf.write(str(path), np.zeros(16000, dtype=np.float32), 16000)
    return path


def _answer(text: str, *segments: dict) -> SimpleNamespace:
    return SimpleNamespace(text=text, language="ukrainian", segments=list(segments))


async def _send(tmp_path, answer, *, second_opinion=False, **settings):
    provider = GroqWhisperSTTProvider(STTSettings(groq_api_key="test-key", **settings))
    provider._client = MagicMock()
    asked: list[tuple] = []

    def request(client, model, audio_path, language, prompt, response_format):
        asked.append((model, language, prompt, response_format))
        return answer

    with patch.object(GroqWhisperSTTProvider, "_request", side_effect=request):
        result = await provider.transcribe(
            _wav(tmp_path), language="uk", dictation=True, second_opinion=second_opinion
        )
    return result, asked


@pytest.mark.parametrize(
    ("text", "segments", "language", "signals"),
    [
        ("вперед", [SPEECH], "uk", 0),
        ("вперед", [SPEECH, NON_SPEECH], "uk", 1),
        ("вперед", [{**NON_SPEECH, "avg_logprob": -0.5}], "uk", 0),
        ("вперед", [{**NON_SPEECH, "no_speech_prob": 0.5}], "uk", 0),
        ("вперед", [LOOPING], "uk", 1),
        ("про это", [SPEECH], "uk", 1),
        ("Ещё", [SPEECH], "uk", 1),
        ("про это", [SPEECH], "en", 0),
        ("про это", [NON_SPEECH, LOOPING], "uk", 3),
        ("вперед", None, "uk", 0),
        ("вперед", [{"no_speech_prob": True, "avg_logprob": -2.0}], "uk", 0),
    ],
)
def test_a_piece_is_doubted_for_non_speech_loops_and_russian_letters(
    text, segments, language, signals
):
    assert read_doubts(text, segments, language).signals == signals


@pytest.mark.parametrize(
    ("segments", "non_speech"),
    [([NON_SPEECH, NON_SPEECH], True), ([NON_SPEECH, SPEECH], False), ([], False)],
)
def test_only_a_piece_whose_every_segment_is_non_speech_holds_no_speech(segments, non_speech):
    assert read_doubts("дякую", segments, "uk").is_non_speech is non_speech


def test_the_dictation_sentence_comes_after_the_glossary():
    prompt, cut = dictation_prompt("Tauri, FastAPI", "uk")

    assert prompt == f"Tauri, FastAPI\n{LANGUAGE_SENTENCES['uk']}"
    assert cut == 0


def test_glossary_terms_are_trimmed_so_the_sentence_fits_the_budget():
    glossary = ", ".join(f"термін{i:03d}" for i in range(43))
    assert len(glossary) <= WHISPER_PROMPT_CHAR_BUDGET

    prompt, cut = dictation_prompt(glossary, "uk")

    assert len(prompt) <= WHISPER_PROMPT_CHAR_BUDGET
    assert prompt.endswith(f"\n{LANGUAGE_SENTENCES['uk']}")
    assert cut > 0
    assert glossary.startswith(prompt.split("\n")[0])


def test_a_dictation_without_a_glossary_still_gets_its_sentence():
    assert dictation_prompt("  ", "uk") == (LANGUAGE_SENTENCES["uk"], 0)


def test_a_language_without_a_sentence_gets_the_glossary_alone():
    assert dictation_prompt("Tauri", "auto") == ("Tauri", 0)


async def test_a_dictation_piece_asks_verbose_json_in_its_language_with_the_prompt(tmp_path):
    result, asked = await _send(tmp_path, _answer("вперед", SPEECH), initial_prompt="Tauri")

    assert asked == [
        ("whisper-large-v3-turbo", "uk", f"Tauri\n{LANGUAGE_SENTENCES['uk']}", "verbose_json")
    ]
    assert result.text == "вперед"
    assert result.doubts == 0


async def test_a_second_opinion_is_asked_of_the_stronger_model(tmp_path):
    _result, asked = await _send(tmp_path, _answer("вперед", SPEECH), second_opinion=True)

    assert [model for model, *_ in asked] == ["whisper-large-v3"]


async def test_a_dictation_piece_reports_how_many_doubts_its_answer_raised(tmp_path):
    result, _asked = await _send(tmp_path, _answer("в керівництві. про это", SPEECH, LOOPING))

    assert result.text == "в керівництві. про это"
    assert result.doubts == 2


async def test_a_piece_whose_every_segment_is_non_speech_comes_back_empty(tmp_path):
    result, _asked = await _send(tmp_path, _answer("Дякую за перегляд!", NON_SPEECH))

    assert result.text == ""


def test_every_send_asks_temperature_zero(tmp_path):
    client = MagicMock()
    client.audio.transcriptions.create.return_value = "ok"

    GroqWhisperSTTProvider._call_groq(
        client, "whisper-large-v3-turbo", _wav(tmp_path), "uk", None
    )

    assert client.audio.transcriptions.create.call_args.kwargs["temperature"] == 0
