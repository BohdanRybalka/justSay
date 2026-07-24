"""Tests for the magic-bytes audio validator (Plan 008 / Task 3 tech-debt batch).

`/pipeline/process-file` used to trust the filename's
extension alone. This module covers the new content-aware validation:
empty/short files, extension/content mismatch (renamed executables), niche
formats we trust on extension, and the MIME→extension family map.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.core.audio_validation import (
    detect_audio_mime,
    mime_for_extension,
    validate_audio_upload,
)
from app.core.constants import ALLOWED_AUDIO_EXTENSIONS, MIME_BY_AUDIO_EXTENSION


def test_every_allowed_extension_has_a_mime():
    """If a new container is added to ALLOWED_AUDIO_EXTENSIONS, the MIME map
    must grow with it — else the Gemini call falls back to ``audio/wav`` and
    silently mangles the upload."""
    missing = ALLOWED_AUDIO_EXTENSIONS - MIME_BY_AUDIO_EXTENSION.keys()
    assert missing == set(), f"Extensions without MIME mapping: {missing}"



def test_detect_wav_riff_header():
    wav = b"RIFF" + (b"\x00" * 4) + b"WAVE" + (b"\x00" * 32)
    assert detect_audio_mime(wav) == "audio/wav"


def test_detect_mp3_with_id3v2_tag():
    mp3 = b"ID3" + (b"\x00" * 32)
    assert detect_audio_mime(mp3) == "audio/mpeg"


@pytest.mark.parametrize("second_byte", [0xFB, 0xF3, 0xE3, 0xF2])
def test_detect_mp3_with_frame_sync(second_byte):
    raw = bytes([0xFF, second_byte]) + (b"\x00" * 32)
    assert detect_audio_mime(raw) == "audio/mpeg"


def test_detect_ogg_container():
    assert detect_audio_mime(b"OggS" + (b"\x00" * 32)) == "audio/ogg"


def test_detect_flac():
    assert detect_audio_mime(b"fLaC" + (b"\x00" * 32)) == "audio/flac"


def test_detect_mp4_ftyp():
    mp4 = (b"\x00" * 4) + b"ftyp" + b"M4A " + (b"\x00" * 16)
    assert detect_audio_mime(mp4) == "audio/mp4"


def test_detect_webm_ebml():
    assert detect_audio_mime(b"\x1a\x45\xdf\xa3" + (b"\x00" * 32)) == "audio/webm"


def test_detect_aiff():
    assert detect_audio_mime(b"FORM" + (b"\x00" * 4) + b"AIFF" + (b"\x00" * 32)) == "audio/aiff"


def test_detect_wma_asf():
    assert detect_audio_mime(b"\x30\x26\xb2\x75" + (b"\x00" * 32)) == "audio/x-ms-wma"


def test_detect_unknown_returns_none():
    assert detect_audio_mime(b"NOT AN AUDIO FILE THAT WE KNOW ABOUT") is None


def test_detect_short_payload_returns_none():
    """Sub-16-byte payloads must not crash the detector — they return None
    so the validator can produce the explicit 400 path."""
    assert detect_audio_mime(b"RIFF") is None
    assert detect_audio_mime(b"") is None



def _wav_bytes(payload_size: int = 1024) -> bytes:
    """Synthesise a minimal RIFF/WAVE header + payload."""
    return b"RIFF" + (b"\x00" * 4) + b"WAVE" + (b"\x00" * 4) + (b"\x00" * payload_size)


def test_validate_accepts_real_wav():
    assert validate_audio_upload(_wav_bytes(), "speech.wav") == "audio/wav"


def test_validate_rejects_empty_file():
    with pytest.raises(HTTPException) as exc_info:
        validate_audio_upload(b"", "speech.wav")
    assert exc_info.value.status_code == 400
    assert "too small" in exc_info.value.detail.lower()


def test_validate_rejects_sub_16_byte_file():
    with pytest.raises(HTTPException) as exc_info:
        validate_audio_upload(b"RIFF" + b"\x00" * 4, "speech.wav")
    assert exc_info.value.status_code == 400


def test_validate_rejects_unknown_extension():
    with pytest.raises(HTTPException) as exc_info:
        validate_audio_upload(_wav_bytes(), "malware.exe")
    assert exc_info.value.status_code == 400
    assert "Unsupported audio format" in exc_info.value.detail


def test_validate_rejects_missing_filename():
    with pytest.raises(HTTPException) as exc_info:
        validate_audio_upload(_wav_bytes(), None)
    assert exc_info.value.status_code == 400


def test_validate_rejects_extension_content_mismatch():
    """A file renamed from .exe to .wav but with no audio magic bytes is rejected."""
    fake = b"MZ" + (b"\x00" * 64)
    with pytest.raises(HTTPException) as exc_info:
        validate_audio_upload(fake, "evil.wav")
    assert exc_info.value.status_code == 400
    assert "does not match" in exc_info.value.detail.lower()


def test_validate_rejects_mp3_content_under_wav_extension():
    """Cross-format spoofing: MP3 frame sync but .wav extension is rejected."""
    mp3 = bytes([0xFF, 0xFB]) + (b"\x00" * 32)
    with pytest.raises(HTTPException) as exc_info:
        validate_audio_upload(mp3, "speech.wav")
    assert exc_info.value.status_code == 400
    assert "audio/mpeg" in exc_info.value.detail


def test_validate_accepts_aac_on_extension_trust():
    """AAC ADTS frames collide with MP3 magic. We trust the .aac extension."""
    fake_aac = bytes([0xFF, 0xF1]) + (b"\x00" * 32)
    assert validate_audio_upload(fake_aac, "voice.aac") == "audio/aac"


def test_validate_accepts_opus_inside_ogg():
    """An .opus file rides in an OggS container — accept the cross-mapping."""
    opus = b"OggS" + (b"\x00" * 32)
    assert validate_audio_upload(opus, "voice.opus") == "audio/ogg"


def test_validate_accepts_m4a_with_mp4_magic():
    m4a = (b"\x00" * 4) + b"ftyp" + b"M4A " + (b"\x00" * 16)
    assert validate_audio_upload(m4a, "voice.m4a") == "audio/mp4"



def test_mime_for_extension_known_formats():
    assert mime_for_extension("audio.wav") == "audio/wav"
    assert mime_for_extension("audio.MP3") == "audio/mpeg"
    assert mime_for_extension("audio.webm") == "audio/webm"


def test_mime_for_extension_unknown_falls_back_to_wav():
    """Defensive default — keeps Gemini calls non-broken if a new container
    bypasses the validator."""
    assert mime_for_extension("strange.xyz") == "audio/wav"
    assert mime_for_extension(None) == "audio/wav"
