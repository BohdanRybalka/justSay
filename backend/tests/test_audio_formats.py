"""Tests for `app.pipeline.upload_validation` — both gates it holds.

The content gate (Plan 008 / Task 3 tech-debt batch): `/pipeline/process-file`
used to trust the filename's extension alone. `validate_audio_upload` covers
empty/short files, extension/content mismatch (renamed executables), niche
formats we trust on extension, and the MIME→extension family map.

The size gate: `read_upload_with_limit` streams an upload in fixed chunks and
raises 413 the moment the running total passes the limit, so an oversized
upload is refused mid-read rather than after the whole payload sits in memory.
The two share a module rather than any code, which is why the tests below are
in the same file and named apart.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from app.core.audio_formats import (
    ALLOWED_AUDIO_EXTENSIONS,
    DETECTED_MIME_TO_EXTENSIONS,
    TRUSTED_EXTENSIONS,
    detect_audio_mime,
    mime_for_extension,
)
from app.core.constants import MAX_UPLOAD_SIZE
from app.pipeline.upload_validation import read_upload_with_limit, validate_audio_upload


def test_every_allowed_extension_is_detectable_or_trusted():
    """An accepted extension the detector cannot recognise is rejected on every
    upload: `validate_audio_upload` demands a magic-bytes match unless the
    extension is explicitly trusted. Adding one to the MIME map without doing
    either makes that format permanently unusable."""
    detectable = {ext for exts in DETECTED_MIME_TO_EXTENSIONS.values() for ext in exts}
    unreachable = ALLOWED_AUDIO_EXTENSIONS - detectable - TRUSTED_EXTENSIONS
    assert unreachable == set(), f"Extensions no upload can satisfy: {unreachable}"


def test_detected_mime_map_names_only_allowed_extensions():
    """The reverse guard: a detector family naming an extension the MIME map
    does not carry would accept a file `mime_for_extension` then mislabels."""
    detectable = {ext for exts in DETECTED_MIME_TO_EXTENSIONS.values() for ext in exts}
    unknown = detectable - ALLOWED_AUDIO_EXTENSIONS
    assert unknown == set(), f"Detector names unknown extensions: {unknown}"


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


class _PayloadUpload:
    """The slice of UploadFile read_upload_with_limit uses: chunked async read.

    Records the size of every chunk it hands back, so a test can state how the
    payload was read and not merely what came out.
    """

    def __init__(self, payload: bytes) -> None:
        self._remaining = payload
        self.chunks_read: list[int] = []

    async def read(self, size: int) -> bytes:
        chunk, self._remaining = self._remaining[:size], self._remaining[size:]
        self.chunks_read.append(len(chunk))
        return chunk

    @property
    def bytes_read(self) -> int:
        return sum(self.chunks_read)


_MULTI_CHUNK_SIZE = 5 * 64 * 1024


@pytest.mark.asyncio
async def test_read_upload_with_limit_returns_a_payload_of_exactly_the_limit():
    payload = b"x" * 4096
    assert await read_upload_with_limit(_PayloadUpload(payload), 4096) == payload


@pytest.mark.asyncio
async def test_read_upload_with_limit_rejects_one_byte_over_the_limit():
    with pytest.raises(HTTPException) as excinfo:
        await read_upload_with_limit(_PayloadUpload(b"x" * 4097), 4096)
    assert excinfo.value.status_code == 413


@pytest.mark.asyncio
async def test_read_upload_with_limit_accumulates_a_payload_spanning_many_chunks():
    """The two boundary cases above fit inside a single read, so they pass
    whatever the chunk size is. This one is the reason the function reads in
    chunks at all: the payload must arrive in several pieces and be joined back
    in order."""
    payload = bytes(i % 251 for i in range(_MULTI_CHUNK_SIZE))
    upload = _PayloadUpload(payload)

    assert await read_upload_with_limit(upload, _MULTI_CHUNK_SIZE) == payload
    assert len(upload.chunks_read) > 2, (
        f"the payload came back in {len(upload.chunks_read)} read(s): the chunked "
        "accumulation this test exists for never ran"
    )


@pytest.mark.asyncio
async def test_read_upload_with_limit_stops_reading_before_buffering_it_all():
    """The running total is checked per chunk, so an oversized upload is
    refused partway through. Reading it whole and only then raising 413 would
    put the entire payload in memory first — which is what the limit exists to
    prevent."""
    upload = _PayloadUpload(b"x" * _MULTI_CHUNK_SIZE)

    with pytest.raises(HTTPException) as excinfo:
        await read_upload_with_limit(upload, 2 * 64 * 1024)

    assert excinfo.value.status_code == 413
    assert upload.bytes_read < _MULTI_CHUNK_SIZE, (
        f"the whole {_MULTI_CHUNK_SIZE}-byte payload was read before the 413: "
        "the size check is not running per chunk"
    )


class _SizedChunk:
    """A chunk that reports a length without holding the bytes.

    `read_upload_with_limit` measures each chunk with `len()` and raises
    before it joins anything, so a gigabyte-scale limit can be driven past
    without allocating a gigabyte.
    """

    def __init__(self, size: int) -> None:
        self._size = size

    def __len__(self) -> int:
        return self._size


class _OversizedUpload:
    """An upload whose very first chunk already exceeds the limit."""

    def __init__(self, size: int) -> None:
        self._size = size

    async def read(self, size: int) -> _SizedChunk:
        return _SizedChunk(self._size)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("max_size", "expected_detail"),
    [
        (512 * 1024, "File too large (max 0.5MB)"),
        (MAX_UPLOAD_SIZE, "File too large (max 25MB)"),
        (1024 * 1024 * 1024, "File too large (max 1024MB)"),
    ],
)
async def test_read_upload_with_limit_states_the_limit_the_user_will_read(
    max_size, expected_detail
):
    """The 413 detail reaches the user: `src/api.ts` surfaces `detail`
    verbatim. Two rounds of edits changed this string unnoticed — flooring it
    to `max 0MB` below a megabyte, then rendering a gigabyte limit as
    `1.02e+03MB` — because nothing asserted what it actually says."""
    with pytest.raises(HTTPException) as excinfo:
        await read_upload_with_limit(_OversizedUpload(max_size + 1), max_size)

    assert excinfo.value.detail == expected_detail
