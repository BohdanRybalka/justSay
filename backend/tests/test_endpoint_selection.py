"""Spec 074: which Windows render endpoint a meeting recording captures.

Everything here is pure. `app.audio.endpoint_selection` imports nothing
platform-specific, and `app.audio.windows_endpoints` keeps every `windll` and
`WINFUNCTYPE` reference inside a function body, so both import and run on the
ubuntu CI runner. The COM lifetime rules are driven against a fake `ole32`.
"""

from __future__ import annotations

import ctypes
import logging
from unittest.mock import MagicMock

import pytest

from app.audio import windows_endpoints
from app.audio.config import AudioSettings
from app.audio.endpoint_selection import resolve_loopback_device
from app.audio.system_source import SystemAudioUnavailableError
from app.audio.windows_endpoints import (
    RPC_E_CHANGED_MODE,
    S_FALSE,
    S_OK,
    com_initialized,
    com_result_is_usable,
    com_result_owns_initialisation,
)


def loopback(name: str, index: int = 0) -> dict:
    return {
        "index": index,
        "name": name,
        "defaultSampleRate": 48000.0,
        "maxInputChannels": 2,
    }


def test_the_communications_role_wins_by_default():
    """AC: the role preference decides, and neither result is inferable from
    the other — a resolver that ignored the preference fails one of the two."""
    role_names = {"communications": "Headset", "console": "Speakers"}
    devices = [loopback("Speakers [Loopback]", 1), loopback("Headset [Loopback]", 2)]

    chosen = resolve_loopback_device(role_names, devices, "communications")

    assert chosen["name"] == "Headset [Loopback]"


def test_the_console_role_wins_when_it_is_the_preference():
    role_names = {"communications": "Headset", "console": "Speakers"}
    devices = [loopback("Speakers [Loopback]", 1), loopback("Headset [Loopback]", 2)]

    chosen = resolve_loopback_device(role_names, devices, "console")

    assert chosen["name"] == "Speakers [Loopback]"


@pytest.mark.parametrize("reverse", [False, True])
def test_an_exact_match_beats_a_substring_match_in_either_enumeration_order(reverse):
    """AC: `pyaudiowpatch`'s own helper matches by substring and returns the
    first hit, so it answers differently depending on enumeration order. Both
    orders are asserted precisely to pin that difference."""
    devices = [
        loopback("Studio Monitors [Loopback]", 1),
        loopback("Studio Monitors 2 [Loopback]", 2),
    ]
    if reverse:
        devices.reverse()

    chosen = resolve_loopback_device(
        {"communications": "Studio Monitors", "console": None}, devices, "communications"
    )

    assert chosen["name"] == "Studio Monitors [Loopback]"


def test_a_substring_match_is_still_used_when_no_exact_name_exists():
    """Not every driver formats the analogue as `<name> [Loopback]`."""
    devices = [loopback("Speakers (Realtek) (loopback)", 3)]

    chosen = resolve_loopback_device(
        {"communications": "Speakers (Realtek)", "console": None}, devices, "communications"
    )

    assert chosen["index"] == 3


def test_the_preferred_role_falls_back_to_the_other_role():
    """AC: communications names an endpoint with no analogue → console's."""
    role_names = {"communications": "Bluetooth Headset", "console": "Speakers"}
    devices = [loopback("Speakers [Loopback]", 1)]

    chosen = resolve_loopback_device(role_names, devices, "communications")

    assert chosen["name"] == "Speakers [Loopback]"


def test_no_role_with_an_analogue_resolves_to_none():
    """AC: neither role has an analogue → None."""
    role_names = {"communications": "Headset", "console": "Speakers"}

    devices = [loopback("Microphone", 4)]

    assert resolve_loopback_device(role_names, devices, "communications") is None


def test_a_role_the_platform_reports_as_none_is_skipped_rather_than_matched():
    """AC: a None role name must never match anything, including by accident."""
    role_names = {"communications": None, "console": "Speakers"}
    devices = [loopback("Speakers [Loopback]", 1)]

    assert resolve_loopback_device(role_names, devices, "communications")["index"] == 1
    silent = {"communications": None, "console": None}
    assert resolve_loopback_device(silent, devices, "console") is None


def test_the_default_role_preference_is_communications():
    """ADR 042: Teams and Zoom render to the communications endpoint."""
    assert AudioSettings().meeting_system_endpoint_role == "communications"




class _FakeOle32:
    """Just the two entry points `com_initialized` uses."""

    def __init__(self, hresult: int):
        self._hresult = hresult
        self.uninitialize_calls = 0

    def CoInitializeEx(self, reserved, model):  # noqa: N802
        return self._hresult

    def CoUninitialize(self):  # noqa: N802
        self.uninitialize_calls += 1


_ARBITRARY_FAILURE = 0x80004005


@pytest.mark.parametrize(
    "hresult,usable,owned",
    [
        (S_OK, True, True),
        (S_FALSE, True, False),
        (RPC_E_CHANGED_MODE, True, False),
        (_ARBITRARY_FAILURE, False, False),
    ],
)
def test_com_initialisation_results_are_classified_by_microsofts_three_successes(
    hresult, usable, owned
):
    """AC: S_FALSE and RPC_E_CHANGED_MODE both mean "COM is usable".

    PortAudio's WASAPI host API initialises COM on the thread that opens it,
    and `WindowsLoopbackSource.__init__` constructs `pyaudiowpatch.PyAudio()`
    before the endpoint lookup — so a naive `hr == 0` check raises on every
    real Windows machine, and `create_system_audio_source` swallows it into a
    silent 501.
    """
    assert com_result_is_usable(hresult) is usable
    assert com_result_owns_initialisation(hresult) is owned


@pytest.mark.parametrize(
    "hresult,expected_uninitialize_calls",
    [(S_OK, 1), (S_FALSE, 0), (RPC_E_CHANGED_MODE, 0)],
)
def test_com_is_uninitialised_only_when_this_caller_initialised_it(
    hresult, expected_uninitialize_calls
):
    """A thread whose COM we did not initialise is not ours to tear down."""
    ole32 = _FakeOle32(hresult)

    with com_initialized(ole32):
        pass

    assert ole32.uninitialize_calls == expected_uninitialize_calls


def test_a_failed_com_initialisation_is_an_unavailable_source():
    ole32 = _FakeOle32(_ARBITRARY_FAILURE)

    with pytest.raises(SystemAudioUnavailableError, match="80004005"):
        with com_initialized(ole32):
            pass

    assert ole32.uninitialize_calls == 0


def test_a_signed_negative_hresult_is_classified_the_same_as_its_unsigned_form():
    """`ctypes` hands back `RPC_E_CHANGED_MODE` as a negative `c_int`."""
    signed = RPC_E_CHANGED_MODE - 0x100000000

    assert com_result_is_usable(signed) is True
    assert com_result_owns_initialisation(signed) is False


def test_com_is_released_even_when_the_block_raises():
    ole32 = _FakeOle32(S_OK)

    with pytest.raises(RuntimeError, match="boom"):
        with com_initialized(ole32):
            raise RuntimeError("boom")

    assert ole32.uninitialize_calls == 1


def _raise_com_failure(*args, **kwargs):
    raise OSError("[WinError -2147024809] The parameter is incorrect")


@pytest.fixture
def com_log(caplog):
    with caplog.at_level(logging.DEBUG, logger="app.audio.windows_endpoints"):
        yield caplog


def _com_failures(caplog) -> list:
    return [
        r for r in caplog.records
        if r.name == "app.audio.windows_endpoints" and r.exc_info
    ]


def test_a_refused_default_endpoint_query_is_recorded(monkeypatch, com_log):
    """ADR 042 exists so a meeting follows the communications endpoint. A COM
    failure here is reported as "this role has no default endpoint", which
    `resolve_loopback_device` then skips — recording from the console endpoint
    instead, the exact defect the ADR prevents, with nothing in the log."""
    monkeypatch.setattr(windows_endpoints, "_method", lambda *a, **k: _raise_com_failure)

    name = windows_endpoints._default_endpoint_name(
        MagicMock(), ctypes.c_void_p(0x1234), windows_endpoints._ROLE_VALUES["communications"]
    )

    assert name is None
    assert len(_com_failures(com_log)) == 1


def test_a_refused_property_store_is_recorded(monkeypatch, com_log):
    monkeypatch.setattr(windows_endpoints, "_method", lambda *a, **k: _raise_com_failure)

    name = windows_endpoints._friendly_name(MagicMock(), ctypes.c_void_p(0x1234))

    assert name is None
    assert len(_com_failures(com_log)) == 1


def test_a_refused_friendly_name_read_is_recorded(monkeypatch, com_log):
    def _open_store(_device, _mode, out):
        out._obj.value = 0x5678

    def _dispatch(_pointer, slot, *_argtypes):
        if slot == windows_endpoints._OPEN_PROPERTY_STORE_SLOT:
            return _open_store
        return _raise_com_failure

    monkeypatch.setattr(windows_endpoints, "_method", _dispatch)
    monkeypatch.setattr(windows_endpoints, "_release", lambda _pointer: None)

    name = windows_endpoints._friendly_name(MagicMock(), ctypes.c_void_p(0x1234))

    assert name is None
    assert len(_com_failures(com_log)) == 1

