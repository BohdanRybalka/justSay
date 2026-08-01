"""The default Windows render endpoint for each ERole, read through COM.

`IMMDeviceEnumerator::GetDefaultAudioEndpoint(eRender, role)` is the only way
to ask for a role other than the console one, and `pyaudiowpatch` exposes no
such parameter. `ctypes` rather than `comtypes`: stdlib, no third compiled
extension in the sidecar, no runtime code generation for PyInstaller to trip
over.

Every `ctypes.windll` and `ctypes.WINFUNCTYPE` reference lives inside a
function body, so this module imports on any platform and the COM-lifetime
rules below are unit-tested against a fake `ole32`.

See docs/adr/042-loopback-follows-the-communications-endpoint.md.
"""

from __future__ import annotations

import ctypes
import logging
from collections.abc import Iterator
from contextlib import contextmanager

from app.audio.endpoint_selection import ENDPOINT_ROLES, EndpointRole
from app.audio.system_source import SystemAudioUnavailableError

log = logging.getLogger(__name__)

S_OK = 0x00000000
S_FALSE = 0x00000001
RPC_E_CHANGED_MODE = 0x80010106

_COM_USABLE_RESULTS = frozenset({S_OK, S_FALSE, RPC_E_CHANGED_MODE})

_COINIT_MULTITHREADED = 0x0
_CLSCTX_ALL = 0x17
_STGM_READ = 0x00000000
_VT_LPWSTR = 31

_E_RENDER = 0
_ROLE_VALUES: dict[EndpointRole, int] = {"console": 0, "communications": 2}

_RELEASE_SLOT = 2
_GET_DEFAULT_AUDIO_ENDPOINT_SLOT = 4
_OPEN_PROPERTY_STORE_SLOT = 4
_GET_VALUE_SLOT = 5


def com_result_is_usable(hresult: int) -> bool:
    """Whether a `CoInitializeEx` result means the thread's COM may be used.

    Microsoft documents three non-error results: `S_OK` when this call did the
    initialisation, `S_FALSE` when the thread was already initialised with the
    same concurrency model, and `RPC_E_CHANGED_MODE` when it was already
    initialised with a different one.
    """
    return _as_unsigned(hresult) in _COM_USABLE_RESULTS


def com_result_owns_initialisation(hresult: int) -> bool:
    """Whether this caller is the one that must call `CoUninitialize`."""
    return _as_unsigned(hresult) == S_OK


def _as_unsigned(hresult: int) -> int:
    return hresult & 0xFFFFFFFF


@contextmanager
def com_initialized(ole32: object) -> Iterator[None]:
    """Hold COM for the block, tearing down only an initialisation we own."""
    hresult = ole32.CoInitializeEx(None, _COINIT_MULTITHREADED)
    if not com_result_is_usable(hresult):
        raise SystemAudioUnavailableError(
            f"CoInitializeEx failed with 0x{_as_unsigned(hresult):08X}"
        )
    owned = com_result_owns_initialisation(hresult)
    try:
        yield
    finally:
        if owned:
            ole32.CoUninitialize()


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_ubyte * 8),
    ]


class _PropertyKey(ctypes.Structure):
    _fields_ = [("fmtid", _GUID), ("pid", ctypes.c_ulong)]


class _PropVariant(ctypes.Structure):
    _fields_ = [
        ("vt", ctypes.c_ushort),
        ("wReserved1", ctypes.c_ushort),
        ("wReserved2", ctypes.c_ushort),
        ("wReserved3", ctypes.c_ushort),
        ("pwszVal", ctypes.c_wchar_p),
        ("_tail", ctypes.c_ulonglong),
    ]


def _guid(data1: int, data2: int, data3: int, tail: bytes) -> _GUID:
    return _GUID(data1, data2, data3, (ctypes.c_ubyte * 8)(*tail))


_CLSID_MM_DEVICE_ENUMERATOR = _guid(
    0xBCDE0395, 0xE52F, 0x467C, b"\x8e\x3d\xc4\x57\x92\x91\x69\x2e"
)
_IID_IMM_DEVICE_ENUMERATOR = _guid(
    0xA95664D2, 0x9614, 0x4F35, b"\xa7\x46\xde\x8d\xb6\x36\x17\xe6"
)
_PKEY_DEVICE_FRIENDLY_NAME = _PropertyKey(
    _guid(0xA45C254E, 0xDF1C, 0x4EFD, b"\x80\x20\x67\xd1\x46\xa8\x50\xe0"), 14
)


def render_endpoint_names() -> dict[EndpointRole, str | None]:
    """The friendly name of the default render endpoint for each role.

    A role with no default endpoint, or whose property store refuses to answer,
    comes back as None so the caller can skip it.
    """
    ole32 = ctypes.windll.ole32
    with com_initialized(ole32):
        enumerator = _create_device_enumerator(ole32)
        try:
            return {
                role: _default_endpoint_name(ole32, enumerator, _ROLE_VALUES[role])
                for role in ENDPOINT_ROLES
            }
        finally:
            _release(enumerator)


def _method(pointer: ctypes.c_void_p, slot: int, *argtypes: object) -> object:
    vtable = ctypes.cast(
        pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))
    ).contents
    prototype = ctypes.WINFUNCTYPE(ctypes.HRESULT, ctypes.c_void_p, *argtypes)
    return prototype(vtable[slot])


def _release(pointer: ctypes.c_void_p) -> None:
    prototype = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)
    vtable = ctypes.cast(
        pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))
    ).contents
    prototype(vtable[_RELEASE_SLOT])(pointer)


def _create_device_enumerator(ole32: object) -> ctypes.c_void_p:
    enumerator = ctypes.c_void_p()
    hresult = ole32.CoCreateInstance(
        ctypes.byref(_CLSID_MM_DEVICE_ENUMERATOR),
        None,
        _CLSCTX_ALL,
        ctypes.byref(_IID_IMM_DEVICE_ENUMERATOR),
        ctypes.byref(enumerator),
    )
    if _as_unsigned(hresult) != S_OK or not enumerator:
        raise SystemAudioUnavailableError(
            f"CoCreateInstance(MMDeviceEnumerator) failed with 0x{_as_unsigned(hresult):08X}"
        )
    return enumerator


def _default_endpoint_name(
    ole32: object, enumerator: ctypes.c_void_p, role: int
) -> str | None:
    device = ctypes.c_void_p()
    get_endpoint = _method(
        enumerator,
        _GET_DEFAULT_AUDIO_ENDPOINT_SLOT,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
    )
    try:
        get_endpoint(enumerator, _E_RENDER, role, ctypes.byref(device))
    except OSError:
        return None
    if not device:
        return None
    try:
        return _friendly_name(ole32, device)
    finally:
        _release(device)


def _friendly_name(ole32: object, device: ctypes.c_void_p) -> str | None:
    store = ctypes.c_void_p()
    open_store = _method(
        device,
        _OPEN_PROPERTY_STORE_SLOT,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_void_p),
    )
    try:
        open_store(device, _STGM_READ, ctypes.byref(store))
    except OSError:
        return None
    if not store:
        return None

    value = _PropVariant()
    get_value = _method(
        store,
        _GET_VALUE_SLOT,
        ctypes.POINTER(_PropertyKey),
        ctypes.POINTER(_PropVariant),
    )
    try:
        get_value(
            store,
            ctypes.byref(_PKEY_DEVICE_FRIENDLY_NAME),
            ctypes.byref(value),
        )
        return value.pwszVal if value.vt == _VT_LPWSTR else None
    except OSError:
        return None
    finally:
        ole32.PropVariantClear(ctypes.byref(value))
        _release(store)
