"""Choosing which render endpoint's loopback analogue a meeting is captured from.

Pure: no COM, no PortAudio, no platform guard, nothing that cannot be imported
on the ubuntu CI runner. Every test of the decision points here.

See docs/adr/042-loopback-follows-the-communications-endpoint.md.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

EndpointRole = Literal["communications", "console"]

ENDPOINT_ROLES: tuple[EndpointRole, ...] = ("communications", "console")

LOOPBACK_SUFFIX = " [Loopback]"


def resolve_loopback_device(
    role_names: Mapping[EndpointRole, str | None],
    loopback_devices: Sequence[Mapping[str, object]],
    preferred_role: EndpointRole,
) -> Mapping[str, object] | None:
    """The loopback device to capture, or None when no role has an analogue.

    Roles are tried in preference order, and a role the platform reports as
    None is skipped rather than matched against.
    """
    for role in _roles_in_preference_order(preferred_role):
        name = role_names.get(role)
        if not name:
            continue
        device = _match_loopback_analogue(str(name), loopback_devices)
        if device is not None:
            return device
    return None


def _roles_in_preference_order(preferred_role: EndpointRole) -> list[EndpointRole]:
    return [preferred_role, *(role for role in ENDPOINT_ROLES if role != preferred_role)]


def _match_loopback_analogue(
    endpoint_name: str, loopback_devices: Sequence[Mapping[str, object]]
) -> Mapping[str, object] | None:
    """Exact match first, substring second.

    `pyaudiowpatch`'s own helper matches by substring alone and returns
    whichever candidate enumeration yields first, so two devices whose names
    share a prefix resolve differently depending on enumeration order.
    """
    exact = endpoint_name + LOOPBACK_SUFFIX
    for device in loopback_devices:
        if str(device.get("name", "")) == exact:
            return device
    for device in loopback_devices:
        if endpoint_name in str(device.get("name", "")):
            return device
    return None
