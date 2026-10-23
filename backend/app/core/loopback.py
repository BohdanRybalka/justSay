"""Whether a URL points at this computer, the only host Local mode may send text to."""

import ipaddress
from urllib.parse import urlparse


def is_loopback_url(url: str) -> bool:
    """True for an http(s) URL whose host is ``localhost`` or a loopback IP address."""
    try:
        parsed = urlparse(url.strip())
        host = parsed.hostname
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https") or not host:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
