"""Fetch the pinned TEN VAD shared library into backend/vendor/ten-vad/.

Spec 033 / docs/adr/019-ten-vad-neural-silence-gate.md, extended to macOS by
spec 170 / docs/adr/070-neural-silence-gate-ships-on-macos.md. Run once
locally to enable the neural VAD layer; a checkout without it degrades to the
spec-029 energy guard alone (``resolve_ten_vad_lib()`` returns ``None``).

    python backend/scripts/fetch_ten_vad.py
    python backend/scripts/fetch_ten_vad.py --platform darwin

One artifact is pinned per shipped platform: ``win32`` takes
``lib/Windows/x64/ten_vad.dll`` and ``darwin`` takes the payload blob of
``lib/macOS/ten_vad.framework``, a fat Mach-O carrying an x86_64 and an arm64
slice. The macOS blob is written as ``libten_vad.dylib`` — the name
``app.audio.vad._platform_lib_name()`` resolves and ``build_sidecar.spec``
bundles — because it is an ``MH_DYLIB`` depending only on system libraries and
loaded by absolute path through ``ctypes.CDLL``, so nothing about the
surrounding framework layout is load-bearing. ``--platform`` defaults to the
host's ``sys.platform`` and exists so a platform's URL and digest can be
verified from any machine; a platform with no pinned artifact exits nonzero
naming it, rather than fetching a library the resolver will never look for.
Both CI workflows run this before the steps that need the library: `ci.yml`
before pytest, `release.yml` before the PyInstaller sidecar build.

Trust-on-first-use: the tag and every artifact's SHA-256 digest are pinned
constants recorded at implementation time and verified on every run. A
mismatch exits nonzero WITHOUT writing the final file — a silently swapped
binary must never land in a shipped sidecar. Re-running with matching files
already on disk is a no-op.
"""

import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

TEN_VAD_TAG = "v1.0"
_RAW_BASE = "https://raw.githubusercontent.com/TEN-framework/ten-vad"

LICENSE_COMMIT = "22a3bcd4509d0faaa8eef4881e8af5f39c178950"

_WINDOWS_DLL_SHA256 = "38937f5604fa93a7941db7b9326992b792fa3731ebf9353973b3234457c6064b"
_MACOS_DYLIB_SHA256 = "81b2de13710670bb94fef315ab50fedc903a21c04c4290c6c2ac28d8b42e715a"
_LICENSE_SHA256 = "03d862b922867b8c3406a0c17791e256ed3d76f197d88c1996c4aebfb79f688e"

_MACOS_FRAMEWORK_PAYLOAD = "lib/macOS/ten_vad.framework/Versions/A/ten_vad"

LICENSE_ARTIFACT = (f"{_RAW_BASE}/{LICENSE_COMMIT}/LICENSE", "LICENSE", _LICENSE_SHA256)

PLATFORM_ARTIFACTS: dict[str, tuple[str, str, str]] = {
    "win32": (
        f"{_RAW_BASE}/{TEN_VAD_TAG}/lib/Windows/x64/ten_vad.dll",
        "ten_vad.dll",
        _WINDOWS_DLL_SHA256,
    ),
    "darwin": (
        f"{_RAW_BASE}/{TEN_VAD_TAG}/{_MACOS_FRAMEWORK_PAYLOAD}",
        "libten_vad.dylib",
        _MACOS_DYLIB_SHA256,
    ),
}

VENDOR_DIR = Path(__file__).resolve().parents[1] / "vendor" / "ten-vad"

_MAX_DOWNLOAD_BYTES = 64 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _too_large(name: str, url: str, detail: str) -> None:
    print(
        f"ERROR: refusing oversized download for {name}\n"
        f"  limit: {_MAX_DOWNLOAD_BYTES} bytes\n"
        f"  {detail}\n"
        f"  url:   {url}",
        file=sys.stderr,
    )
    sys.exit(1)


def _read_bounded(resp, name: str, url: str) -> bytes:
    """Read the response body, refusing anything past ``_MAX_DOWNLOAD_BYTES``.

    Content-Length is only a hint (absent on chunked responses, and a hostile
    server can lie), so it is checked as a cheap early reject and the read
    itself is capped independently: one byte over the cap is read on purpose,
    so an exactly-at-limit body is distinguishable from a truncated one.
    """
    declared = resp.headers.get("Content-Length")
    if declared is not None:
        try:
            declared_bytes = int(declared)
        except ValueError:
            declared_bytes = -1
        if declared_bytes > _MAX_DOWNLOAD_BYTES:
            _too_large(name, url, f"Content-Length: {declared_bytes} bytes")

    payload = resp.read(_MAX_DOWNLOAD_BYTES + 1)
    if len(payload) > _MAX_DOWNLOAD_BYTES:
        _too_large(name, url, "body exceeded the limit while reading")
    return payload


def _fetch_one(url: str, name: str, expected_sha: str) -> bool:
    """Download+verify one artifact. Returns True when it did real work."""
    dest = VENDOR_DIR / name
    if dest.exists() and _sha256(dest) == expected_sha:
        print(f"  {name}: already present and verified — skipping")
        return False

    print(f"  {name}: downloading {url}")
    with urllib.request.urlopen(url, timeout=120.0) as resp:
        payload = _read_bounded(resp, name, url)

    with tempfile.NamedTemporaryFile(delete=False, dir=str(VENDOR_DIR)) as tmp:
        tmp.write(payload)
        staged = Path(tmp.name)

    actual = _sha256(staged)
    if actual != expected_sha:
        staged.unlink(missing_ok=True)
        print(
            f"ERROR: SHA-256 mismatch for {name}\n"
            f"  expected: {expected_sha}\n"
            f"  actual:   {actual}\n"
            f"  url:      {url}",
            file=sys.stderr,
        )
        sys.exit(1)

    shutil.move(str(staged), str(dest))
    print(f"  {name}: verified ({len(payload)} bytes, sha256={actual[:16]}...)")
    return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="fetch_ten_vad")
    parser.add_argument(
        "--platform",
        default=sys.platform,
        help=(
            "sys.platform value to fetch the library for "
            f"(default: this host's, {sys.platform!r}; pinned: "
            f"{', '.join(sorted(PLATFORM_ARTIFACTS))})"
        ),
    )
    args = parser.parse_args(argv)

    platform_artifact = PLATFORM_ARTIFACTS.get(args.platform)
    if platform_artifact is None:
        print(
            f"ERROR: no TEN VAD artifact is pinned for platform {args.platform!r}\n"
            f"  pinned platforms: {', '.join(sorted(PLATFORM_ARTIFACTS))}\n"
            "  nothing was downloaded; this checkout degrades to the energy guard alone",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Fetching TEN VAD {TEN_VAD_TAG} for {args.platform} into {VENDOR_DIR}")
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)

    changed = False
    for url, name, expected_sha in (platform_artifact, LICENSE_ARTIFACT):
        changed |= _fetch_one(url, name, expected_sha)

    print("Done." if changed else "Done (nothing to do).")


if __name__ == "__main__":
    main()
