"""Fetch the pinned TEN VAD Windows x64 shared library into backend/vendor/ten-vad/.

Spec 033 / docs/adr/019-ten-vad-neural-silence-gate.md. Run once locally to
enable the neural VAD layer; a checkout without it degrades to the spec-029
energy guard alone (``resolve_ten_vad_lib()`` returns ``None``). The release
workflow runs it on the Windows leg before the PyInstaller sidecar build.

    python backend/scripts/fetch_ten_vad.py

Trust-on-first-use: the tag and both artifacts' SHA-256 digests are pinned
constants recorded at implementation time and verified on every run. A
mismatch exits nonzero WITHOUT writing the final file — a silently swapped
binary must never land in a shipped sidecar. Re-running with matching files
already on disk is a no-op.
"""

import hashlib
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

# STDLIB ONLY, deliberately. httpx is declared only in the `dev` extra, while
# release.yml's Windows leg installs ".[cloud,audio]" and receives httpx only
# transitively via groq — availability by accident, which nothing pins. This
# script runs before the sidecar build in that exact environment.

# Pinned upstream release. TEN-framework/ten-vad has exactly two tags (v1.0,
# v1.0-ONNX); v1.0 is the prebuilt-native-library one. Verified against the
# tag's real tree at implementation time — the DLL genuinely lives at
# lib/Windows/x64/ten_vad.dll (plan AC 21).
TEN_VAD_TAG = "v1.0"
_RAW_BASE = "https://raw.githubusercontent.com/TEN-framework/ten-vad"

# The tag's tree carries NO license file (verified: repo root at v1.0 holds
# only .gitignore/README.md/requirements.txt/setup.py plus directories), so
# the license is pinned separately by an immutable commit SHA on `main`
# rather than by tag. See plan.md Deviations.
LICENSE_COMMIT = "22a3bcd4509d0faaa8eef4881e8af5f39c178950"

_DLL_SHA256 = "38937f5604fa93a7941db7b9326992b792fa3731ebf9353973b3234457c6064b"
_LICENSE_SHA256 = "03d862b922867b8c3406a0c17791e256ed3d76f197d88c1996c4aebfb79f688e"

# (url, destination filename, expected sha256)
_ARTIFACTS = [
    (f"{_RAW_BASE}/{TEN_VAD_TAG}/lib/Windows/x64/ten_vad.dll", "ten_vad.dll", _DLL_SHA256),
    (f"{_RAW_BASE}/{LICENSE_COMMIT}/LICENSE", "LICENSE", _LICENSE_SHA256),
]

VENDOR_DIR = Path(__file__).resolve().parents[1] / "vendor" / "ten-vad"

# The hash gate protects integrity but runs only AFTER the body is in memory.
# A redirected or compromised endpoint serving a multi-GB body would exhaust
# the build machine's RAM before verification ever happens, so the read is
# bounded first. The real artifacts are ~510 KB (DLL) and ~1 KB (LICENSE);
# 64 MB is generous headroom for an upstream growing, tight enough to matter.
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
    # urlopen follows redirects and raises HTTPError on non-2xx by default,
    # so this matches the previous raise_for_status + follow_redirects flow.
    with urllib.request.urlopen(url, timeout=120.0) as resp:
        payload = _read_bounded(resp, name, url)

    # Hash the payload BEFORE it reaches its final name: a mismatch must
    # leave no partially-trusted file behind for the sidecar build to pick up.
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


def main() -> None:
    print(f"Fetching TEN VAD {TEN_VAD_TAG} into {VENDOR_DIR}")
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)

    changed = False
    for url, name, expected_sha in _ARTIFACTS:
        changed |= _fetch_one(url, name, expected_sha)

    print("Done." if changed else "Done (nothing to do).")


if __name__ == "__main__":
    main()
