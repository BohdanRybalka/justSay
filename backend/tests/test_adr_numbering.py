"""ADR numbers must be unique across docs/adr/.

docs/ is local-only (gitignored), so this guard cannot run in GitHub CI —
it runs on every local test pass instead, which is where ADRs get written.
Two ADRs once shared the number 010; this keeps that from recurring.
"""

import re
from collections import Counter
from pathlib import Path

import pytest

ADR_DIR = Path(__file__).resolve().parents[2] / "docs" / "adr"


def test_adr_numbers_are_unique() -> None:
    if not ADR_DIR.is_dir():
        pytest.skip("docs/adr not present (docs/ is local-only, absent in CI)")
    numbers = [
        match.group(1)
        for path in ADR_DIR.glob("*.md")
        if (match := re.match(r"(\d{3})-", path.name))
    ]
    assert numbers, "no numbered ADR files found in docs/adr"
    duplicates = sorted(n for n, count in Counter(numbers).items() if count > 1)
    assert not duplicates, f"duplicate ADR numbers: {duplicates}"
