"""No test in this suite may install packages into the machine running it.

Spec 149: a pipeline test reached `local_setup._run_pip_install` and ran a real
`pip install .[local]` mid-suite, which is also the only reason a later test
could `import faster_whisper` on a CI runner that never installed it. The
autouse `_no_real_package_installs` fixture in `conftest.py` closes that; this
module pins both halves of it -- the pure predicate it decides with, and the two
spawn entry points it patches.
"""

import asyncio
import subprocess
import sys

import pytest

from tests.conftest import is_package_installer_command

_WHISPER_SERVER_ARGV = [
    "C:/Program Files/JustSay/whisper-server.exe",
    "--model",
    "ggml-large-v3-turbo.bin",
    "--host",
    "127.0.0.1",
    "--port",
    "8081",
]


@pytest.mark.parametrize(
    "argv",
    [
        [sys.executable, "-m", "pip", "install", "x"],
        ["pip", "install", "x"],
        ["pip3.12.exe", "install", "x"],
        ["uv", "pip", "install", "x"],
        ["uv", "add", "x"],
        ["uv", "sync"],
        [sys.executable, "-m", "ensurepip"],
    ],
)
def test_installer_commands_are_recognised(argv):
    assert is_package_installer_command(argv) is True


@pytest.mark.parametrize(
    "argv",
    [
        [sys.executable, "-c", "import app"],
        ["nvidia-smi", "--query-gpu=name", "--format=csv"],
        _WHISPER_SERVER_ARGV,
    ],
)
def test_ordinary_commands_are_left_alone(argv):
    assert is_package_installer_command(argv) is False


def test_subprocess_run_of_an_installer_raises_instead_of_spawning():
    with pytest.raises(RuntimeError, match="tried to run a package installer"):
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--help"],
            capture_output=True,
        )


@pytest.mark.asyncio
async def test_create_subprocess_exec_of_an_installer_raises_instead_of_spawning():
    with pytest.raises(RuntimeError, match="tried to run a package installer"):
        await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "pip",
            "install",
            "--help",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
