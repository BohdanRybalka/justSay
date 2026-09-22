"""The settings slice each feature package owns: how it reads, and how it is read.

Two spellings of a slice's env prefix resolve to the same field --
`JUSTSAY_STT_MODE` and `JUSTSAY_STT__MODE` -- and both are documented, so a
`.env` written against either keeps working. The second half guards the name:
a module that imports a slice global and also names a parameter after it reads
whichever object the caller passed, silently, and no runtime error says so.
"""

import ast

import pytest

from app.audio.config import AudioSettings
from app.core.package_settings import PackageSettings
from app.embeddings.config import EmbeddingSettings
from app.stt.config import STTSettings
from tests.app_modules import app_modules

_SLICES = (
    (STTSettings, "JUSTSAY_STT_", "whisper_model_size", "medium", "medium"),
    (AudioSettings, "JUSTSAY_AUDIO_", "sample_rate", "48000", 48000),
    (
        EmbeddingSettings,
        "JUSTSAY_EMBEDDINGS_",
        "ollama_host",
        "http://probe:1",
        "http://probe:1",
    ),
)

_SLICE_GLOBALS = frozenset({"stt_settings", "audio_settings", "embedding_settings"})

_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)


@pytest.mark.parametrize("doubled", [False, True], ids=["one_underscore", "two_underscores"])
@pytest.mark.parametrize(
    "settings_cls, env_prefix, field, env_value, expected",
    _SLICES,
    ids=[slice_case[0].__name__ for slice_case in _SLICES],
)
def test_both_spellings_of_a_slice_env_prefix_reach_the_field(
    monkeypatch, settings_cls, env_prefix, field, env_value, expected, doubled
):
    """Every slice answers to its prefix with one trailing underscore and with two.

    Each class declares the one-underscore prefix itself; the doubled one is the
    form `env_nested_delimiter` on the composition root produced, and dropping it
    turns a set variable into silence -- a `gemini_api_key` that never arrives
    surfaces as an auth failure rather than as a configuration error.
    """
    variable = f"{env_prefix}{'_' if doubled else ''}{field.upper()}"
    monkeypatch.setenv(variable, env_value)
    assert getattr(settings_cls(), field) == expected, (
        f"{variable} did not reach {settings_cls.__name__}.{field}"
    )


def test_the_one_underscore_spelling_wins_when_both_are_set(monkeypatch):
    """Precedence is stated, not incidental: the prefix each class declares for
    itself outranks the doubled fallback."""
    monkeypatch.setenv("JUSTSAY_STT_WHISPER_MODEL_SIZE", "small")
    monkeypatch.setenv("JUSTSAY_STT__WHISPER_MODEL_SIZE", "medium")
    assert STTSettings().whisper_model_size == "small"


def _slice_globals_imported_and_shadowed() -> dict[str, tuple[frozenset[str], frozenset[str]]]:
    """Per module under `app/`: the slice globals it imports, and the ones it
    also names as a parameter.

    Asserts its own result non-empty -- a walk finding nothing would leave the
    gate below green over an empty package.
    """
    found: dict[str, tuple[frozenset[str], frozenset[str]]] = {}
    for module, path in app_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported: set[str] = set()
        parameters: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported |= {alias.asname or alias.name for alias in node.names} & _SLICE_GLOBALS
            elif isinstance(node, _FUNCTIONS):
                args = node.args
                named = (
                    *args.posonlyargs,
                    *args.args,
                    *args.kwonlyargs,
                    args.vararg,
                    args.kwarg,
                )
                parameters |= {arg.arg for arg in named if arg is not None} & _SLICE_GLOBALS
        if imported or parameters:
            found[module] = (frozenset(imported), frozenset(parameters))
    assert found, "no module under app/ imports or names a settings slice global"
    return found


def test_no_module_both_imports_a_slice_global_and_names_a_parameter_after_it():
    """The collision `app.stt` carries today, made loud instead of silent.

    `routing` and `local_setup` take an `STTSettings` parameter spelled
    `stt_settings`, the same name `app.stt.config` binds its live instance to.
    Neither module imports that instance, so nothing reads the wrong object --
    and the day one does, this gate fires instead of the parameter quietly
    shadowing the global for the whole function body.
    """
    shadowing = {
        module: sorted(imported & parameters)
        for module, (imported, parameters) in _slice_globals_imported_and_shadowed().items()
        if imported & parameters
    }
    assert not shadowing, (
        "These modules import a settings slice global and also name a parameter after "
        f"it, so inside those functions the global is unreachable: {shadowing}. Rename "
        "the parameter or read the global through its module."
    )


def test_a_subclass_without_a_prefix_reads_no_bare_underscore_variable(monkeypatch):
    """Doubling an empty prefix would make ``_FIELD`` a live environment name.

    A subclass that declares no ``env_prefix`` must keep the declared sources
    alone, so an unrelated ``_MODE`` in the environment cannot reach it.
    """

    class Prefixless(PackageSettings):
        mode: str = "cloud"

    monkeypatch.setenv("_MODE", "local")
    assert Prefixless().mode == "cloud"
