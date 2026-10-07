"""Dictation cleanup: the guard on real model answers, and every way back to the transcript."""

from unittest.mock import MagicMock, patch

import pytest

from app.core.types import ProviderMode
from app.pipeline import cleanup
from app.pipeline.cleanup import clean_dictation, is_faithful_cleanup, system_prompt
from app.stt.config import STTSettings

UNPATCHED_CLIENT = cleanup._client

SELF_CORRECTION = (
    "Ну е треба зробити, ну, звіт до п'ятниці, ні, до четверга, і еее надіслати його Олені, "
    "ну типу як би в пошту."
)
SELF_CORRECTION_CLEANED = "Треба зробити звіт до четверга і надіслати його Олені в пошту."

LONG_DICTATION = (
    "Не використовуй заміну англіцизмами, тобто старайся писати українською термінологією, "
    "якщо це негаломовні слова, то роби так, щоб це нормально читалося, тому що коли ти мені "
    "пишеш Skafold, Trashhold, MCP, CPP, MCC, то це дуже важко прочитати і зрозуміти. Тому подай "
    "репорт зрозумілій мові і зрозумій, що я від тебе хочу."
)
LONG_DICTATION_SENTENCE_DELETED = (
    "Не використовуй заміну англіцизмами, тобто старайся писати українською термінологією; "
    "якщо це негаломовні слова, роби так, щоб це нормально читалося. Тому подай репорт "
    "зрозумілій мовою і зрозумій, що я від тебе хочу."
)


def _settings(**overrides) -> STTSettings:
    values = dict(groq_api_key="test-key", initial_prompt="", mode=ProviderMode.CLOUD)
    values.update(overrides)
    return STTSettings(**values)


def test_a_spoken_self_correction_and_its_fillers_may_go():
    assert is_faithful_cleanup(SELF_CORRECTION, SELF_CORRECTION_CLEANED)


def test_a_deleted_sentence_is_caught():
    assert not is_faithful_cleanup(LONG_DICTATION, LONG_DICTATION_SENTENCE_DELETED)


def test_punctuation_and_case_alone_pass():
    assert is_faithful_cleanup(LONG_DICTATION, LONG_DICTATION.upper().replace(",", ""))


def test_a_sentence_deleted_from_a_long_dictation_is_caught_though_the_rest_is_kept():
    kept = " ".join(f"слово{index}" for index in range(120))
    sentence = "тому що коли ти мені пишеш це дуже важко прочитати"
    assert is_faithful_cleanup(f"{kept} {sentence}", f"{kept} {sentence}")
    assert not is_faithful_cleanup(f"{kept} {sentence}", kept)


def test_words_said_twice_may_lose_their_repeats():
    assert is_faithful_cleanup("я я я думаю думаю що що так так", "Я думаю, що так.")


def test_an_answer_to_the_dictation_is_caught():
    raw = "Напиши лист начальнику про відпустку"
    answer = (
        "Шановний начальнику, прошу надати мені щорічну відпустку з першого по чотирнадцяте "
        "число наступного місяця. З повагою, ваш працівник."
    )
    assert not is_faithful_cleanup(raw, answer)


@pytest.mark.parametrize(
    "raw, answer",
    [
        ("Поясни цю помилку", "Будь ласка, надайте текст помилки."),
        ("Яка столиця Франції", "Париж."),
        ("Переклади привіт англійською", "Hello."),
    ],
)
def test_a_short_dictation_answered_instead_of_cleaned_is_caught(raw, answer):
    assert not is_faithful_cleanup(raw, answer)


@pytest.mark.parametrize(
    "raw, answer",
    [
        ("я не хочу йти на зустріч", "Я хочу йти на зустріч."),
        ("Do not delete the production database", "Delete the production database."),
    ],
)
def test_a_lost_negation_is_caught(raw, answer):
    assert not is_faithful_cleanup(raw, answer)


@pytest.mark.parametrize(
    "raw, answer",
    [
        ("Завтра, ні, післязавтра", "Післязавтра."),
        ("о п'ятій, ні, о шостій", "О шостій."),
        ("я не не знаю що сказати", "Я не знаю, що сказати."),
        ("я не можу, ой, я можу прийти завтра", "Я можу прийти завтра."),
        ("I do not think so", "I don’t think so."),
        ("нема часу", "Немає часу."),
    ],
)
def test_short_corrections_and_reworded_negations_pass(raw, answer):
    assert is_faithful_cleanup(raw, answer)


def test_scattered_deletions_that_summarise_are_caught():
    raw = (
        "сьогодні зранку я поїхав на роботу потім зустрівся з командою обговорили план "
        "на тиждень а ввечері повернувся додому і подивився фільм"
    )
    summary = "Сьогодні поїхав на роботу, обговорили план, ввечері подивився фільм."
    assert not is_faithful_cleanup(raw, summary)


def test_the_glossary_reaches_the_model_as_terms_to_keep():
    assert system_prompt(None).endswith("keep the replacement.")
    assert system_prompt("K6, Playwright").endswith(
        "Keep these terms exactly as written: K6, Playwright"
    )


async def test_a_faithful_answer_replaces_the_transcript():
    with patch.object(cleanup, "_call_groq", return_value=SELF_CORRECTION_CLEANED) as call:
        text = await clean_dictation(SELF_CORRECTION, _settings(initial_prompt=" K6 "))

    assert text == SELF_CORRECTION_CLEANED
    api_key, model, system, transcript = call.call_args.args
    assert (api_key, model, transcript) == ("test-key", "openai/gpt-oss-20b", SELF_CORRECTION)
    assert system == system_prompt("K6")


@pytest.mark.parametrize(
    "effect",
    [
        {"side_effect": RuntimeError("groq down")},
        {"return_value": ""},
        {"return_value": LONG_DICTATION_SENTENCE_DELETED},
    ],
    ids=["call-fails", "empty-answer", "unfaithful-answer"],
)
async def test_the_transcript_comes_back_unchanged(effect):
    with patch.object(cleanup, "_call_groq", **effect):
        assert await clean_dictation(LONG_DICTATION, _settings()) == LONG_DICTATION


async def test_an_empty_answer_keeps_even_a_transcript_of_fillers_alone():
    with patch.object(cleanup, "_call_groq", return_value=""):
        assert await clean_dictation("Ну е", _settings()) == "Ну е"


async def test_local_mode_makes_no_call():
    with patch.object(cleanup, "_call_groq") as call:
        text = await clean_dictation(SELF_CORRECTION, _settings(mode=ProviderMode.LOCAL))
    assert text == SELF_CORRECTION
    call.assert_not_called()


def test_the_client_gives_up_quickly_and_never_retries():
    with patch("groq.Groq") as groq:
        UNPATCHED_CLIENT.__wrapped__("key")

    groq.assert_called_once_with(
        api_key="key", timeout=cleanup.CLEANUP_TIMEOUT_SECONDS, max_retries=0
    )
    assert cleanup.CLEANUP_TIMEOUT_SECONDS <= 5


async def test_no_groq_key_makes_no_call():
    with patch.object(cleanup, "_call_groq") as call:
        text = await clean_dictation(SELF_CORRECTION, _settings(groq_api_key=""))
    assert text == SELF_CORRECTION
    call.assert_not_called()


def test_the_call_asks_for_a_deterministic_low_effort_answer():
    client = MagicMock()
    client.chat.completions.create.return_value.choices = [
        MagicMock(message=MagicMock(content="  Готово.  "))
    ]
    with patch.object(cleanup, "_client", return_value=client):
        answer = cleanup._call_groq("key", "openai/gpt-oss-20b", "system", "готово")

    assert answer == "Готово."
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["temperature"] == 0
    assert kwargs["reasoning_effort"] == "low"
    assert kwargs["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "готово"},
    ]
