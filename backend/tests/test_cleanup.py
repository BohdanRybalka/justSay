"""Dictation cleanup: the guard on real model answers, and every way back to the transcript."""

from unittest.mock import MagicMock, patch

import pytest

from app.core.types import ProviderMode
from app.pipeline import cleanup
from app.pipeline.cleanup import clean_dictation, is_faithful_cleanup
from app.stt.config import STTSettings

UNPATCHED_CLIENT = cleanup._client

SELF_CORRECTION = (
    "Ну е треба зробити, ну, звіт до п'ятниці, ні, до четверга, і еее надіслати його Олені, "
    "ну типу як би в пошту."
)
SELF_CORRECTION_KEPT = (
    "Треба зробити звіт до п'ятниці, ні, до четверга, і надіслати його Олені, як би в пошту."
)
SELF_CORRECTION_DROPPED = "Треба зробити звіт до четверга і надіслати його Олені в пошту."

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


def test_fillers_and_hesitations_may_go_and_the_self_correction_stays():
    assert is_faithful_cleanup(SELF_CORRECTION, SELF_CORRECTION_KEPT)


def test_the_reported_self_correction_is_kept_whole():
    raw = "ну, е, зробимо до п'ятниці, ні, до четверга"
    assert is_faithful_cleanup(raw, "Зробимо до п'ятниці, ні, до четверга.")
    assert not is_faithful_cleanup(raw, "Зробимо до четверга.")


@pytest.mark.parametrize(
    "raw, answer",
    [
        (SELF_CORRECTION, SELF_CORRECTION_DROPPED),
        ("Завтра, ні, післязавтра", "Післязавтра."),
        ("о п'ятій, ні, о шостій", "О шостій."),
        ("я не можу, ой, я можу прийти завтра", "Я можу прийти завтра."),
        ("я я я думаю думаю що що так так", "Я думаю, що так."),
        ("я не не знаю що сказати", "Я не знаю, що сказати."),
    ],
)
def test_a_dropped_self_correction_or_repeat_is_caught(raw, answer):
    assert not is_faithful_cleanup(raw, answer)


@pytest.mark.parametrize(
    "raw, answer",
    [
        ("I do not think so", "I don’t think so."),
        ("нема часу", "Немає часу."),
        ("подай репорт зрозумілій мові", "Подай репорт зрозумілою мовою."),
        ("завтра я піду в магазин", "Я піду в магазин завтра."),
    ],
)
def test_a_reworded_or_reordered_dictation_is_caught(raw, answer):
    assert not is_faithful_cleanup(raw, answer)


def test_a_deleted_sentence_is_caught():
    assert not is_faithful_cleanup(LONG_DICTATION, LONG_DICTATION_SENTENCE_DELETED)


def test_punctuation_and_case_alone_pass():
    assert is_faithful_cleanup(LONG_DICTATION, LONG_DICTATION.upper().replace(",", ""))


def test_any_apostrophe_spelling_passes():
    assert is_faithful_cleanup("до п'ятниці", "До пʼятниці.")
    assert is_faithful_cleanup("до п’ятниці", "До п'ятниці.")
    assert is_faithful_cleanup("до п‘ятниці", "До п'ятниці.")


def test_a_sentence_deleted_from_a_long_dictation_is_caught_though_the_rest_is_kept():
    kept = " ".join(f"слово{index}" for index in range(120))
    sentence = "тому що коли ти мені пишеш це дуже важко прочитати"
    assert is_faithful_cleanup(f"{kept} {sentence}", f"{kept} {sentence}")
    assert not is_faithful_cleanup(f"{kept} {sentence}", kept)


@pytest.mark.parametrize(
    "raw, answer",
    [
        ("я не хочу", "Я Node хочу."),
        ("запусти тести в плей райт", "Запусти тести в Playwright."),
        ("спитай у чат джіпіті", "Спитай у чат GPT."),
        ("тести", "Тести Playwright."),
    ],
)
def test_no_word_may_be_replaced_or_added(raw, answer):
    assert not is_faithful_cleanup(raw, answer)


@pytest.mark.parametrize(
    "raw, answer",
    [
        ("як би ти хотів", "Ти хотів?"),
        ("я як би не знаю", "Я не знаю."),
        ("Це було як би добре", "Це було як добре."),
        ("Це було як би добре", "Це було би добре."),
    ],
)
def test_as_if_is_never_dropped(raw, answer):
    assert not is_faithful_cleanup(raw, answer)


@pytest.mark.parametrize(
    "raw, answer",
    [
        ("е-е, ееее, еммм, ну, типу, зробимо", "Зробимо."),
        ("Е, зробимо. Ем, завтра.", "Зробимо. Завтра."),
    ],
)
def test_fillers_and_drawn_out_hesitations_may_go(raw, answer):
    assert is_faithful_cleanup(raw, answer)


@pytest.mark.parametrize(
    "raw, answer",
    [
        ("товщина 5 мм", "Товщина 5."),
        ("відстань 5 м від дому", "Відстань 5 від дому."),
        ("вітамін Е", "Вітамін."),
        ("зробимо, Е, завтра", "Зробимо, завтра."),
        ("ЕМ працює", "Працює."),
        ("I err on the side of caution", "I on the side of caution."),
        ("ммм, добре", "Добре."),
        ("um, okay", "Okay."),
        ("відсоток 5%", "Відсоток 5."),
        ("ціна 100 ₴", "Ціна 100."),
    ],
)
def test_a_unit_letter_symbol_or_other_sound_may_not_go(raw, answer):
    assert not is_faithful_cleanup(raw, answer)


def test_an_answer_with_no_word_left_is_caught():
    assert not is_faithful_cleanup("ну", ".")


def test_a_dictionary_term_the_speaker_said_may_stay():
    assert is_faithful_cleanup("ну запусти Playwright", "Запусти Playwright.")


@pytest.mark.parametrize(
    "raw, answer",
    [
        ("turn right at the lights", "Turn at the lights."),
        ("я думаю, ось файл", "Я думаю, файл."),
    ],
)
def test_a_word_that_is_a_filler_only_sometimes_may_not_go(raw, answer):
    assert not is_faithful_cleanup(raw, answer)


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


def test_scattered_deletions_that_summarise_are_caught():
    raw = (
        "сьогодні зранку я поїхав на роботу потім зустрівся з командою обговорили план "
        "на тиждень а ввечері повернувся додому і подивився фільм"
    )
    summary = "Сьогодні поїхав на роботу, обговорили план, ввечері подивився фільм."
    assert not is_faithful_cleanup(raw, summary)


def test_the_prompt_names_only_the_removable_fillers_and_keeps_self_corrections():
    prompt = cleanup._SYSTEM_PROMPT
    assert f"filler words: {', '.join(cleanup.CLEANUP_FILLERS)};" in prompt
    assert "remove the hesitation sounds е and ем," in prompt
    assert "um, uh" not in prompt
    assert "як би" not in prompt
    assert "right" not in prompt
    assert "including self-corrections" in prompt
    assert "keep only the corrected version" not in prompt


async def test_a_faithful_answer_replaces_the_transcript():
    with patch.object(cleanup, "_call_groq", return_value=SELF_CORRECTION_KEPT) as call:
        text = await clean_dictation(SELF_CORRECTION, _settings(initial_prompt="Kubernetes"))

    assert text == SELF_CORRECTION_KEPT
    api_key, model, system, transcript = call.call_args.args
    assert (api_key, model, transcript) == ("test-key", "openai/gpt-oss-20b", SELF_CORRECTION)
    assert system == cleanup._SYSTEM_PROMPT
    assert "Kubernetes" not in system


@pytest.mark.parametrize(
    "raw, answer",
    [("як би ти хотів", "Ти хотів?"), ("я як би не знаю", "Я не знаю.")],
)
async def test_an_answer_that_drops_as_if_keeps_the_transcript(raw, answer):
    with patch.object(cleanup, "_call_groq", return_value=answer):
        assert await clean_dictation(raw, _settings()) == raw


async def test_an_answer_that_swaps_a_word_for_a_dictionary_term_keeps_the_transcript():
    with patch.object(cleanup, "_call_groq", return_value="Я Node хочу."):
        text = await clean_dictation("я не хочу", _settings(initial_prompt="Node, Kubernetes"))
    assert text == "я не хочу"


@pytest.mark.parametrize(
    "raw, answer, text",
    [
        (
            "Ну, е, зробимо до п'ятниці, ні, до четверга.",
            "зробимо до п'ятниці, ні, до четверга.",
            "Зробимо до п'ятниці, ні, до четверга.",
        ),
        ("Готово. Ну, е, відправ звіт.", "Готово. відправ звіт.", "Готово. Відправ звіт."),
        ("Ну, е, як би піду.", "як би піду.", "Як би піду."),
        ("Ну, зробимо, ну, гаразд.", "зробимо. гаразд.", "Зробимо. гаразд."),
        ("Готово? Ну, відправ.", "Готово? відправ.", "Готово? Відправ."),
        ("Готово! Ну, відправ.", "Готово! відправ.", "Готово! Відправ."),
        ("Готово… Ну, відправ.", "Готово… відправ.", "Готово… Відправ."),
        ("Готово\nНу, відправ.", "Готово\nвідправ.", "Готово\nВідправ."),
        ("Ну... зробимо", "зробимо", "Зробимо"),
        ("Ну, е. зробимо", "зробимо.", "Зробимо."),
        ("Ну ну зробимо", "ну зробимо", "Ну зробимо"),
    ],
)
async def test_a_sentence_whose_leading_filler_went_gets_its_capital_back(raw, answer, text):
    with patch.object(cleanup, "_call_groq", return_value=answer):
        assert await clean_dictation(raw, _settings()) == text


@pytest.mark.parametrize(
    "raw, answer",
    [
        ("ну, е, зробимо до п'ятниці, ні, до четверга", "зробимо до п'ятниці, ні, до четверга"),
        ("Так. Ну, зробимо.", "Так, зробимо."),
        ("Ну, зробимо.", "ну, зробимо."),
        ("Так. зробимо.", "Так. зробимо."),
        ("Ну, iPhone зламався.", "iPhone зламався."),
        ("Ну, eBay працює.", "eBay працює."),
    ],
)
async def test_no_other_letter_changes_case(raw, answer):
    with patch.object(cleanup, "_call_groq", return_value=answer):
        assert await clean_dictation(raw, _settings()) == answer


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


async def test_a_dropped_self_correction_keeps_the_transcript():
    with patch.object(cleanup, "_call_groq", return_value=SELF_CORRECTION_DROPPED):
        assert await clean_dictation(SELF_CORRECTION, _settings()) == SELF_CORRECTION


async def test_an_empty_answer_keeps_even_a_transcript_of_fillers_alone():
    with patch.object(cleanup, "_call_groq", return_value=""):
        assert await clean_dictation("Ну е", _settings()) == "Ну е"


async def test_an_answer_with_no_word_left_keeps_the_transcript():
    with patch.object(cleanup, "_call_groq", return_value="."):
        assert await clean_dictation("Ну", _settings()) == "Ну"


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
