"""The listed phrases, sound labels and the marks touching them leave a transcript; no word does."""

import time

import pytest

from app.pipeline.invented_phrases import (
    INVENTED_PHRASES,
    MAX_WORDS_PER_VOICED_SECOND,
    outpaces_speech,
    without_invented_phrases,
)


@pytest.mark.parametrize("phrase", INVENTED_PHRASES)
def test_every_listed_phrase_alone_leaves_nothing(phrase):
    assert without_invented_phrases(phrase) == ""


@pytest.mark.parametrize(
    "text",
    [
        "Дякую за перегляд!",
        "ДЯКУЮ ЗА ПЕРЕГЛЯД...",
        "Thanks for watching.",
        "thanks, for watching!",
        "Subtitles by the Amara.org community",
        "Субтитри зроблені спільнотою Amara.org",
        "Субтитры создавал DimaTorzok",
        "субтитры сделал dimatorzok.",
        "[silence]",
        "[BLANK_AUDIO]",
        "[Музика]",
        "♪♪",
        "♫♫",
        "(Дякую за перегляд)",
        "[Music][Applause]",
        "Дякую за перегляд ...",
        "- Thanks for watching! -",
        "Thanks for watching __",
        "Дякую за перегляд. Дякую за перегляд. Дякую за перегляд.",
        "[Music] Thank you for watching!",
    ],
)
def test_a_phrase_leaves_in_any_case_with_any_punctuation_and_when_repeated(text):
    assert without_invented_phrases(text) == ""


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Привіт. Дякую за перегляд. Бувай.", "Привіт. Бувай."),
        ("Привіт, дякую за перегляд, бувай", "Привіт, бувай"),
        ("Так [inaudible] добре", "Так добре"),
        ("Дякую за перегляд, підпишіться на канал", "підпишіться на канал"),
        ("Привіт(Дякую за перегляд)Бувай", "Привіт Бувай"),
        ("(Thanks for watching)Привіт", "Привіт"),
        ("[Music] Так ... ні", "Так ... ні"),
        ("[Music] Ціна 5 – 10 грн", "Ціна 5 – 10 грн"),
        ("♪ 2 + 2 = 4", "2 + 2 = 4"),
        ("Дякую за перегляд. Ціна 50 % і 2", "Ціна 50 % і 2"),
    ],
)
def test_the_words_around_a_phrase_stay_with_the_spaces_closed_up(text, expected):
    assert without_invented_phrases(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Thank you.",
        "Дякую.",
        "Дякую за увагу",
        "Дякую за переглядання",
        "вдякую за перегляд",
        "Мені (тихо) сказали  це  двічі\n",
        "",
    ],
)
def test_text_without_a_listed_phrase_comes_back_exactly_as_it_was(text):
    assert without_invented_phrases(text) == text


def test_a_long_run_of_marks_is_cut_in_a_blink_not_in_seconds():
    text = "-" * 10000 + " ні Thanks for watching"

    started = time.perf_counter()
    result = without_invented_phrases(text)

    assert result == "-" * 10000 + " ні"
    assert time.perf_counter() - started < 1.0


@pytest.mark.parametrize(
    ("text", "voiced"),
    [
        ("and I'll see you next time.", 0.1),
        ("Thank you.", 0.1),
        ("Дякую за перегляд!", 0.08),
        ("один два три чотири п'ять шість сім вісім", 0.9),
        ("слово інше", 0.0),
    ],
)
def test_more_words_than_a_voice_can_say_in_the_voice_heard_are_made_up(text, voiced):
    assert outpaces_speech(text, voiced)


@pytest.mark.parametrize(
    ("text", "voiced"),
    [
        ("Дякую!", 0.1),
        ("— Так.", 0.2),
        ("... моделями ...", 0.0),
        ("моделями.", 0.0),
        ("Так, добре", 0.6),
        (" ".join(["слово"] * 34), 11.6),
        ("", 0.0),
    ],
)
def test_a_single_word_and_a_real_pace_are_not_made_up(text, voiced):
    assert not outpaces_speech(text, voiced)


def test_the_pace_limit_is_a_word_in_less_than_a_hundred_and_thirty_three_milliseconds():
    assert MAX_WORDS_PER_VOICED_SECOND * 0.5 == pytest.approx(3.76, abs=0.01)
    assert not outpaces_speech(" ".join(["слово"] * 3), 0.5)
    assert outpaces_speech(" ".join(["слово"] * 4), 0.5)


def test_a_sound_with_a_second_of_voice_or_more_is_never_judged_by_its_pace():
    eight = " ".join(["слово"] * 8)

    assert outpaces_speech(eight, 0.99)
    assert not outpaces_speech(eight, 1.0)
    assert not outpaces_speech(" ".join(["слово"] * 102), 13.0)
