import pytest

from app.core.loopback import is_loopback_url


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:11434",
        "https://LOCALHOST",
        "http://127.0.0.1:11434",
        "http://127.8.9.10",
        "http://[::1]:11434",
    ],
)
def test_a_url_on_this_computer_is_loopback(url):
    assert is_loopback_url(url) is True


@pytest.mark.parametrize(
    "url",
    [
        "",
        "localhost:11434",
        "http://evil.example",
        "http://localhost@evil.example",
        "http://127.0.0.1.evil.example",
        "http://10.0.0.5:11434",
        "http://0.0.0.0:11434",
        "ftp://localhost",
    ],
)
def test_any_other_url_is_not(url):
    assert is_loopback_url(url) is False
