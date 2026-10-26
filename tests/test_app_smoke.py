from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_starts_without_errors():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("Upload PDFs" in info.value for info in at.info)


def test_build_without_files_shows_error():
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.text_input[0].set_value("dummy-key")  # API key field
    at.button[0].click().run()
    assert not at.exception
    assert any("Upload at least one PDF" in e.value for e in at.error)
