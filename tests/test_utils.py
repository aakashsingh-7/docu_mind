import pytest

from utils import validate_pdf_upload


class Upload:
    def __init__(self, name, data):
        self.name = name
        self._data = data

    def getvalue(self):
        return self._data


def test_valid_pdf_passes():
    validate_pdf_upload(Upload("a.pdf", b"%PDF-1.4 content"))


def test_none_rejected():
    with pytest.raises(ValueError):
        validate_pdf_upload(None)


def test_wrong_extension_rejected():
    with pytest.raises(ValueError, match="not a PDF"):
        validate_pdf_upload(Upload("a.txt", b"%PDF-1.4"))


def test_bad_header_rejected():
    with pytest.raises(ValueError, match="valid PDF"):
        validate_pdf_upload(Upload("a.pdf", b"hello"))


def test_empty_file_rejected():
    with pytest.raises(ValueError):
        validate_pdf_upload(Upload("a.pdf", b""))
