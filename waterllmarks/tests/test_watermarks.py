"""Tests for the watermarking algorithms."""

import pytest

from waterllmarks.watermarks import Rizzo2016, TextWatermark, TokenWatermark


def _test_watermark(algorithm: TextWatermark):
    """Test a watermarking algorithm."""
    text = "Digital watermarking has become crucially important in authentication and copyright protection of the digital contents, since more and more data are daily generated and shared online through digital archives, blogs and social networks. Out of all, text watermarking is a more difficult task in comparison to other media watermarking. Text cannot be always converted into image, it accounts for a far smaller amount of data (eg. social network posts) and the changes in short texts would strongly affect the meaning or the overall visual form. In this paper we propose a text watermarking technique based on homoglyph characters substitution for latin symbols1. The proposed method is able to efficiently embed a password based watermark in short texts by strictly preserving the content. In particular, it uses alternative Unicode symbols to ensure visual indistinguishability and length preservation, namely content-preservation. To evaluate our method, we use a real dataset of 1.8 million New York articles. The results show the effectiveness of our approach providing an average length of 101 characters needed to embed a 64bit password based watermark."
    watermarked_text = algorithm.apply(text)
    assert algorithm.check(watermarked_text)
    assert algorithm.remove(watermarked_text) == text


def test_text_watermark():
    """Test the TextWatermark base class."""
    algorithm = TextWatermark()
    with pytest.raises(NotImplementedError):
        algorithm.apply("test")
    with pytest.raises(NotImplementedError):
        algorithm.remove("test")
    with pytest.raises(NotImplementedError):
        algorithm.check("test")


def test_rizzo2016():
    """Test the Rizzo2016 watermarking algorithm."""
    with pytest.raises(TypeError):
        algorithm = Rizzo2016()
    with pytest.raises(TypeError):
        algorithm = Rizzo2016("test")
    algorithm = Rizzo2016(b"0123456789ABCDEF")
    _test_watermark(algorithm)


def test_tokenwatermark():
    """Test the TokenWatermark algorithm."""
    algorithm = TokenWatermark(b"0123456789ABCDEF")
    _test_watermark(algorithm)
