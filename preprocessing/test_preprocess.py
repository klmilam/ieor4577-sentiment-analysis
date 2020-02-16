"""Test file for preprocess.py"""

from preprocessing import preprocess

def clean_text(text):
    """Calls clean_text method for testing."""
    return preprocess.PreprocessTweets(text).clean_text()


def test_clean_text():
    """Tests clean_text function."""
    input_text = "@my_handler here is my tweet http://www.google.com got it?"
    output_text = "here is my tweet got it?"
    assert clean_text(input_text) == output_text


def tokenize_text(text):
    """Calls tokenize_text method for testing."""
    return preprocess.PreprocessTweets(text).tokenize_text()


def test_tokenize_text():
    """Tests tokenize_text function."""
    input_text = "here is my tweet got it?"
    output_text = ["here", "is", "my", "tweet", "got", "it", "?"]
    assert tokenize_text(input_text) == output_text
