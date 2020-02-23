"""Test file for preprocess.py"""

import pytest

from preprocessing import preprocess

def clean_text(text):
    """Calls clean_text method for testing."""
    return preprocess.PreprocessTweets(text).clean_text()


def test_clean_text():
    """Tests clean_text function."""
    input_text = "@my_handler Here is my tweet http://www.google.com got it?"
    output_text = "here is my tweet got it?"
    assert clean_text(input_text) == output_text


def test_clean_text_with_list():
    """Tests tokenize_text function with list input."""
    input_list = ["here", "is", "my", "tweet", "got", "it", "?"]
    with pytest.raises(TypeError):
        clean_text(input_list)


def tokenize_text(text):
    """Calls tokenize_text method for testing."""
    return preprocess.PreprocessTweets(text).tokenize_text()


def test_tokenize_text():
    """Tests tokenize_text function."""
    input_text = "here is my tweet got it?"
    output_text = ["here", "is", "my", "tweet", "got", "it", "?"]
    assert tokenize_text(input_text) == output_text


def test_tokenize_text_with_list():
    """Tests tokenize_text function with list input."""
    input_list = ["here", "is", "my", "tweet", "got", "it", "?"]
    with pytest.raises(TypeError):
        tokenize_text(input_list)


def replace_token_with_index(tokens, input_json):
    """Calls replace_token_with_index method for testing."""
    return preprocess.PreprocessTweets(
        tokens, token_indices_json=input_json).replace_token_with_index()


def test_replace_token_with_index():
    """Tests replace_token_with_index function."""
    input_tokens = ["here", "is", "my", "tweet", "got", "it", "?"]
    output_indices = [229, 32, 29, 274, 143, 33, 14]
    input_json = "preprocessing/token_indices.json"
    assert replace_token_with_index(
        input_tokens, input_json) == output_indices


def test_replace_token_with_index_with_string():
    """Tests replace_token_with_index function with String input."""
    input_string = "here is my tweet got it?"
    input_json = "preprocessing/artifacts/token_indices.json"
    with pytest.raises(TypeError):
        replace_token_with_index(input_string, input_json)


def test_replace_token_with_index_with_unknown_token():
    """Tests replace_token_with_index function with an unknown token."""
    input_tokens = ["hereis", "is", "my", "tweet", "got", "it", "?"]
    output_indices = [1, 32, 29, 274, 143, 33, 14]
    input_json = "preprocessing/token_indices.json"
    assert replace_token_with_index(
        input_tokens, input_json) == output_indices



def test_replace_token_with_index_with_zip():
    """Tests replace_token_with_index with a zipped file."""
    input_tokens = ["here", "is", "my", "tweet", "got", "it", "?"]
    output_indices = [229, 32, 29, 274, 143, 33, 14]
    input_json = "preprocessing/artifacts.zip/token_indices.json"
    assert replace_token_with_index(
        input_tokens, input_json) == output_indices


def pad_sequence(indices, size):
    """Calls pad_sequence method for testing."""
    return preprocess.PreprocessTweets(
        indices, max_length_tweet=size).pad_sequence()


def test_pad_sequence_add_padding():
    """Tests pad_sequence function for too short input."""
    input_indices = [229, 32, 29, 274, 143, 33, 14]
    output_indices = [229, 32, 29, 274, 143, 33, 14, 0, 0, 0]
    assert pad_sequence(input_indices, 10) == output_indices


def test_pad_sequence_slice():
    """Tests pad_sequence function for too long input."""
    input_indices = [229, 32, 29, 274, 143, 33, 14]
    output_indices = [229, 32, 29, 274, 143]
    assert pad_sequence(input_indices, 5) == output_indices


def test_pad_sequence_with_string():
    """Tests replace_token_with_index function with String input."""
    input_string = "here is my tweet got it?"
    with pytest.raises(TypeError):
        pad_sequence(input_string, 10)


def run_pipeline(input_text, max_length_tweet=10):
    """Calls run_pipeline method for testing."""
    input_json = "preprocessing/token_indices.json"
    return preprocess.run_pipeline(
        input_text, max_length_tweet, input_json)


def test_run_pipeline():
    """Tests run_pipeline function."""
    input_text = "@my_handler Here is my tweet http://www.google.com got it?"
    output_indices = [229, 32, 29, 274, 143, 33, 14, 0, 0, 0]
    assert run_pipeline(input_text) == output_indices
