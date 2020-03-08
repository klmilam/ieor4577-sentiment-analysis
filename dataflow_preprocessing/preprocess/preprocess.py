"""Preprocessing library to clean and embed tweets"""

import json
import os
from urllib.parse import urlparse
import zipfile

from nltk.tokenize import TweetTokenizer


class PreprocessTweets():
    """Library to perform preprocessing for sentiment analysis on tweets."""
    #pylint: disable=too-many-arguments
    def __init__(self, input_data, max_length_dictionary=None,
                 max_length_tweet=None, token_indices_json=None,
                 embedding=None):
        self.input = input_data
        self.max_length_dictionary = max_length_dictionary
        self.max_length_tweet = max_length_tweet
        self.token_indices_json = token_indices_json
        self.embedding = embedding
        if token_indices_json:
            self.load_embedding_dictionary()


    def clean_text(self):
        """Cleans text by removing URLs, hashtags, and usernames."""
        if not isinstance(self.input, str):
            raise TypeError('Input to tokenize_text must be a string.')

        cleaned_tweet = ""
        for word in self.input.split(" "):
            if word.startswith("@") or word.startswith("#"):
                continue
            try:
                result = urlparse(word)
                if all([result.scheme, result.netloc]):
                    continue
                cleaned_tweet += word.lower() + " "
            except:
                pass
        return cleaned_tweet.strip()


    def load_embedding_dictionary(self):
        """Loads embedding dictionary from zipped local file."""
        if self.token_indices_json and ".zip/" in self.token_indices_json:
            archive_path = os.path.abspath(self.token_indices_json)
            split = archive_path.split(".zip")
            split_orig = self.token_indices_json.split(".zip")
            archive_path = split[0] + ".zip"
            path_inside = split_orig[0].split("/")[-1] + split_orig[1]
            archive = zipfile.ZipFile(archive_path, "r")
            self.embedding = json.loads(archive.read(path_inside).decode("utf-8"))
        else:
            with open(self.token_indices_json) as file:
                self.embedding = json.load(file)
        return self.embedding

    def tokenize_text(self):
        """Tokenizes tweet into an array of tokens."""
        if not isinstance(self.input, str):
            raise TypeError('Input to tokenize_text must be a string.')
        tknzr = TweetTokenizer()
        return tknzr.tokenize(self.input)


    def replace_token_with_index(self):
        """Replaces each token with its index in the embedding dictionary."""
        if not isinstance(self.input, list):
            raise TypeError('Input to replace_token_with_index must be a list.')

        tokens = []

        for word in self.input:
            if word in self.embedding:
                index = self.embedding[word]
            else:
                index = 1
            max_length = self.max_length_dictionary
            if not max_length or index < max_length:
                tokens.append(index)
        return tokens


    def pad_sequence(self):
        """Pads and slices list of indices."""
        if not isinstance(self.input, list):
            raise TypeError('Input to pad_sequence must be a list.')

        tokens = self.input[:self.max_length_tweet]
        tokens = tokens + [0]*(self.max_length_tweet-len(tokens))
        return tokens


def run_pipeline(input_data, max_length_tweet, embedding=None,
                 max_length_dictionary=None):
    """Runs pipeline."""
    cleaned_text = PreprocessTweets(input_data).clean_text()
    tokenized_text = PreprocessTweets(cleaned_text).tokenize_text()
    text_indices = PreprocessTweets(
        tokenized_text,
        max_length_dictionary=max_length_dictionary,
        embedding=embedding).replace_token_with_index()
    padded_indices = PreprocessTweets(
        text_indices, max_length_tweet=max_length_tweet).pad_sequence()
    return padded_indices
