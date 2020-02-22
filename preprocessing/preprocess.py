"""Preprocessing library to clean and embed tweets"""

import json
import os
from urllib.parse import urlparse
import zipfile

from nltk.tokenize import TweetTokenizer
import tensorflow as tf

class PreprocessTweets():
    """Library to perform preprocessing for sentiment analysis on tweets."""
    def __init__(self, input_data, max_length_dictionary=None,
                 max_length_tweet=None, token_indices_json=None):
        self.input = input_data
        self.max_length_dictionary = max_length_dictionary
        self.max_length_tweet = max_length_tweet
        self.token_indices_json = token_indices_json


    def clean_text(self):
        """Cleans text by removing URLs, hashtags, and usernames."""
        if not isinstance(self.input, str):
            raise TypeError('Input to tokenize_text must be a string.')

        cleaned_tweet = ""
        for word in self.input.split(" "):
            if word.startswith("@") or word.startswith("#"):
                continue
            result = urlparse(word)
            if all([result.scheme, result.netloc]):
                continue
            cleaned_tweet += word.lower() + " "
        return cleaned_tweet.strip()


    def load_embedding_dictionary(self):
        """Loads embedding dictionary from zipped local file."""
        if ".zip/" in self.token_indices_json:
            archive_path = os.path.abspath(self.token_indices_json)
            split = archive_path.split(".zip/")
            archive_path = split[0] + ".zip"
            path_inside = split[1]
            archive = zipfile.ZipFile(archive_path, "r")
            embedding = json.loads(archive.read(path_inside).decode("utf8"))
        else:
            with tf.io.gfile.GFile(self.token_indices_json) as file:
                embedding = json.load(file)
        return embedding


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
        embedding = self.load_embedding_dictionary()

        for word in self.input:
            index = embedding[word]
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
