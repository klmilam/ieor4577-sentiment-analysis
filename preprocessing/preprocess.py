"""Preprocessing library to clean and embed tweets"""

from urllib.parse import urlparse

from nltk.tokenize import TweetTokenizer
import tensorflow as tf

class PreprocessTweets():
    """Library to perform preprocessing for sentiment analysis on tweets."""
    def __init__(self, text, max_length_dictionary=None, max_length_tweet=None):
        self.text = text
        self.max_length_dictionary = max_length_dictionary
        self.max_length_tweet = max_length_tweet


    def clean_text(self):
        """Cleans text by removing URLs, hashtags, and usernames."""
        cleaned_tweet = ""
        for word in self.text.split(" "):
            if word.startswith("@") or word.startswith("#"):
                continue
            result = urlparse(word)
            if all([result.scheme, result.netloc]):
                continue
            cleaned_tweet += word + " "
        return cleaned_tweet.strip()


    def tokenize_text(self):
        """Tokenizes tweet into an array of tokens.

        Args:
            tweet: a cleaned string.

        Output:
            An array of ints (tokens).
        """
        tknzr = TweetTokenizer()
        return tknzr.tokenize(self.text)
