"""Unit tests for the text-cleaning functions. Run from the repository folder: python -m unittest discover tests"""
import os
import sys
import tempfile
import unittest

# These settings stop the import from installing packages or writing to Kaggle's folders
os.environ["MVSA_INSTALL_EXTRAS"] = "0"
os.environ["MVSA_OUT"] = tempfile.mkdtemp()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mvsa.preprocessing import clean_text, segment_hashtag  # noqa: E402


class CleanTextTests(unittest.TestCase):
    def test_removes_urls(self):
        self.assertEqual(clean_text("nice day https://t.co/abc"), "nice day")

    def test_replaces_mentions(self):
        self.assertEqual(clean_text("thanks @someone"), "thanks @user")

    def test_removes_retweet_prefix(self):
        self.assertEqual(clean_text("RT @someone: nice day"), "nice day")

    def test_decodes_html_entities(self):
        self.assertEqual(clean_text("salt &amp; pepper"), "salt & pepper")

    def test_shortens_elongated_words(self):
        self.assertEqual(clean_text("soooo good"), "soo good")

    def test_keeps_case_and_punctuation(self):
        self.assertEqual(clean_text("NOT good!!!"), "NOT good!!!")

    def test_link_only_post_becomes_empty(self):
        self.assertEqual(clean_text("https://t.co/abc"), "")

    def test_splits_camel_case_hashtag(self):
        self.assertEqual(segment_hashtag("unicornNation"), "unicorn Nation")


if __name__ == "__main__":
    unittest.main()
