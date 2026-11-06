"""Unit tests for reading captions in pictures. The test that reads real text needs the optional text reader."""
import os
import sys
import unittest
from io import BytesIO
from unittest import mock

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mvsa import ocr  # noqa: E402
from mvsa.predict import SentimentModel, _read_bytes  # noqa: E402


def picture(text=None):
    image = Image.new("RGB", (900, 300), (40, 90, 150))
    if text:
        ImageDraw.Draw(image).text((40, 100), text, fill="white", font=ImageFont.load_default(size=56))
    buffer = BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


class ReadCaptionTests(unittest.TestCase):
    @unittest.skipUnless(ocr.is_available(), "rapidocr_onnxruntime is not installed")
    def test_reads_text_in_a_picture(self):
        self.assertIn("best day ever", ocr.read_caption(picture("Best day ever")).lower())

    @unittest.skipUnless(ocr.is_available(), "rapidocr_onnxruntime is not installed")
    def test_returns_none_without_text(self):
        self.assertIsNone(ocr.read_caption(picture()))

    def test_ignores_low_confidence_text(self):
        found = [([], "clear", "0.9"), ([], "unclear", "0.2")]
        with mock.patch.object(ocr, "_engine", lambda image: (found, None)):
            self.assertEqual(ocr.read_caption(b"image"), "clear")

    def test_reading_bytes_leaves_the_file_ready(self):
        image = BytesIO(b"data")
        self.assertEqual(_read_bytes(image), b"data")
        self.assertEqual(image.read(), b"data")


class PredictWithCaptionTests(unittest.TestCase):
    def predict(self, text, read_caption, available=True, found="great"):
        model = SentimentModel.__new__(SentimentModel)
        model.labels = ["negative", "neutral", "positive"]
        seen = {}
        model.embed_text = lambda t: seen.setdefault("text", t)
        model.embed_image = lambda image: "image"
        model.proba_from_embeddings = lambda t, i: [__import__("numpy").array([0.2, 0.3, 0.5])]
        with mock.patch.object(ocr, "is_available", return_value=available), mock.patch.object(ocr, "read_caption", return_value=found):
            return model.predict(text, BytesIO(b"image"), read_caption=read_caption), seen

    def test_uses_a_caption_found_in_the_picture(self):
        result, seen = self.predict(None, True)
        self.assertEqual((seen["text"], result["caption_read"], result["used"]), ("great", "great", ["text", "image"]))

    def test_a_typed_caption_wins(self):
        result, seen = self.predict("typed", True)
        self.assertEqual((seen["text"], result["caption_read"]), ("typed", None))

    def test_picture_alone_when_no_caption_is_found(self):
        result, seen = self.predict(None, True, found=None)
        self.assertEqual((seen, result["used"]), ({}, ["image"]))

    def test_reports_a_missing_text_reader(self):
        result, _ = self.predict(None, True, available=False)
        self.assertEqual((result["ocr_missing"], result["used"]), (True, ["image"]))


if __name__ == "__main__":
    unittest.main()
