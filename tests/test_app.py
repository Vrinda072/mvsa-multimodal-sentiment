"""Unit tests for how the demo page's requests are checked. They use a stand-in model, so they don't need CLIP."""
import base64
import os
import sys
import unittest
from io import BytesIO

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from server import run_prediction  # noqa: E402


class FakeModel:
    def predict(self, text, image):
        return {"text": text, "has_image": image is not None}


def encoded_png():
    buffer = BytesIO()
    Image.new("RGB", (4, 4)).save(buffer, "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


class RunPredictionTests(unittest.TestCase):
    def test_accepts_text_only(self):
        self.assertEqual(run_prediction(FakeModel(), {"text": " hello "}), {"text": "hello", "has_image": False})

    def test_accepts_image_only(self):
        self.assertEqual(run_prediction(FakeModel(), {"text": "", "image": encoded_png()}), {"text": None, "has_image": True})

    def test_accepts_both(self):
        self.assertEqual(run_prediction(FakeModel(), {"text": "hi", "image": encoded_png()}), {"text": "hi", "has_image": True})

    def test_rejects_blank_input(self):
        with self.assertRaises(ValueError):
            run_prediction(FakeModel(), {"text": "   ", "image": None})

    def test_rejects_a_file_that_is_not_an_image(self):
        not_image = "data:image/png;base64," + base64.b64encode(b"not an image").decode()
        with self.assertRaises(ValueError):
            run_prediction(FakeModel(), {"image": not_image})

    def test_rejects_invalid_base64(self):
        with self.assertRaises(ValueError):
            run_prediction(FakeModel(), {"image": "data:image/png;base64,???"})


if __name__ == "__main__":
    unittest.main()
