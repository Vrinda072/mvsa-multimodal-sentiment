"""A small local web page for trying the final model on your own posts.

It uses only Python's standard library plus the project's dependencies. It listens on this computer only, and
uploaded images are kept in memory for the prediction and never saved.

Usage: python app/server.py
"""
import argparse
import base64
import binascii
import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO

APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(APP_DIR, "..", "src"))

MAX_BODY_BYTES = 30 * 1024 * 1024   # a 20 MB image grows by about a third when base64-encoded
UNREADABLE_IMAGE = "That file couldn't be read as an image. Please use a JPG or PNG."


def run_prediction(model, payload):
    """Validate one request from the page and return the model's prediction. Raises ValueError for bad input."""
    from PIL import Image, UnidentifiedImageError

    text = (payload.get("text") or "").strip() or None
    image = None
    if payload.get("image"):
        try:
            data = base64.b64decode(payload["image"].split(",")[-1], validate=True)
            with Image.open(BytesIO(data)) as check:
                check.verify()
        except (binascii.Error, UnidentifiedImageError, OSError, SyntaxError, ValueError):
            raise ValueError(UNREADABLE_IMAGE)
        image = BytesIO(data)
    if text is None and image is None:
        raise ValueError("Add a picture, a caption, or both.")
    try:
        return model.predict(text, image)
    except OSError:
        raise ValueError(UNREADABLE_IMAGE)


def make_handler(model):
    class Handler(BaseHTTPRequestHandler):
        def send_body(self, status, content_type, body):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, status, content):
            self.send_body(status, "application/json", json.dumps(content).encode())

        def do_GET(self):
            if self.path not in ("/", "/index.html"):
                return self.send_json(404, {"error": "Not found."})
            with open(os.path.join(APP_DIR, "index.html"), "rb") as f:
                self.send_body(200, "text/html; charset=utf-8", f.read())

        def do_POST(self):
            if self.path != "/predict":
                return self.send_json(404, {"error": "Not found."})
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY_BYTES:
                self.close_connection = True
                return self.send_json(413, {"error": "That image is too large. Please use one under 20 MB."})
            try:
                self.send_json(200, run_prediction(model, json.loads(self.rfile.read(length))))
            except ValueError as e:
                self.send_json(400, {"error": str(e)})

        def log_message(self, *args):
            pass   # keep the terminal tidy

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Run the sentiment demo page on this computer.")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true", help="don't open the page automatically")
    args = parser.parse_args()

    from mvsa.predict import SentimentModel

    print("Loading the model. The first run downloads CLIP (about 600 MB), which can take a while...")
    model = SentimentModel()
    model._load_clip()
    try:
        server = HTTPServer(("127.0.0.1", args.port), make_handler(model))
    except OSError:
        raise SystemExit(f"Port {args.port} is in use. Try another one: python app/server.py --port {args.port + 1}")
    url = f"http://127.0.0.1:{args.port}"
    print(f"Ready at {url}\nPress Ctrl+C here to stop.")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
