"""A small local web page for trying the saved model on your own posts.

It uses only Python's standard library plus the project's dependencies. It listens on this computer only, and
uploaded images are kept in memory for the prediction and never saved. Every answer carries the numbers behind
the prediction (src/mvsa/explain.py), which the page draws.

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
# Make the mvsa package in src/ importable when the server is started directly
sys.path.insert(0, os.path.join(APP_DIR, "..", "src"))

MAX_BODY_BYTES = 30 * 1024 * 1024   # a 20 MB image grows by about a third when base64-encoded
UNREADABLE_IMAGE = "That file couldn't be read as an image. Please use a JPG or PNG."


def read_request(payload, need_input=True):
    """Validate one request from the page and return its caption and picture. Raises ValueError for bad input."""
    from PIL import Image, UnidentifiedImageError

    # An empty or whitespace-only caption counts as no caption
    text = (payload.get("text") or "").strip() or None
    image = None
    if payload.get("image"):
        try:
            # The page sends the picture as a data URL ("data:image/jpeg;base64,..."); keep only the base64 part
            data = base64.b64decode(payload["image"].split(",")[-1], validate=True)
            # verify() checks the file is a readable image without decoding all of it
            with Image.open(BytesIO(data)) as check:
                check.verify()
        except (binascii.Error, UnidentifiedImageError, OSError, SyntaxError, ValueError):
            raise ValueError(UNREADABLE_IMAGE)
        # Keep the picture in memory only; it is never written to disk
        image = BytesIO(data)
    if need_input and text is None and image is None:
        raise ValueError("Add a picture, a caption, or both.")
    return text, image


def run_prediction(model, payload):
    """Return the model's prediction for one request from the page. Raises ValueError for bad input."""
    text, image = read_request(payload)
    try:
        return model.predict(text, image)
    except OSError:
        raise ValueError(UNREADABLE_IMAGE)


def run_explanation(model, stats, payload):
    """Return the prediction with the numbers behind it. Raises ValueError for bad input.

    A request can ask for the example post's picture ("example_image") instead of sending one, so the example's
    caption can be edited without the picture being on this computer.
    """
    from mvsa.explain import explain_post

    # An uploaded picture always takes priority over the example picture
    use_example = bool(payload.get("example_image")) and not payload.get("image")
    text, image = read_request(payload, need_input=not use_example)
    try:
        return explain_post(model, stats, text, image, example_image=use_example)
    except OSError:
        raise ValueError(UNREADABLE_IMAGE)


# Build the request handler class with the model and statistics bound in, so they are loaded only once
def make_handler(model, stats):
    class Handler(BaseHTTPRequestHandler):
        # Send one complete HTTP response
        def send_body(self, status, content_type, body):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, status, content):
            self.send_body(status, "application/json", json.dumps(content).encode())

        # GET /example returns the example post's explanation; GET / returns the page itself
        def do_GET(self):
            if self.path == "/example":
                from mvsa.explain import explain_example
                return self.send_json(200, explain_example(model, stats))
            if self.path not in ("/", "/index.html"):
                return self.send_json(404, {"error": "Not found."})
            with open(os.path.join(APP_DIR, "index.html"), "rb") as f:
                self.send_body(200, "text/html; charset=utf-8", f.read())

        # POST /predict takes a caption and/or picture as JSON and returns the explained prediction
        def do_POST(self):
            if self.path != "/predict":
                return self.send_json(404, {"error": "Not found."})
            length = int(self.headers.get("Content-Length") or 0)
            # Refuse very large uploads before reading them into memory
            if length > MAX_BODY_BYTES:
                self.close_connection = True
                return self.send_json(413, {"error": "That image is too large. Please use one under 20 MB."})
            try:
                self.send_json(200, run_explanation(model, stats, json.loads(self.rfile.read(length))))
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

    from mvsa.explain import load_stats
    from mvsa.predict import SentimentModel

    print("Loading the model. The first run downloads CLIP (about 600 MB), which can take a while...")
    model = SentimentModel()
    # Load CLIP now rather than on the first request, so the first prediction is not slow
    model._load_clip()
    try:
        # 127.0.0.1: only this computer can reach the page, not other devices on the network
        server = HTTPServer(("127.0.0.1", args.port), make_handler(model, load_stats()))
    except OSError:
        raise SystemExit(f"Port {args.port} is in use. Try another one: python app/server.py --port {args.port + 1}")
    url = f"http://127.0.0.1:{args.port}"
    print(f"Ready at {url}\nPress Ctrl+C here to stop.")
    if not args.no_browser:
        # Open the browser half a second later, once the server is listening
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
