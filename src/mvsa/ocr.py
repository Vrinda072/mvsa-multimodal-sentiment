"""Read a caption that is part of a picture, such as the text on a meme or a screenshot of a post."""
MIN_CONFIDENCE = 0.5   # ignore text the reader is unsure about

_engine = None


def is_available():
    """Check whether the optional text reader is installed."""
    try:
        import rapidocr_onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def read_caption(image_bytes):
    """Return the text found in an image as one string, or None if there isn't any."""
    global _engine
    if _engine is None:
        from rapidocr_onnxruntime import RapidOCR
        _engine = RapidOCR()
    found, _ = _engine(image_bytes)
    words = [text.strip() for _, text, confidence in (found or []) if float(confidence) >= MIN_CONFIDENCE]
    return " ".join(w for w in words if w) or None
