"""Text-cleaning functions shared by the preprocessing notebooks and the prediction code."""
import html
import re
import unicodedata
from functools import lru_cache

# Optional helpers: without them, the functions below use simpler built-in fallbacks
try:
    import emoji
except ImportError:
    emoji = None
try:
    import wordsegment
    wordsegment.load()
except ImportError:
    wordsegment = None
try:
    import ftfy
except ImportError:
    ftfy = None


# Patterns for each kind of text cleaned below
URL_RE      = re.compile(r"(https?://\S+|www\.\S+|\bt\.co/\S+)", re.I)
MENTION_RE  = re.compile(r"@\w+")
RT_RE       = re.compile(r"^\s*RT\b\s*(@user)?\s*:?\s*", re.I)
HASHTAG_RE  = re.compile(r"#(\w+)")
ELONG_RE    = re.compile(r"(\w)\1{2,}")
EMPTY_PLACEHOLDER = "no text"

@lru_cache(maxsize=None)                               # the same hashtag is only segmented once
def segment_hashtag(tag):
    """Split a hashtag into words: first at case and digit boundaries, then with wordsegment if still one word."""
    split = re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", " ", tag).replace("_", " ")
    if wordsegment is not None and " " not in split and tag.isalpha() and len(tag) > 5:
        split = " ".join(wordsegment.segment(tag))
    return split

def is_emoji_char(ch):
    if emoji is not None:
        return emoji.is_emoji(ch)
    o = ord(ch)                                         # fallback: symbols in the emoji code-point ranges
    return unicodedata.category(ch) == "So" and (o >= 0x1F000 or 0x2600 <= o <= 0x27BF)

def count_emojis(text):
    return emoji.emoji_count(text) if emoji is not None else sum(is_emoji_char(c) for c in text)

def emojis_to_text(text):
    if emoji is not None:
        return emoji.replace_emoji(text, replace=lambda e, data: f" {data['en'].strip(':').replace('_', ' ')} ")
    text = text.replace("️", "").replace("‍", "")
    return "".join(f" {unicodedata.name(c, '').lower()} " if is_emoji_char(c) else c for c in text)

def clean_text(t):
    """Apply the cleaning steps listed above, in that order."""
    t = html.unescape(t or "")
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub("@user", t)
    t = RT_RE.sub("", t)
    t = HASHTAG_RE.sub(lambda m: segment_hashtag(m.group(1)), t)
    t = emojis_to_text(t)
    t = ELONG_RE.sub(r"\1\1", t)
    return re.sub(r"\s+", " ", t).strip()
