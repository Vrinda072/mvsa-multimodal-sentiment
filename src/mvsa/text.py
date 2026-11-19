"""Text-cleaning functions shared by the preprocessing notebooks and the prediction code.

Training captions and captions typed into the demo go through the same clean_text function, so the
models always see text prepared in exactly the same way.
"""
import html
import re
import unicodedata
from functools import lru_cache

# Optional helpers: without them, the functions below use simpler built-in fallbacks
try:
    import emoji          # names and counts emoji reliably
except ImportError:
    emoji = None
try:
    import wordsegment    # splits run-together words, e.g. "happybday" -> "happy bday"
    wordsegment.load()    # loads its word-frequency tables once, at import time
except ImportError:
    wordsegment = None
try:
    import ftfy           # repairs text with broken character encoding (used in predict.py)
except ImportError:
    ftfy = None


# Patterns for each kind of text cleaned below
URL_RE      = re.compile(r"(https?://\S+|www\.\S+|\bt\.co/\S+)", re.I)   # web links, including t.co short links
MENTION_RE  = re.compile(r"@\w+")                                         # @username mentions
# A retweet marker at the very start of a caption, e.g. "RT @user: ". Because of the ^ anchor only a leading
# marker is removed; an "RT @user:" later in the caption (a retweet of a retweet) is left in the text.
RT_RE       = re.compile(r"^\s*RT\b\s*(@user)?\s*:?\s*", re.I)
HASHTAG_RE  = re.compile(r"#(\w+)")                                       # a hashtag; group 1 is the word without #
ELONG_RE    = re.compile(r"(\w)\1{2,}")                                   # a character repeated 3+ times, e.g. "soooo"
EMPTY_PLACEHOLDER = "no text"   # used when a caption is empty after cleaning, so CLIP always gets some input

@lru_cache(maxsize=None)                               # the same hashtag is only segmented once
def segment_hashtag(tag):
    """Split a hashtag into words: first at case and digit boundaries, then with wordsegment if still one word."""
    # Insert a space between a lower- and upper-case letter ("GoCrimson" -> "Go Crimson") and between letters
    # and digits ("summer2016" -> "summer 2016"); underscores become spaces too
    split = re.sub(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", " ", tag).replace("_", " ")
    # An all-lowercase tag such as "happybday" has no boundaries to split on, so use the word-frequency model.
    # Short tags (5 letters or fewer) are left alone, because they are usually a single word already.
    if wordsegment is not None and " " not in split and tag.isalpha() and len(tag) > 5:
        split = " ".join(wordsegment.segment(tag))
    return split

def is_emoji_char(ch):
    """True if the character is an emoji."""
    if emoji is not None:
        return emoji.is_emoji(ch)
    o = ord(ch)                                         # fallback: symbols in the emoji code-point ranges
    return unicodedata.category(ch) == "So" and (o >= 0x1F000 or 0x2600 <= o <= 0x27BF)

def count_emojis(text):
    """Number of emoji in a text (used as an engineered feature)."""
    return emoji.emoji_count(text) if emoji is not None else sum(is_emoji_char(c) for c in text)

def emojis_to_text(text):
    """Replace each emoji with its name in words, e.g. a smiling face becomes " smiling face ".

    CLIP and TF-IDF both work on words, so describing an emoji keeps its sentiment instead of dropping it.
    """
    if emoji is not None:
        # data['en'] is the emoji's English name, e.g. ":red_heart:" -> "red heart"
        return emoji.replace_emoji(text, replace=lambda e, data: f" {data['en'].strip(':').replace('_', ' ')} ")
    # Fallback without the emoji library: drop invisible joiner and variation characters, then use the
    # Unicode name of each emoji character
    text = text.replace("️", "").replace("‍", "")
    return "".join(f" {unicodedata.name(c, '').lower()} " if is_emoji_char(c) else c for c in text)

def clean_text(t):
    """Apply the cleaning steps listed above, in that order."""
    t = html.unescape(t or "")                          # "&amp;" -> "&"; None becomes an empty string
    t = URL_RE.sub(" ", t)                              # links carry no sentiment, so they are removed
    t = MENTION_RE.sub("@user", t)                      # usernames are anonymised but the mention is kept
    t = RT_RE.sub("", t)                                # remove a leading retweet marker (see RT_RE above)
    t = HASHTAG_RE.sub(lambda m: segment_hashtag(m.group(1)), t)   # "#GoCrimson" -> "Go Crimson"
    t = emojis_to_text(t)                               # emoji -> words
    t = ELONG_RE.sub(r"\1\1", t)                        # "soooo" -> "soo", so variants share one form
    return re.sub(r"\s+", " ", t).strip()               # collapse repeated spaces
