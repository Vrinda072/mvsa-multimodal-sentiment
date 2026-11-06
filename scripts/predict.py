"""Predict the sentiment of a new post with the final model.

Usage:
  python scripts/predict.py --text "What a great day" --image photo.jpg
  python scripts/predict.py --text "What a great day"
  python scripts/predict.py --image photo.jpg

Giving both a text and an image uses the selected late-fusion model. Giving only one uses that modality's
model alone, which scored lower on the test set (macro-F1 0.518 for text and 0.561 for image, against 0.603).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from mvsa.predict import SentimentModel  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Predict positive, negative or neutral for a new post.")
    parser.add_argument("--text", help="the post's text")
    parser.add_argument("--image", help="path to the post's image")
    args = parser.parse_args()
    if args.image is None and (args.text is None or not args.text.strip()):
        parser.error("give a non-empty --text, an --image, or both")
    if args.image is not None and not os.path.isfile(args.image):
        parser.error(f"image not found: {args.image}")

    result = SentimentModel().predict(args.text, args.image)
    print(f"Prediction: {result['label']}")
    print(f"Score:      {result['score']:.3f}")
    print(f"Used:       {' and '.join(result['used'])}")
    print("All classes:")
    for label, probability in sorted(result["probabilities"].items(), key=lambda item: -item[1]):
        print(f"  {label:9s} {probability:.3f}")


if __name__ == "__main__":
    main()
