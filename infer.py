# ── infer.py ─────────────────────────────────────────────────
# Command line tool for testing the classifier outside of Maya.
#
# Run from your terminal like this:
#   python infer.py path/to/texture.png
#   python infer.py texture1.png texture2.png texture3.png
#
# Prints a JSON result for each image to stdout.
#
# This file is NOT used by Maya — Maya imports classifier.py
# directly and runs inference in-process without subprocess.
# This file exists purely for testing and debugging.
# ─────────────────────────────────────────────────────────────

import sys
import json

# Import the shared predict function from classifier.py
# This means all inference logic lives in one place
from classifier import predict


if __name__ == "__main__":

    # Expect at least one texture path after the script name
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: python infer.py texture.png"}))
        sys.exit(1)

    # All arguments after the script name are texture paths
    image_paths = sys.argv[1:]

    # Classify each image and collect results.
    # The model loads on the first predict() call and stays
    # loaded for all subsequent calls — no repeated cold start.
    results = []
    for path in image_paths:
        result = predict(path)
        results.append(result)

        # Also print each result as we go so you can see progress
        print(json.dumps(result))