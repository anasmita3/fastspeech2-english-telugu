"""Telugu: preprocess. All the heavy lifting is in common/pipeline.py (shared with the other language)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from common import pipeline  # noqa: E402

if __name__ == "__main__":
    pipeline.main_preprocess("telugu", os.path.join(HERE, "config.yaml"))
