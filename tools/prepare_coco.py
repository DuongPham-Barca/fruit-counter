#!/usr/bin/env python
"""Build the initial orange/apple COCO dataset."""

import argparse
import json
from pathlib import Path

from fruit_counter.prepare_coco import prepare


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/datasets/coco-fruit"))
    parser.add_argument("--cache", type=Path, default=Path("data/cache/coco"))
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.cache, args.val_fraction, args.workers)))


if __name__ == "__main__":
    main()
