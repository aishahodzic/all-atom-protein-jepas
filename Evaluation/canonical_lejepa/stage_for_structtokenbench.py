"""Stage canonical EMA/raw encoder weights for the unchanged STB JEPA loader."""

from __future__ import annotations

import argparse

from Evaluation.canonical_lejepa.checkpoint import (
    stage_structtokenbench_checkpoint,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument(
        "--encoder-source", choices=("ema", "raw"), default="ema"
    )
    args = parser.parse_args()
    output = stage_structtokenbench_checkpoint(
        args.checkpoint, args.output, source=args.encoder_source
    )
    print(output)


if __name__ == "__main__":
    main()

