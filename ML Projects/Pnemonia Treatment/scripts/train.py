"""Training script entry point (thin wrapper over the CLI)."""
import argparse
import sys
from pneumonia.cli import build_parser


def main() -> int:
    parser = build_parser()
    # If no subcommand given, default to train for convenience
    argv = sys.argv[1:]
    if not argv:
        argv = ["train", "--config", "configs/default.yaml", "--data", "synthetic"]
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
