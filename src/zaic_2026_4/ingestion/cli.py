"""Command-line interface for running the ingestion pipeline."""

import argparse
from pathlib import Path
import sys

from zaic_2026_4.ingestion.service import (
    DEFAULT_BASE_OUTPUT_DIR,
    ingest_business,
    ingest_split,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for CLI execution."""
    parser = argparse.ArgumentParser(
        description="Ingest multi-format tabular data into SQLite databases."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--split",
        type=str,
        help="Name or path of dataset split (e.g. 'train').",
    )
    group.add_argument(
        "--business",
        type=str,
        help="Path to a single business directory (e.g. 'dataset/train/data/biz_101').",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Target output directory (default: {DEFAULT_BASE_OUTPUT_DIR}/<split>).",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="dataset",
        help="Base directory containing dataset splits (default: 'dataset').",
    )
    parser.add_argument(
        "--no-views",
        action="store_true",
        help="Disable automatic generation of unified VIEWs for identical schemas.",
    )
    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    create_views = not args.no_views
    out_dir = Path(args.output_dir) if args.output_dir else None

    if args.business:
        biz_path = Path(args.business)
        print(f"Ingesting business: {biz_path} ...")
        db_path = ingest_business(
            biz_path, output_dir=out_dir, create_views=create_views
        )
        print(f"Done! Database generated at: {db_path}")
    elif args.split:
        print(f"Ingesting dataset split: {args.split} ...")
        results = ingest_split(
            args.split,
            base_dir=args.base_dir,
            output_dir=out_dir,
            create_views=create_views,
        )
        print(f"Successfully ingested {len(results)} businesses for split '{args.split}'.")


if __name__ == "__main__":
    main()
