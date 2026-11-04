"""Command-line interface for database profiling and MinHash join detection."""

import argparse
import json
from pathlib import Path
import sys
import time

from zaic_2026_4.profiling.raw_profiler import RawDatabaseProfiler


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Profile SQLite databases and detect undocumented FK-PK joins."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--db",
        type=str,
        help="Path to a single SQLite database file.",
    )
    group.add_argument(
        "--dir",
        type=str,
        help="Directory containing SQLite database files (*.db).",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Target output JSON path (for single database profiling).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Target output directory for profiles (for batch directory profiling).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="MinHash Jaccard threshold for FK-PK candidate joins (default: 0.7).",
    )
    parser.add_argument(
        "--num-perm",
        type=int,
        default=128,
        help="Number of MinHash permutations (default: 128).",
    )
    parser.add_argument(
        "--no-views",
        action="store_true",
        help="Disable profiling of SQLite VIEWs.",
    )
    parser.add_argument(
        "--all-tables",
        action="store_true",
        help="Do not skip partition tables even if a unified VIEW exists.",
    )
    parser.add_argument(
        "--include-continuous-numerics",
        action="store_true",
        help="Include continuous numeric/float columns in MinHash Jaccard calculations.",
    )
    return parser


def profile_single_db(
    db_path: Path,
    args: argparse.Namespace,
    output_path: Path | None = None,
) -> dict:
    """Profile one database and optionally persist profile to JSON."""
    profiler = RawDatabaseProfiler(
        db_path=db_path,
        num_perm=args.num_perm,
        include_views=not args.no_views,
        skip_partition_tables_if_view_exists=not args.all_tables,
        cross_table_only=True,
        jaccard_threshold=args.threshold,
        exclude_continuous_numerics=not args.include_continuous_numerics,
    )
    profile = profiler.profile_database()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)

    return profile


def main() -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    if args.db:
        db_file = Path(args.db)
        if not db_file.is_file():
            print(f"Error: Database file does not exist: {db_file}", file=sys.stderr)
            sys.exit(1)

        t0 = time.perf_counter()
        out_file = Path(args.output) if args.output else None
        profile = profile_single_db(db_file, args, output_path=out_file)
        elapsed = time.perf_counter() - t0

        tbl_count = len(profile["tables"])
        join_count = len(profile["undocumented_joins"])
        print(
            f"Successfully profiled {db_file.name} in {elapsed:.3f}s: "
            f"{tbl_count} entities, {join_count} candidate join paths."
        )
        if out_file:
            print(f"Profile saved to: {out_file}")

    elif args.dir:
        dir_path = Path(args.dir)
        if not dir_path.is_dir():
            print(f"Error: Directory does not exist: {dir_path}", file=sys.stderr)
            sys.exit(1)

        db_files = sorted(
            [
                p
                for p in dir_path.iterdir()
                if p.is_file() and p.suffix.lower() in (".db", ".sqlite", ".sqlite3")
            ]
        )
        if not db_files:
            print(f"No SQLite database files (.db, .sqlite, .sqlite3) found in {dir_path}", file=sys.stderr)
            sys.exit(1)

        out_dir = Path(args.output_dir) if args.output_dir else dir_path / "profiles"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Profiling {len(db_files)} databases in {dir_path} ...")
        t_start = time.perf_counter()
        total_joins = 0

        for db_file in db_files:
            out_file = out_dir / f"{db_file.stem}_profile.json"
            profile = profile_single_db(db_file, args, output_path=out_file)
            total_joins += len(profile["undocumented_joins"])

        total_elapsed = time.perf_counter() - t_start
        print(
            f"Completed profiling {len(db_files)} databases in {total_elapsed:.2f}s "
            f"({total_elapsed / len(db_files):.3f}s/db). "
            f"Discovered {total_joins} total candidate joins."
        )
        print(f"Profiles written to: {out_dir}")


if __name__ == "__main__":
    main()
