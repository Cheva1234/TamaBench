"""Deprecated V1 helper: export auditable per-attempt records without rewriting README.

Use `tamabench export` for new workflows. V1 results are not pooled into V2 rankings.
"""
import argparse
from tamabench.experiment import export_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", default="tamabench_results.db")
    parser.add_argument("--output-dir", default="tamabench-export")
    parser.add_argument("--experiment-id")
    args = parser.parse_args()
    from pathlib import Path
    if not Path(args.db_path).is_file():
        parser.error("Results database does not exist; run a benchmark first")
    rows = export_summary(args.db_path, args.output_dir, args.experiment_id)
    print(f"Exported {len(rows)} separate attempts to {args.output_dir}; README was not modified.")
