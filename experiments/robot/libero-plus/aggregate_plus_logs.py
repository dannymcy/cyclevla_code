"""
aggregate_plus_logs.py

Aggregate the per-(suite, category) eval logs produced by
`run_libero_plus_eval_decomposed_progress_{transit,mbr}.py` into **7 per-category logs
per stage** that sum success rates across the 4 LIBERO-Plus suites
(libero_spatial / libero_object / libero_goal / libero_10).

Why: each eval invocation writes one log per (suite, category) slice — handy per-slice
but not the per-category total across the benchmark. This script scans the eval log
directory, re-parses every `EVAL-*.txt` (taking the latest log per (suite, category) by
mtime so re-runs cleanly overwrite earlier numbers), and writes:

    rollouts-plus/logs_plus_decomposed_progress_transit/AGGREGATE-<Category>.txt   (x7)
    rollouts-plus/logs_plus_decomposed_progress_mbr/AGGREGATE-<Category>.txt       (x7)

Categories: Camera / Robot / Language / Light / Background / Noise / Layout.

Usage:
    # process both transit and mbr log dirs with the default paths used by the launcher
    python experiments/robot/libero-plus/aggregate_plus_logs.py

    # or aim at a specific log dir
    python experiments/robot/libero-plus/aggregate_plus_logs.py \\
        --log_dir ./rollouts-plus/logs_plus_decomposed_progress_transit --stage transit

Idempotent: re-running just overwrites the AGGREGATE-*.txt files with fresh totals.
"""

import argparse
import datetime
import glob
import os
import re
import sys

# Allow `from experiments.robot.libero_plus_utils import ...` when run from anywhere.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_THIS_DIR, "../../..")))

from experiments.robot.libero_plus_utils import (  # noqa: E402
    CATEGORY_FULL_NAMES,
    PLUS_CATEGORY_COUNTS,
    SUITES,
    category_slug,
)


# Default log dirs (match the eval scripts' GenerateConfig defaults).
DEFAULT_TRANSIT_LOG_DIR = "./rollouts-plus/logs_plus_decomposed_progress_transit"
DEFAULT_MBR_LOG_DIR = "./rollouts-plus/logs_plus_decomposed_progress_mbr"

# Extract the suite from filenames like `EVAL-libero_spatial-Camera-frac10-openvla-...txt`.
_LOG_NAME_RE = re.compile(r"^EVAL-(libero_(?:spatial|object|goal|10))-")
# Match a per-category line emitted by the eval scripts, e.g.
#   "  Camera Viewpoints: 12/15 (80.0%)  [full suite: 376 variants]"
_CAT_LINE_RE = re.compile(r"^\s+(.+?):\s+(\d+)/(\d+)\s+\(")


def _parse_log_for_category_stats(path):
    """Return {full_category_name: (successes, episodes)} from one EVAL-*.txt log."""
    out = {}
    in_block = False
    with open(path, "r") as f:
        for line in f:
            if line.startswith("Per-category success rates"):
                in_block = True
                continue
            if not in_block:
                continue
            m = _CAT_LINE_RE.match(line)
            if m:
                cat = m.group(1).strip()
                if cat in CATEGORY_FULL_NAMES:
                    out[cat] = (int(m.group(2)), int(m.group(3)))
                continue
            # First non-matching, non-empty line ends the per-category block (the rest of
            # the log holds wandb / latency summaries we don't care about).
            if line.strip():
                in_block = False
    return out


def aggregate_dir(log_dir, stage_label):
    """Scan all EVAL-*.txt logs in `log_dir`, write 7 AGGREGATE-<Category>.txt files."""
    if not os.path.isdir(log_dir):
        print(f"[aggregate] {log_dir} not found — skipping {stage_label}")
        return

    log_paths = sorted(
        glob.glob(os.path.join(log_dir, "EVAL-*.txt")), key=os.path.getmtime
    )
    if not log_paths:
        print(f"[aggregate] no EVAL-*.txt logs in {log_dir} — skipping {stage_label}")
        return

    # Iterate oldest -> newest and let later writes overwrite earlier ones, giving
    # "last run wins" semantics per (suite, category) — so a re-run of any slice cleanly
    # supersedes its prior numbers and we never double-count.
    data = {}  # (suite, full_category) -> (successes, episodes)
    for path in log_paths:
        m = _LOG_NAME_RE.search(os.path.basename(path))
        if not m:
            continue
        suite = m.group(1)
        for cat, (s, e) in _parse_log_for_category_stats(path).items():
            data[(suite, cat)] = (s, e)

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for cat in sorted(CATEGORY_FULL_NAMES):
        slug = category_slug(cat)
        per_suite_rows = []
        total_s = total_e = 0
        for suite in SUITES:
            if (suite, cat) in data:
                s, e = data[(suite, cat)]
                rate = (s / e * 100.0) if e else 0.0
                per_suite_rows.append((suite, s, e, rate))
                total_s += s
                total_e += e
        full_total = sum(PLUS_CATEGORY_COUNTS[s].get(cat, 0) for s in SUITES)
        agg_rate = (total_s / total_e * 100.0) if total_e else 0.0

        out_path = os.path.join(log_dir, f"AGGREGATE-{slug}.txt")
        with open(out_path, "w") as f:
            f.write(f"LIBERO-Plus aggregate — {cat} ({stage_label}) — generated {now}\n")
            f.write("=" * 72 + "\n")
            f.write("Per-suite breakdown:\n")
            if per_suite_rows:
                for suite, s, e, rate in per_suite_rows:
                    f.write(f"  {suite:<15s} {s:>5d}/{e:<5d}  ({rate:5.1f}%)\n")
            else:
                f.write("  (no data yet — no EVAL-*.txt contained this category)\n")
            f.write("-" * 72 + "\n")
            f.write(
                f"Aggregate ({len(per_suite_rows)} "
                f"suite{'s' if len(per_suite_rows) != 1 else ''}): "
                f"{total_s}/{total_e} ({agg_rate:.1f}%)"
                f"   [full Plus for this category: {full_total} variants]\n"
            )
        print(f"[aggregate] wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate LIBERO-Plus eval logs across the 4 suites, per category."
    )
    parser.add_argument(
        "--log_dir", type=str, default=None,
        help="Aggregate a single log dir (with --stage). If omitted, processes the "
             "default transit and mbr dirs under rollouts-plus/.",
    )
    parser.add_argument(
        "--stage", type=str, default=None, choices=["transit", "mbr", "cyclevla"],
        help="Label used in the aggregate file header (required with --log_dir).",
    )
    args = parser.parse_args()

    if args.log_dir:
        if args.stage is None:
            parser.error("--stage is required when --log_dir is given")
        aggregate_dir(args.log_dir, args.stage)
    else:
        aggregate_dir(DEFAULT_TRANSIT_LOG_DIR, "transit")
        aggregate_dir(DEFAULT_MBR_LOG_DIR, "mbr")


if __name__ == "__main__":
    main()
