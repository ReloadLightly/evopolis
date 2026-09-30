"""Run Task 01: python -m evopolis.reproduce (from the repository root)."""

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import resource
import time

from .analysis import figure_groups, rank_tests, summarize, table_markdown, write_csv
from .sources import acquire, REVISION


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/task01"))
    parser.add_argument("--figure-dir", type=Path, default=Path("docs/assets"))
    args = parser.parse_args()
    started = time.perf_counter()
    expected_path = Path(__file__).with_name("source_checksums.json")
    expected = json.loads(expected_path.read_text()) if expected_path.exists() else {}
    cached = all((args.raw_dir / name).exists() for name in ("sustainable_behavior.csv", "sustainable_behavior.ipynb", "paper.xml"))
    manifest = acquire(args.raw_dir, expected)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.raw_dir / "sustainable_behavior.csv"
    from .inventory import inventory
    from .replay import replay
    from .plotting import plot
    print("Inventorying released records…", flush=True)
    evidence = inventory(path)
    write_json(args.output_dir / "inventory.json", evidence)
    print("Replaying resource accounting…", flush=True)
    replay_result = replay(path, args.output_dir)
    write_json(args.output_dir / "replay.json", replay_result)
    print("Aggregating Figure 2A cohorts…", flush=True)
    groups = figure_groups(path)
    summary = summarize(groups)
    write_csv(args.output_dir / "figure2a_groups.csv", groups)
    write_csv(args.output_dir / "figure2a_summary.csv", summary)
    write_csv(args.output_dir / "rank_tests.csv", rank_tests(groups))
    (args.output_dir / "figure2a_table.md").write_text(table_markdown(summary))
    plot(groups, summary, args.figure_dir)
    write_json(args.output_dir / "manifest.json", {
        "paper_doi": "10.1038/s41467-025-58043-7", "upstream_revision": REVISION,
        "sources": manifest, "inventory": "inventory.json", "replay": "replay.json",
        "software_license": "Apache-2.0 (adapted notebook definitions)",
        "data_license": "CC-BY-4.0", "figure": "Figure 2A, recorded BC1 and human Experiment 1",
        "uncertainty": "Marginal mean +/- 1.96 * population SD / sqrt(number of games); notebook helper, added to figure",
        "exclusions": "Exact notebook condition selection; all 40 logged rows per game; undefined Gini remains NaN",
    })
    run = {"seconds": round(time.perf_counter() - started, 3),
           "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 3),
           "python": platform.python_version(), "platform": platform.platform(),
           "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "matplotlib")},
           "raw_sources_cached": cached}
    write_json(args.output_dir / "runtime.json", run)
    print(table_markdown(summary))
    print(json.dumps(run))


if __name__ == "__main__":
    main()
