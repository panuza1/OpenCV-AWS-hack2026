#!/usr/bin/env python3
"""Run the deterministic Phase 3 benchmark and write its artifacts."""

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from physicseye.evaluate import (
    build_synthetic_scenarios,
    build_tracker_scenario,
    run_benchmark,
    write_benchmark,
)
from physicseye.tracker import track_video
from scripts.generate_sample import SCENARIOS, generate_samples


def _read_truth(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--report", type=Path, default=Path("docs/EVALUATION.md"))
    parser.add_argument("--sample-dir", type=Path, default=Path("examples/generated"))
    args = parser.parse_args(argv)

    scenarios = build_synthetic_scenarios()
    for sample in generate_samples(args.sample_dir):
        name = sample["scenario"]
        scenarios.append(
            build_tracker_scenario(
                f"opencv_tracker_{name}",
                track_video(sample["video"]),
                _read_truth(sample["ground_truth"]),
                SCENARIOS[name]["deceleration_px_s2"],
            )
        )

    benchmark = run_benchmark(scenarios)
    paths = write_benchmark(benchmark, args.output_dir, args.report)
    print(
        json.dumps(
            {
                "phase3_acceptance": "PASS" if benchmark["acceptance"]["passed"] else "FAIL",
                "adaptive_improvement_vs_frozen_pct": benchmark["acceptance"]["adaptive_improvement_vs_frozen_pct"],
                "scenario_count": len(benchmark["scenarios"]),
                "prediction_rows": sum(len(scenario["rows"]) for scenario in benchmark["scenarios"]),
                "artifacts": paths,
            },
            indent=2,
        )
    )
    return 0 if benchmark["acceptance"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
