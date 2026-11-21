"""Run a three-round threshold study with a local CSV or a synthetic sample."""
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import random
import subprocess
import sys
from tempfile import TemporaryDirectory


def write_sample(path):
    rng = random.Random(42)
    features = [f"V{i}" for i in range(1, 29)]
    with path.open("w", newline="", encoding="utf-8") as stream:
        rows = csv.writer(stream)
        rows.writerow(["Time", *features, "Amount", "Class"])
        for i in range(2000):
            label = int(i % 20 == 0)
            values = [rng.gauss(0, 1) for _ in features]
            values[0] += 2 * label
            values[1] -= 2 * label
            rows.writerow([i, *values, round(rng.expovariate(1 / 80), 2), label])


def main():
    scripts = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, help="Use a local ULB CSV instead of synthetic demo data")
    parser.add_argument("--output", type=Path, help="Fresh output directory")
    args = parser.parse_args()
    if args.csv is not None and not args.csv.is_file():
        parser.error(f"CSV does not exist: {args.csv}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output or scripts.parent / "results" / "logs" / f"demo_{stamp}"
    if output.exists():
        parser.error(f"Output already exists: {output}")

    with TemporaryDirectory() as temp:
        source = args.csv
        if source is None:
            print("Synthetic demo data. Metrics are for pipeline verification.", flush=True)
            source = Path(temp) / "synthetic_demo.csv"
            write_sample(source)
        run = subprocess.run([
            sys.executable, str(scripts / "run_threshold_study.py"),
            "--csv", str(source), "--output", str(output),
            "--rounds", "3", "--schemes", "dir0.5", "--seeds", "0",
        ])
        if run.returncode:
            return run.returncode
    audit = subprocess.run([
        sys.executable, str(scripts / "verify_study_rankings.py"), str(output),
    ])
    if audit.returncode == 0:
        print(f"Open {output.resolve() / 'comparison.png'} for the paired test results.")
    return audit.returncode


if __name__ == "__main__":
    sys.exit(main())
