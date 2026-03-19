"""
Script to generate the Brainvita training dataset.

Usage:
    python3 generate.py [test|full]

    test  — 100 rows to verify correctness (default)
    full  — massive dataset, ~2 hours
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rice_ml.data_generator.generator import generate_csv

OUTPUT = os.path.join(os.path.dirname(__file__), "brainvita_dataset.csv")

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"

    if mode == "test":
        print("TEST MODE: 100 rows to verify correctness\n")
        generate_csv(
            output_path=OUTPUT,
            samples_per_size=5,
            min_circles=6,
            max_circles=25,
            seed=42,
        )

    elif mode == "full":
        print("FULL MODE: massive dataset (~2 hours)\n")

        # Tier 1: 6-12, dirt cheap
        print("=" * 50)
        print("TIER 1: 6-12 circles")
        print("=" * 50)
        generate_csv(
            output_path=OUTPUT,
            samples_per_size=50000,
            min_circles=6,
            max_circles=12,
            seed=42,
        )

        # Tier 2: 13-18, still fast
        t2 = OUTPUT.replace(".csv", "_t2.csv")
        print("\n" + "=" * 50)
        print("TIER 2: 13-18 circles")
        print("=" * 50)
        generate_csv(
            output_path=t2,
            samples_per_size=10000,
            min_circles=13,
            max_circles=18,
            seed=500000,
        )

        # Tier 3: 19-22, moderate
        t3 = OUTPUT.replace(".csv", "_t3.csv")
        print("\n" + "=" * 50)
        print("TIER 3: 19-22 circles")
        print("=" * 50)
        generate_csv(
            output_path=t3,
            samples_per_size=2000,
            min_circles=19,
            max_circles=22,
            seed=1000000,
        )

        # Tier 4: 23-25, slow
        t4 = OUTPUT.replace(".csv", "_t4.csv")
        print("\n" + "=" * 50)
        print("TIER 4: 23-25 circles")
        print("=" * 50)
        generate_csv(
            output_path=t4,
            samples_per_size=500,
            min_circles=23,
            max_circles=25,
            seed=2000000,
        )

        # Merge
        print("\nMerging tiers...")
        with open(OUTPUT, "a") as out:
            for path in [t2, t3, t4]:
                with open(path, "r") as f:
                    next(f)  # skip header
                    for line in f:
                        out.write(line)
                os.remove(path)

        size_mb = os.path.getsize(OUTPUT) / (1024 * 1024)
        print(f"\nDone. File: {OUTPUT}")
        print(f"Size: {size_mb:.1f} MB")

    else:
        print(f"Unknown mode '{mode}'. Use 'test' or 'full'.")
