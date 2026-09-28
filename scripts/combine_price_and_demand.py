"""Combine all PRICE_AND_DEMAND CSVs in data/ into one clean, sorted dataframe."""

import glob
import os

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "nem_combined_clean.csv")


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "PRICE_AND_DEMAND_*.csv")))
    if not files:
        raise SystemExit(f"No PRICE_AND_DEMAND CSVs found in {DATA_DIR}")

    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)

    df["SETTLEMENTDATE"] = pd.to_datetime(df["SETTLEMENTDATE"], format="%Y/%m/%d %H:%M:%S")
    df = df.drop_duplicates(subset=["REGION", "SETTLEMENTDATE"])
    df = df.sort_values(["REGION", "SETTLEMENTDATE"]).reset_index(drop=True)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df)} rows from {len(files)} files to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
