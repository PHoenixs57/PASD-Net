#!/usr/bin/env python3
"""Randomly split a directory of WAV files into train/test subsets.

Copies the files into <output_dir>/train and <output_dir>/test, and writes a
split index CSV (filename, subset) recording the exact partition -- this CSV
is the releasable record of the split.

Usage:
    python scripts/divide-train-test.py <input_dir> <output_dir> [--ratio 0.8] [--seed 42]

Note: the published split index lists file identifiers only; ensure the source
filenames contain no location, device, or deployment-time information before
releasing the CSV.
"""
import argparse
import csv
import os
import random
import shutil


def split_dataset(input_dir, output_dir, train_ratio=0.8, seed=42, list_csv=None):
    """Randomly split WAV files from input_dir into train/test and copy them.

    Writes a (filename, subset) CSV index so the exact partition can be
    released and re-checked without the audio itself.
    """
    if not os.path.isdir(input_dir):
        raise ValueError(f"{input_dir} is not a valid directory")

    wav_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".wav")]
    if not wav_files:
        print("No WAV files found in the input directory.")
        return

    random.seed(seed)
    shuffled = sorted(wav_files)
    random.shuffle(shuffled)
    num_train = int(len(shuffled) * train_ratio)
    train_files = shuffled[:num_train]
    test_files = shuffled[num_train:]

    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for file_name in train_files:
        shutil.copy(os.path.join(input_dir, file_name), os.path.join(train_dir, file_name))
    for file_name in test_files:
        shutil.copy(os.path.join(input_dir, file_name), os.path.join(test_dir, file_name))

    # Split index: the releasable record of the exact partition
    if list_csv is None:
        list_csv = os.path.join(output_dir, "split_index.csv")
    with open(list_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "subset"])
        for file_name in train_files:
            writer.writerow([file_name, "train"])
        for file_name in test_files:
            writer.writerow([file_name, "test"])

    print(f"Found {len(shuffled)} WAV files (seed={seed}, ratio={train_ratio}):")
    print(f"    train: {len(train_files)}")
    print(f"    test:  {len(test_files)}")
    print(f"Train files copied to: {train_dir}")
    print(f"Test files copied to:  {test_dir}")
    print(f"Split index written to: {list_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Randomly split a directory of WAV files into train/test subsets "
        "and write a split index CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_dir", help="directory containing the WAV files")
    parser.add_argument("output_dir", help="output directory for the train/ and test/ copies")
    parser.add_argument("-r", "--ratio", type=float, default=0.8,
                        help="train fraction (remainder goes to test)")
    parser.add_argument("-s", "--seed", type=int, default=42,
                        help="random seed for reproducible splits")
    parser.add_argument("--list-csv", type=str, default=None,
                        help="path for the split index CSV (default: <output_dir>/split_index.csv)")
    args = parser.parse_args()

    split_dataset(args.input_dir, args.output_dir, args.ratio, args.seed, args.list_csv)


if __name__ == "__main__":
    main()
