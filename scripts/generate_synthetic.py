#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import os
import random
import sys
from datetime import datetime


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_checksums(root_dir, files):
    checksum_path = os.path.join(root_dir, "checksums.sha256")
    with open(checksum_path, "w", encoding="utf-8") as handle:
        for rel_path in files:
            full_path = os.path.join(root_dir, rel_path)
            handle.write(f"{_sha256_file(full_path)}  {rel_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate deterministic synthetic data"
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    target_dir = os.path.join(
        args.data_root,
        "synthetic",
        args.name,
        f"v{args.version}",
    )

    if os.path.exists(target_dir):
        if not args.force:
            raise FileExistsError(f"Target exists: {target_dir}")
        for entry in os.listdir(target_dir):
            path = os.path.join(target_dir, entry)
            if os.path.isdir(path):
                for root, dirs, files in os.walk(path, topdown=False):
                    for file_name in files:
                        os.remove(os.path.join(root, file_name))
                    for dir_name in dirs:
                        os.rmdir(os.path.join(root, dir_name))
                os.rmdir(path)
            else:
                os.remove(path)
    else:
        os.makedirs(target_dir, exist_ok=True)

    random.seed(args.seed)
    data_path = os.path.join(target_dir, "data.csv")
    header = [f"feature_{idx + 1}" for idx in range(args.cols)] + ["label"]

    with open(data_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for _ in range(args.rows):
            features = [round(random.random(), 6) for _ in range(args.cols)]
            label = random.choice([0, 1])
            writer.writerow(features + [label])

    metadata = {
        "name": args.name,
        "version": args.version,
        "rows": args.rows,
        "cols": args.cols,
        "seed": args.seed,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "generator": "basic_tabular_v1",
    }
    metadata_path = os.path.join(target_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    _write_checksums(target_dir, ["data.csv", "metadata.json"])
    print(f"Synthetic dataset written to {target_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
