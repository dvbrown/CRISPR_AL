#!/usr/bin/env python3
import argparse
import hashlib
import os
import sys


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checksums(checksum_file):
    entries = []
    with open(checksum_file, "r", encoding="utf-8") as handle:
        for line in handle:
            entry = line.strip()
            if not entry or entry.startswith("#"):
                continue
            expected, rel_path = entry.split(None, 1)
            entries.append((expected, rel_path))
    return entries


def main():
    parser = argparse.ArgumentParser(description="Verify sha256 checksums")
    parser.add_argument("--path", required=True, help="Root directory to verify")
    parser.add_argument(
        "--checksum-file",
        default="checksums.sha256",
        help="Checksum file name or path",
    )
    args = parser.parse_args()

    root_dir = args.path
    checksum_path = args.checksum_file
    if not os.path.isabs(checksum_path):
        checksum_path = os.path.join(root_dir, checksum_path)

    if not os.path.exists(checksum_path):
        raise FileNotFoundError(f"Checksum file not found: {checksum_path}")

    failures = []
    for expected, rel_path in _load_checksums(checksum_path):
        target = os.path.join(root_dir, rel_path)
        if not os.path.exists(target):
            failures.append(f"missing {rel_path}")
            continue
        actual = _sha256_file(target)
        if actual != expected:
            failures.append(f"mismatch {rel_path}")

    if failures:
        raise RuntimeError("Checksum verification failed: " + ", ".join(failures))
    print(f"Checksums verified for {root_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
