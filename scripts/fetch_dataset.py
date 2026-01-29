#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime


def _load_yaml(path):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to read registry.yaml. Install with: pip install pyyaml"
        ) from exc
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _find_dataset(registry, name, version):
    matches = [
        item for item in registry.get("datasets", []) if item.get("name") == name
    ]
    if not matches:
        raise ValueError(f"Dataset '{name}' not found in registry")
    if version:
        matches = [item for item in matches if str(item.get("version")) == str(version)]
        if not matches:
            raise ValueError(f"Dataset '{name}' version '{version}' not found")
    if len(matches) > 1:
        versions = ", ".join(str(item.get("version")) for item in matches)
        raise ValueError(f"Multiple versions found for '{name}': {versions}")
    return matches[0]


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checksums(root_dir, checksum_file):
    failures = []
    with open(checksum_file, "r", encoding="utf-8") as handle:
        for line in handle:
            entry = line.strip()
            if not entry or entry.startswith("#"):
                continue
            expected, rel_path = entry.split(None, 1)
            target = os.path.join(root_dir, rel_path)
            if not os.path.exists(target):
                failures.append(f"missing {rel_path}")
                continue
            actual = _sha256_file(target)
            if actual != expected:
                failures.append(f"mismatch {rel_path}")
    if failures:
        raise RuntimeError("Checksum verification failed: " + ", ".join(failures))


def _default_target_subdir(entry):
    dataset_type = entry.get("type", "real")
    name = entry.get("name")
    version = entry.get("version")
    return f"{dataset_type}/{name}/v{version}"


def main():
    parser = argparse.ArgumentParser(description="Fetch a dataset from local storage")
    parser.add_argument("--registry", default="manifests/registry.yaml")
    parser.add_argument("--name", required=True)
    parser.add_argument("--version")
    parser.add_argument("--data-root", help="Override data root in registry")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    registry = _load_yaml(args.registry)
    entry = _find_dataset(registry, args.name, args.version)

    source = entry.get("source", {})
    local_path = source.get("local_path")
    if not local_path:
        raise ValueError("registry entry is missing source.local_path")

    data_root = args.data_root or registry.get("data_root", "data")
    target_subdir = entry.get("target_subdir") or _default_target_subdir(entry)
    target_dir = os.path.join(data_root, target_subdir)

    if os.path.exists(target_dir):
        if not args.force:
            raise FileExistsError(f"Target exists: {target_dir}")
        if not args.dry_run:
            shutil.rmtree(target_dir)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "action": "copy",
                    "source": local_path,
                    "target": target_dir,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                },
                indent=2,
            )
        )
        return

    if os.path.isdir(local_path):
        shutil.copytree(local_path, target_dir)
    else:
        os.makedirs(target_dir, exist_ok=True)
        shutil.copy2(local_path, target_dir)

    checksum_file = entry.get("checksum_file")
    if checksum_file:
        checksum_path = os.path.join(target_dir, checksum_file)
        if os.path.exists(checksum_path):
            _verify_checksums(target_dir, checksum_path)

    print(f"Fetched '{args.name}' to {target_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
