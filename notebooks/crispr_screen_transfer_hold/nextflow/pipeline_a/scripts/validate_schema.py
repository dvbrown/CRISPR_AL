"""Validate all metric JSON files against metrics.schema.json.

Reads all *.json files in the current directory and validates each against the schema.
Exits non-zero if any validation fails (causes Nextflow process to fail).
"""
import argparse
import glob
import json
import sys

import jsonschema


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema-json", required=True, help="Path to metrics.schema.json")
    parser.add_argument("--output",      default="validation_report.txt")
    args = parser.parse_args()

    with open(args.schema_json) as f:
        schema = json.load(f)

    json_files = sorted(glob.glob("*.json"))
    # Exclude the schema file itself if it landed here
    json_files = [p for p in json_files if not p.endswith("schema.json")]

    n_pass, n_fail = 0, 0
    lines = []

    for path in json_files:
        try:
            with open(path) as f:
                record = json.load(f)
            jsonschema.validate(instance=record, schema=schema)
            n_pass += 1
            lines.append(f"PASS  {path}")
        except Exception as e:
            n_fail += 1
            lines.append(f"FAIL  {path}: {e}")

    report = "\n".join(lines)
    summary = f"\nSchema validation: {n_pass} PASS / {n_fail} FAIL / {len(json_files)} total"
    report += summary

    with open(args.output, "w") as f:
        f.write(report + "\n")

    print(report)

    if n_fail > 0:
        print(f"\nERROR: {n_fail} JSON file(s) failed schema validation", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
