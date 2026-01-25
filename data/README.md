Data layout

This directory stores real datasets, synthetic fixtures, models, and embeddings.
The contents are intentionally gitignored to keep the repository lightweight.
Tracked test fixtures live under tests/data/ instead.

Layout

- data/real/<dataset_name>/vYYYYMMDD/
- data/synthetic/<dataset_name>/vYYYYMMDD/
- data/models/<model_name>/vYYYYMMDD/
- data/embeddings/<artifact_name>/vYYYYMMDD/

Each version directory should include:
- metadata.json
- checksums.sha256

Data registry and scripts

- manifests/registry.yaml is the single source of truth for datasets and versions.
- scripts/fetch_dataset.py copies a dataset from local storage into data/.
- scripts/generate_synthetic.py creates reproducible synthetic fixtures under data/.
- scripts/verify_checksums.py validates checksums.sha256 for a dataset.

Testing fixtures

- tests/data/ contains small, tracked datasets for unit and smoke tests.
- Use scripts/create_gears_iterpert_tiny.py to regenerate the GEARS/IterPert fixture.
