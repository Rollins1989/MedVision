# Contributing to MedVision

Thank you for your interest in improving MedVision.

## Development workflow

1. Create a focused branch from `main`.
2. Keep changes scoped to one feature, experiment, or documentation improvement.
3. Add or update tests for behavioral changes.
4. Run the local quality checks before opening a pull request:

```bash
pytest -q
ruff check .
ruff format --check .
```

5. Document new experiments with their dataset, split, checkpoint, configuration, metrics, and limitations.
6. Never commit medical images, private patient information, credentials, model secrets, or large generated datasets.

## Research changes

For model or evaluation changes, please report:

- dataset/provenance;
- train/validation/test or external split strategy;
- checkpoint and configuration;
- primary and secondary metrics;
- threshold-selection procedure;
- uncertainty or confidence intervals where appropriate;
- known limitations and failure modes.

External medical datasets should remain outside the repository unless their redistribution terms explicitly permit inclusion.

## Pull requests

Please use a descriptive title and explain:

- what changed;
- why it changed;
- how it was tested;
- whether metrics or scientific conclusions changed.
