# Implementation Schedule

Updated: 2026-06-19

## Done

- Inspected repository structure; no prior `private_docs/` existed.
- Identified the active histogram path in `src/histvae/data_handler.py`.
- Added strict `count` / `density` histogram selection.
- Kept `count` as the default for existing experiments.
- Added `HistVAE.prep_data(..., histogram_mode=...)` runtime selection.
- Added 1D density tests and actual-data smoke validation.
- Updated public usage documentation.
- Added the standard `private_docs/` working record.

## Verification

- `pytest -q tests/test_histogram_modes.py`: 8 passed
- `pytest -m smoke -q`: 16 passed, 5 deselected
- `RUN_SLOW_HISTVAE_TESTS=0 pytest -q`: 16 passed, 5 skipped
- `RUN_SLOW_HISTVAE_TESTS=1 pytest -m slow -q`: 5 passed, 16 deselected
- Attached CSV smoke: 519,118 rows, 134 groups, finite 1D density input and
  reconstruction with shape `(8, 1, 64)`

## Next

- Create a dataset-specific analysis config and sample-level train/test policy.
- Compare density and count modes under the same split and seeds.

## Deferred

- CLI repair or redesign
- General preprocessing refactor

## Temporary shims

None.
