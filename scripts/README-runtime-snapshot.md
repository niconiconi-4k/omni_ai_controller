# Post-start snapshot verification

Run `python scripts/verify-runtime-snapshot.py RELEASE_DIRECTORY` for a read-only
post-start check. This does not deploy, restart services, load application modules,
query business APIs, or write financial data.

The release must have `snapshot-sha256.json` and `prepared.json`. The verifier
checks their hash binding, all recorded source hashes and inventories, and recorded
helper hashes where present. Missing files, unsafe paths, symlinks, special files,
and arbitrary extra files fail the check. Checks remain enabled under `python -O`.

The sole extra-file exception is a conventional CPython bytecode cache in
`omni_ai_controller/omni_ai_controller/__pycache__` whose corresponding Python
source is recorded in the frozen manifest. Its **contents are not attested**;
the report explicitly distinguishes cache paths from source integrity. This is
not a bytecode security scanner, a replacement for trusted release provenance,
or authorization to redeploy an already deployed release. The frozen preparation
and deployment helpers remain unchanged and retain their stricter pre-start gate.

Regression coverage lives in `tests/test_runtime_snapshot_verifier.py`.