
# Repository handoff checklist

Publication preparation on 2026-09-07:

- [x] Public repository requested by the project owner; shared-server assets remain external.
- [x] Initialize Git at the project root, independently of the unrelated home-directory repository.
- [x] Keep GNU GPL v3, upstream attribution, and the pinned upstream source archive intact.
- [x] Keep data/checkpoint hashes in `artifacts/manifests/assets.json`.
- [x] Exclude HDF5, NumPy arrays, checkpoints, logs, generated reports and experiment outputs from Git; retain selected small provenance/configuration records.
- [x] Check the staged files and upstream archive for common credential patterns (no findings; this is a pattern-based check).
- [x] Run the existing test suite with CPU execution: 61 tests passed.
- [x] Run a CPU snapshot inference with the existing shared checkpoint successfully.
- [x] Add [new-member setup and shared data/model instructions](../operations/COLLABORATION_zh-CN.md).

Historical absolute paths are retained where they identify shared-server assets
or frozen experiments. The collaboration guide explains which paths need
adjustment for a personal clone; not every historical script is portable through
`LANDAU_DATA_ROOT` alone. The team author placeholder remains unchanged pending
an agreed contributor list.

Creating the GitHub repository and pushing this commit require an authenticated
GitHub account. Check the actual Git remote and GitHub repository for publication
status; this preparation checklist is not evidence that a push has completed.
