# Source cleanup report

## Input verified

- Source review archive: `landau_source_review_20260731_171756.tar.gz`
- Verified SHA256: `e267ecd8af2b01db980b5bb829a4585834a62a54dc3c29a8b784e79efc2644db`
- Reviewed files in source bundle: 313 manifest entries, including 158 Python files and 4 notebooks.

## Main issues found in the research directory

1. The root README described an early MLP/FNO curve experiment and no longer
   represented the final x-v trajectory project.
2. The README stated MIT, while the actual repository license file was GNU GPL
   v3. The release follows the actual GPL v3 file and preserves upstream PIC
   attribution.
3. `pic/__init__.py` contained an upstream machine-specific absolute
   `sys.path` entry. It was removed.
4. User-facing functionality was spread across more than one hundred
   `stage*.py` experiment, comparison, acceptance and packaging scripts.
5. The research directory mixed source code with multi-gigabyte HDF5 datasets,
   many checkpoints, NPZ rollouts, plots, logs and profiling outputs.
6. Several metadata files stored machine-specific absolute paths.

## Cleanup performed

- extracted a functional package under `src/landau_surrogate`;
- exposed stable commands in the top-level `scripts/` directory;
- replaced internal stage-module imports with package-qualified imports;
- retained historical class names and checkpoint stage identifiers where
  needed for checkpoint compatibility;
- moved original notebooks and small PIC examples to `legacy/`;
- excluded all large data, checkpoints, logs and generated figures;
- added data, method, physics-diagnostic, reproducibility and limitations docs;
- added artifact names and hashes without server-specific absolute paths;
- added synthetic HDF5 tests that require no research data.

## Validation completed

- SHA256 of uploaded source archive matched the supplied checksum;
- all packaged Python files passed `compileall`;
- all command wrappers returned `--help` successfully;
- editable package installation succeeded with local build isolation disabled;
- synthetic test suite passed: **9 tests passed**;
- no unresolved `from stage...` imports remain;
- no machine-specific `/wangx/...` path remains in executable source.

## Validation still required with external artifacts

The source bundle did not contain the formal Stage 9B checkpoint or the formal
271 MB conservative cache. Before partner handoff, place those files locally
and verify:

1. checkpoint loading;
2. one snapshot inference;
3. one 30-step Stage 9B rollout;
4. output array shapes and finiteness;
5. SHA256 values recorded in `docs/DATA_AND_CHECKPOINTS.md`.
