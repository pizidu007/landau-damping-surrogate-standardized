
# Release notes: source-only handoff candidate 0.1.0

## Included

- reusable snapshot and rollout HDF5 readers;
- conditional snapshot and stepper FNO implementations;
- one-step, multi-step and positivity-aware training code;
- snapshot and free-rollout inference;
- spectral, positivity, mode, moment, Poisson and field-energy diagnostics;
- conservative M0/M2 cache compression core and audit tools;
- adapted original PIC package;
- data-free synthetic contract tests;
- documentation of data, artifacts, results, limitations and stage mapping.

## Deliberately excluded

- full-resolution mother HDF5;
- compact training cache;
- all PyTorch checkpoints;
- generated figures, GIFs and logs;
- smoke/formal output directories;
- 100+ acceptance, comparison and packaging scripts;
- deprecated early MLP/DeepONet exploration branches.

## Compatibility note

Core class names and checkpoint stage strings are retained internally so the
formal research checkpoints can still be loaded after the file/module cleanup.
Actual formal-checkpoint loading remains to be verified once the checkpoint is
placed beside this source release.
