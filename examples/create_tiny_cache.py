
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def create_tiny_cache(path: Path) -> Path:
    case_count, time_count, nx, nv = 3, 4, 8, 9
    x = np.linspace(0.0, 1.0, nx, endpoint=False, dtype=np.float32)
    v = np.linspace(-4.0, 4.0, nv, dtype=np.float32)
    t = np.arange(time_count, dtype=np.float32) * 0.5
    split = np.asarray([0, 1, 2], dtype=np.uint8)
    groups = np.asarray([0, 1, 2], dtype=np.uint8)
    k = np.asarray([0.45, 0.55, 0.65], dtype=np.float32)
    alpha = np.asarray([0.005, 0.025, 0.050], dtype=np.float32)
    phase_velocity = np.asarray([3.0, 2.7, 2.5], dtype=np.float32)

    fields = []
    conditions = []
    physical = []
    sample_case = []
    sample_time = []
    sample_split = []
    sample_group = []
    for c in range(case_count):
        for j in range(time_count):
            xx = 2.0 * np.pi * x[:, None]
            vv = v[None, :]
            field = alpha[c] * np.cos(xx - 0.3 * j) * np.exp(-0.5 * vv**2)
            fields.append(field.astype(np.float32))
            conditions.append(np.asarray([0.0, 0.0, -1.0 + 2.0*j/(time_count-1)], dtype=np.float32))
            physical.append(np.asarray([k[c], alpha[c], t[j]], dtype=np.float32))
            sample_case.append(c)
            sample_time.append(j)
            sample_split.append(split[c])
            sample_group.append(groups[c])

    with h5py.File(path, "w") as h:
        h.attrs["status"] = "COMPLETE"
        h.attrs["dataset_version"] = "synthetic-test-v1"
        h.attrs["source_dataset_sha256"] = "synthetic"
        h.attrs["normalization_sha256"] = "synthetic"
        h.attrs["x_stride"] = 1
        h.attrs["v_stride"] = 1
        g = h.create_group("grids")
        g.create_dataset("normalized_x", data=x)
        g.create_dataset("velocity", data=v)
        g.create_dataset("phase_time", data=t)
        cgrp = h.create_group("cases")
        cgrp.create_dataset("case_id", data=np.asarray([b"train", b"val", b"test"]))
        cgrp.create_dataset("k", data=k)
        cgrp.create_dataset("alpha", data=alpha)
        cgrp.create_dataset("phase_velocity", data=phase_velocity)
        cgrp.create_dataset("split_code", data=split)
        cgrp.create_dataset("group_code", data=groups)
        s = h.create_group("samples")
        s.create_dataset("field", data=np.asarray(fields, dtype=np.float32))
        s.create_dataset("condition", data=np.asarray(conditions, dtype=np.float32))
        s.create_dataset("physical_condition", data=np.asarray(physical, dtype=np.float32))
        s.create_dataset("case_index", data=np.asarray(sample_case, dtype=np.int32))
        s.create_dataset("time_index", data=np.asarray(sample_time, dtype=np.int16))
        s.create_dataset("split_code", data=np.asarray(sample_split, dtype=np.uint8))
        s.create_dataset("group_code", data=np.asarray(sample_group, dtype=np.uint8))
    return path


if __name__ == "__main__":
    output = Path("examples/tiny_cache.h5")
    output.parent.mkdir(parents=True, exist_ok=True)
    print(create_tiny_cache(output))
