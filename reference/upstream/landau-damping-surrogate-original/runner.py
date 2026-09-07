"""
Parameter sweep runner for dataset generation.
Each simulation → 1 HDF5 file. Skip if exists. Resumable.
"""
import os
import h5py
import hashlib
import json
import numpy as np
from simulation import run_simulation, DEFAULT_PARAMS


def params_to_id(params, vary_keys=None):
    """
    Unique string id from params (for filename).
    """
    if vary_keys is None:
        vary_keys = ["n", "Te_0", "Ti_0", "dX", "sim_time"]
    sub = {k: params[k] for k in vary_keys if k in params}
    s = json.dumps(sub, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:12]


def save_result_hdf5(result, filepath):
    """
    Save run_simulation result to single HDF5 file.
    Structure:
      /E, /phi, /rho, /ne, /ni, /Te, /ve  : datasets
      /x, /t
      attrs: params as JSON string
    """
    with h5py.File(filepath, "w") as f:
        for key in ["E", "phi", "rho", "ne", "ni", "Te", "ve", "x", "t",
                    "energy_history", "t_energy"]:
            if key in result:
                f.create_dataset(key, data=result[key], compression="gzip")
        # metadata: params (convert for JSON serializability)
        p = result.get("params", {})
        p_ser = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                 for k, v in p.items()}
        f.attrs["params"] = json.dumps(p_ser)


def load_result_hdf5(filepath):
    """Load one HDF5 run into dict."""
    with h5py.File(filepath, "r") as f:
        out = {}
        for k in f.keys():
            out[k] = f[k][...]
        if "params" in f.attrs:
            out["params"] = json.loads(f.attrs["params"])
    return out


def run_sweep(
    out_dir="data/runs",
    param_grid=None,
    vary_keys=None,
    base_params=None,
    skip_existing=True,
):
    """
    Sweep over parameter grid, run simulation, save 1 file per run.

    Parameters
    ----------
    out_dir : str
        Directory for HDF5 files.
    param_grid : dict
        Keys = param names, values = list of values.
        Example: {"n": [1e17, 3e17], "Te_0": [30, 50]}
    vary_keys : list
        Params used for unique id (default: n, Te_0, Ti_0, dX, sim_time).
    base_params : dict
        Fixed params. Overridden by param_grid.
    skip_existing : bool
        If True, skip runs where file exists.
    """
    if param_grid is None:
        param_grid = {"n": [3e17], "Te_0": [50], "Ti_0": [10]}

    vary_keys = vary_keys or list(param_grid.keys())
    base = {**DEFAULT_PARAMS, **(base_params or {})}

    os.makedirs(out_dir, exist_ok=True)

    # Cartesian product of param_grid
    keys = list(param_grid.keys())
    vals = list(param_grid.values())
    n_total = 1
    for v in vals:
        n_total *= len(v)

    idx = 0
    for combo in _product(vals):
        params = {**base, **dict(zip(keys, combo))}
        fid = params_to_id(params, vary_keys)
        filepath = os.path.join(out_dir, f"run_{fid}.h5")

        if skip_existing and os.path.exists(filepath):
            print(f"[{idx+1}/{n_total}] skip (exists): {filepath}")
            idx += 1
            continue

        print(f"[{idx+1}/{n_total}] run: {dict(zip(keys, combo))}")
        try:
            result = run_simulation(params, use_restart=False)
            save_result_hdf5(result, filepath)
            print(f"    -> {filepath}")
        except Exception as e:
            print(f"    ERROR: {e}")
        idx += 1


def _product(arrs):
    """Cartesian product of value lists."""
    if not arrs:
        yield ()
        return
    for v in arrs[0]:
        for rest in _product(arrs[1:]):
            yield (v,) + rest


if __name__ == "__main__":
    run_sweep(
        out_dir="data/runs",
        param_grid={
            "n": [2e17, 3e17],
            "Te_0": [30, 50],
            "Ti_0": [10],
        },
        base_params={"sim_time": 5e-7, "n_average": 5000},
        skip_existing=True,
    )
