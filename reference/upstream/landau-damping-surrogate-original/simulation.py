"""
Simulation runner for 1D PIC.
Provides run_simulation(params) for programmatic use and dataset generation.
"""
import numpy as np
from pic.plasma import plasma


DEFAULT_PARAMS = {
    "Lx": 1e-2,
    "dX": 5e-5,
    "n": 3e17,
    "dT": 3e-12,
    "Te_0": 50,
    "Ti_0": 10,
    "Npart_factor": 10,
    "n_average": 5000,
    "sim_time": 1e-7,
    "verbose": True,
}


def run_simulation(params=None, use_restart=False, restart_path="data/restart_pla.dat"):
    """
    Run 1D PIC simulation and return field history.

    Parameters
    ----------
    params : dict, optional
        Simulation parameters. Missing keys use DEFAULT_PARAMS.
    use_restart : bool
        If True, try to load restart file (must match params).
    restart_path : str
        Path to restart pickle file.

    Returns
    -------
    result : dict
        E, phi, rho, ne, ni, Te, ve, x, t, params, energy_history, t_energy
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    Lx = p["Lx"]
    dX = p["dX"]
    n = p["n"]
    dT = p["dT"]
    Te_0 = p["Te_0"]
    Ti_0 = p["Ti_0"]
    Npart_factor = p["Npart_factor"]
    n_average = p["n_average"]
    sim_time = p["sim_time"]
    verbose = p.get("verbose", True)

    Nx = int(Lx / dX)
    Lx = Nx * dX
    Npart = Npart_factor * Nx
    Nt = int(sim_time / dT)

    if verbose:
        print(f"Nx={Nx}, Npart={Npart}, Nt={Nt}")

    pla = None
    if use_restart:
        try:
            import pickle
            pla = pickle.load(open(restart_path, "rb"))
            if pla.Nx != Nx or pla.n != n:
                pla = None
        except Exception:
            pla = None

    if pla is None:
        pla = plasma(dT, Nx, Lx, Npart, n, Te_0, Ti_0, n_average=n_average)

    if not pla.v:
        raise RuntimeError("Plasma init validation failed")

    pla.Do_diags = True
    pla.n_0 = 0

    # Initial perturbation for Landau damping (single mode k = 2*pi/Lx)
    pla.ele.x += 0.05 * np.cos((2 * np.pi / Lx) * pla.ele.x)

    E_list = []
    phi_list = []
    rho_list = []
    ne_list = []
    ni_list = []
    Te_list = []
    ve_list = []
    t_list = []
    energy_history = []
    t_energy = []

    # Initial E from perturbed density, then save energy at t=0
    pla.compute_rho()
    pla.solve_poisson()
    energy_history.append(float(np.sum(pla.E**2)))
    t_energy.append(0.0)

    for nt in range(Nt):
        pla.pusher()
        pla.boundary()
        pla.compute_rho()
        pla.solve_poisson()
        pla.diags(nt)

        energy_history.append(float(np.sum(pla.E**2)))
        t_energy.append((nt + 1) * pla.dT)

        if (nt - pla.n_0 + 1) % pla.n_average == 0:
            d = pla.data[pla.lastkey]
            phi = d["phi"]
            E = -np.gradient(phi, pla.dx)

            E_list.append(E.copy())
            phi_list.append(phi.copy())
            rho_list.append(d["rho"].copy())
            ne_list.append(d["ne"].copy())
            ni_list.append(d["ni"].copy())
            Te_list.append(d["Te"].copy())
            ve_list.append(d["ve"].copy())
            t_list.append(nt * pla.dT)

            if verbose and (nt % (10 * pla.n_average) == 0 or nt == Nt - 1):
                print(f"\r t = {nt*pla.dT*1e6:.4f} / {Nt*pla.dT*1e6:.4f} μs", end="")

    if verbose:
        print()

    return {
        "E": np.array(E_list),
        "phi": np.array(phi_list),
        "rho": np.array(rho_list),
        "ne": np.array(ne_list),
        "ni": np.array(ni_list),
        "Te": np.array(Te_list),
        "ve": np.array(ve_list),
        "x": pla.x_j.copy(),
        "t": np.array(t_list),
        "energy_history": np.array(energy_history),
        "t_energy": np.array(t_energy),
        "params": p,
    }
