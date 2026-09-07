# Run from repo root: python surrogate/quick_test.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import LandauSurrogate
import os
import time
import glob
import argparse
from datetime import datetime


def _find_matching_pic_data(
    te,
    lx,
    data_dir="sweep_2d_results/data",
    te_tol_rel=0.01,
    lx_tol_rel=0.01,
):
    """
    Find PIC .npy whose stored (te, lx) is closest to query.
    Returns (dict, rel_err_te, rel_err_lx) if within tolerances, else (None, ...).
    """
    files = glob.glob(os.path.join(data_dir, "*.npy"))
    if not files:
        files = glob.glob(os.path.join("../", data_dir, "*.npy"))
    if not files:
        return None, float("inf"), float("inf")

    best = None
    best_te_rel = float("inf")
    best_lx_rel = float("inf")
    best_dist = float("inf")
    for f in files:
        try:
            d = np.load(f, allow_pickle=True).item()
            te_f = float(d.get("te"))
            lx_f = float(d.get("lx"))
            te_rel = abs(te_f - te) / max(abs(te), 1.0)
            lx_rel = abs(lx_f - lx) / max(abs(lx), 1e-12)
            dist = te_rel + lx_rel
            if dist < best_dist:
                best_dist = dist
                best = d
                best_te_rel = te_rel
                best_lx_rel = lx_rel
        except Exception:
            continue
    if best is None:
        return None, float("inf"), float("inf")
    if best_te_rel > te_tol_rel or best_lx_rel > lx_tol_rel:
        return None, best_te_rel, best_lx_rel
    return best, best_te_rel, best_lx_rel

def quick_check(te, lx, te_tol_rel=0.01, lx_tol_rel=0.01, output_dir="surrogate/outputs"):
    # 1. Load model
    model = LandauSurrogate()
    model.load_state_dict(torch.load("surrogate/models/landau_model.pth"))
    model.eval()
    
    # 2. Load normalization parameters
    norm = torch.load("surrogate/models/norm_params.pth")
    in_min, in_max = norm["min"], norm["max"]
    out_mean = norm.get("out_mean", torch.tensor([0.0]))
    out_std = norm.get("out_std", torch.tensor([1.0]))
    
    # 3. Predict
    t_eval = np.linspace(0, 1e-7, 500)
    inputs = torch.tensor([[te, lx, t] for t in t_eval], dtype=torch.float32)
    inputs_norm = (inputs - in_min) / (in_max - in_min)
    
    # --- Time batch inference (500 points) ---
    start_time = time.perf_counter()  # High-resolution timer
    with torch.no_grad():
        y_pred_norm = model(inputs_norm)
        log_e_pred = (y_pred_norm * out_std + out_mean).flatten().numpy()

    # 4. (Optional) Overlay original PIC data if present
    plt.figure(figsize=(10, 6))
    
    end_time = time.perf_counter()
    runtime_ms = (end_time - start_time) * 1000  # Convert to ms

    pic_data, te_rel, lx_rel = _find_matching_pic_data(
        te, lx, te_tol_rel=te_tol_rel, lx_tol_rel=lx_tol_rel
    )
    if pic_data is not None:
        plt.semilogy(pic_data['t'], pic_data['energy'], 'b--', alpha=0.5, label='Original PIC')
    else:
        print(
            f"No PIC overlay for Te={te}, Lx={lx}: "
            f"nearest rel errors (Te={te_rel:.3%}, Lx={lx_rel:.3%}) exceed "
            f"tol (Te={te_tol_rel:.3%}, Lx={lx_tol_rel:.3%})."
        )
    
    plt.semilogy(t_eval, 10**log_e_pred, 'r-', linewidth=2, label=f'AI Surrogate ({te}eV)')
   
    plt.title(f"AI Prediction (Te={te}eV, Lx={lx}cm)\nRuntime: {runtime_ms:.4f} ms")
    # ----------------------------------
    plt.xlabel("Time [s]")
    plt.ylabel("Field Energy")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    os.makedirs(output_dir, exist_ok=True)
    out_png = os.path.join(output_dir, f"mlp_quick_te{int(te)}_lx{lx:.4f}.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved plot: {out_png}")

    return runtime_ms

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_tag", type=str, default="", help="Optional suffix for output folder name")
    parser.add_argument("--te_tol_rel", type=float, default=0.01, help="Relative tolerance for Te match")
    parser.add_argument("--lx_tol_rel", type=float, default=0.01, help="Relative tolerance for Lx match")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.run_tag}" if args.run_tag else ""
    output_dir = os.path.join("surrogate/outputs", f"run_{ts}{suffix}")

    # Example sweep over electron temperatures (eV)
    temperatures = [100, 500, 1000, 1500, 2000]
    runtimes = []
    for t in temperatures:
        rt = quick_check(
            t,
            0.01,
            te_tol_rel=args.te_tol_rel,
            lx_tol_rel=args.lx_tol_rel,
            output_dir=output_dir,
        )
        runtimes.append(rt)

    # Print mean runtime across temperatures
    print("\n" + "="*30)
    print(f"Average Runtime: {np.mean(runtimes):.4f} ms")
    print(f"Output directory: {output_dir}")
    print("="*30)