# Lightweight evaluation / plotting helper for the MLP surrogate
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import LandauSurrogate

def quick_check(te, lx):
    model = LandauSurrogate()
    model.load_state_dict(torch.load("surrogate/models/landau_model.pth"))
    model.eval()
    
    # Load normalization parameters
    norm = torch.load("surrogate/models/norm_params.pth")
    in_min, in_max = norm["min"], norm["max"]
    out_mean = norm.get("out_mean", torch.tensor([0.0]))
    out_std = norm.get("out_std", torch.tensor([1.0]))
    
    # Time axis for prediction
    t_eval = np.linspace(0, 1e-7, 500)
    inputs = torch.tensor([[te, lx, t] for t in t_eval], dtype=torch.float32)
    inputs_norm = (inputs - in_min) / (in_max - in_min)
    
    with torch.no_grad():
        y_pred_norm = model(inputs_norm)
        log_e_pred = (y_pred_norm * out_std + out_mean).numpy()
    
    plt.semilogy(t_eval, 10**log_e_pred, 'r-', label=f'AI: {te}eV')
    plt.title(f"AI Prediction at Te={te}eV, Lx={lx}cm")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.show()

# After training, uncomment to run:
# quick_check(te=500, lx=0.01)