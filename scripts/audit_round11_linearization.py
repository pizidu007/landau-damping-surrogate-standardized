"""Read-only linear stability audit of the instantaneous A closure near equilibrium.

This does not approximate a history closure by an instantaneous one. Only A is
accepted, and the Gauss-compatible three-variable linear fluid system is used.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from landau_surrogate.training.continuum_history_closure import construct


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    checkpoint=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    config=copy.deepcopy(checkpoint["config"])
    if config["history_span"]!=0:raise ValueError("This audit is exact only for instantaneous A")
    model=construct(config,checkpoint["normalization"],torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"]);model.eval();model.requires_grad_(False)
    theta=(torch.arange(128)+.5)*(2*torch.pi/128)
    rows=[]
    for reflected in (False,True):
        model.enforce_reflection=reflected
        for k,alpha in ((.445,.030),(.472,.032),(.474,.051)):
            pressure=1.+alpha**2/(2*k**2)
            equilibrium=torch.zeros(1,4,128)
            equilibrium[:,0]=1.;equilibrium[:,2]=pressure
            def closure(state):
                return model(state[:,None].expand(-1,8,-1,-1),torch.ones(1,8,dtype=torch.bool),
                             torch.tensor([k]),torch.tensor([alpha]))
            for mode in range(1,config["maximum_mode"]+1):
                wave=k*mode;signal=torch.cos(mode*theta)
                amplitude=torch.fft.rfft(signal,norm="forward")[mode]
                response=[]
                for channel in range(4):
                    direction=torch.zeros_like(equilibrium);direction[:,channel]=signal
                    _,gradient=torch.autograd.functional.jvp(closure,equilibrium,direction)
                    response.append(complex((torch.fft.rfft(gradient,norm="forward")[0,mode]/amplitude).item()))
                hn,hu,hp,he=response
                # Gauss: i*wave*E=-delta_n, hence E=i*delta_n/wave.
                matrix=np.array([[0.,-1j*wave,0.],[-1j/wave,0.,-1j*wave],
                                 [-hn-1j*he/wave,-3j*pressure*wave-hu,-hp]],dtype=np.complex128)
                eigen=np.linalg.eigvals(matrix)
                z=eigen*config["dt"]
                amplification=np.abs(1+z+z**2/2+z**3/6+z**4/24)
                rows.append({"reflection":reflected,"K":k,"alpha":alpha,"pressure_equilibrium":pressure,
                             "mode":mode,"physical_wave_number":wave,
                             "max_real_eigenvalue":float(eigen.real.max()),
                             "max_rk4_amplification":float(amplification.max()),
                             "closure_derivatives_n_u_p_E":[[v.real,v.imag] for v in response],
                             "eigenvalues":[[float(v.real),float(v.imag)] for v in eigen]})
    report={"checkpoint":str(args.checkpoint.resolve()),"training_performed":False,"test_used":False,
            "scope":"Instantaneous A only, three existing weak validation parameters, homogeneous equilibria with converted initial field energy. Continuous linearized growth and RK4 amplification; this alone does not establish the cause of a nonlinear failure.",
            "significant_growth_threshold":.001,"cases":rows}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+"\n")
    for reflected in (False,True):
        group=[r for r in rows if r["reflection"]==reflected]
        positive=[r for r in group if r["max_real_eigenvalue"]>.001]
        worst=max(group,key=lambda r:r["max_real_eigenvalue"])
        fundamental=[r for r in group if r["mode"]==1]
        print(json.dumps({"reflection":reflected,"unstable_modes":len(positive),"total_modes":len(group),
                          "maximum_growth":worst["max_real_eigenvalue"],"worst_mode":worst["mode"],
                          "fundamental_growth":[r["max_real_eigenvalue"] for r in fundamental]}),flush=True)


if __name__=="__main__":main()
