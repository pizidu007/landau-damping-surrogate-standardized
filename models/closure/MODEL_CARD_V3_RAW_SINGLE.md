# closure_fno_pic_v3_raw_single_research

Research-only, single-frame FNO trained on the three-seed mean CUDA-PIC
trajectory `k=0.35, A=0.10`. It maps `(M0, M1/M0, M2)` to the mode-24 filtered
`dM3/dx`.

The selected seed reaches validation/test relative L2 `0.04411/0.04798`, but
the paper-like mode-24 free rollout stops at `t=1.8`. The checkpoint is retained
to reproduce the experiment and must not replace the current deployment model.

SHA-256:
`0c98b6e89555c8152b0b6bc46d73e346eadf84216468a1b535c8fb72ebd955de`
