# CUDA 非线性 PIC 数据生成

正式数据生成只允许 CUDA，不提供隐式 CPU fallback。入口为：

```bash
python scripts/generate_nonlinear_pic.py \
  --profile smoke \
  --run-id audit_smoke_v1 \
  --device cuda:0
```

默认输出到：

```text
$LANDAU_DATA_ROOT/nonlinear/runs/<run_id>/cases/*.h5
```

## 三档配置

- `smoke`：16384 粒子、10 步，仅验证 CUDA、HDF5 和 M0--M3 合同；
- `pilot`：3,276,800 粒子/case，`dt=0.05`，演化到 `t=40`，3 个代表 case；
- `formal`：26,214,400 粒子/case，`dt=0.05`，演化到 `t=60`，60 个 case。

正式配置覆盖 `k=[0.30,0.35,0.40,0.45,0.55]`、
`alpha=[0.05,0.075,0.10,0.125]` 和 3 个 quiet-start 离散化副本。
所有 seed 跟随同一 `(k,alpha)` split，避免轨迹级泄漏。

## 多 GPU 分片

每个 GPU 启动一个独立进程；不同进程写不同 case 文件，不共享 HDF5：

```bash
python scripts/generate_nonlinear_pic.py \
  --profile formal --run-id nonlinear_formal_v1 \
  --device cuda:0 --shard-count 8 --shard-index 0 --resume
```

其余 GPU 使用相同命令，将 `device` 和 `shard-index` 分别改为 1--7。
启动前必须用 `nvidia-smi` 确认显存和作业归属；本工具不会抢占或终止其他任务。

## 输出合同

每个 case 保存：

- `raw_moments/m0..m3`；
- `fluid/density, velocity, pressure, temperature`；
- `fluid/heat_flux, heat_flux_gradient`；
- `fields/electric` 与三种能量；
- 稀疏 `phase_space/f`；
- CUDA 运行时间、峰值显存、能量漂移和零均值热流散度审计。

case 先写入 `.incomplete` 临时文件，验证有限性后原子改名；`--resume` 只跳过
`status=COMPLETE` 的文件。

全部分片结束后运行：

```bash
python scripts/audit_nonlinear_pic.py \
  --profile formal \
  --run-dir "$LANDAU_DATA_ROOT/nonlinear/runs/nonlinear_formal_v1" \
  --require-complete
```

## 当前正式产物

`nonlinear_formal_v1` 已完成 60/60 case 并通过审计。数据划分为
30 train、15 validation、15 test；case 文件合计约 5.43 GiB。最大总能量
相对跨度为 `5.1272e-5`，速度域溢出率为 0，最大热流梯度空间均值绝对值为
`1.7863e-9`。完整汇总由审计程序写入数据根下的 `summary.json` 和
`case_metrics.csv`。
