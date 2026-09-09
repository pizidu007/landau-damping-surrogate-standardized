# 小型 FNO + SSM + 残差闭合实验包

解压后在本目录操作即可，不需要服务器，也不用再下载数据或模型。

```text
README.md        先读这一页
data/            完整数组数据：139 train、20 validation、36 旧 diagnostic test
checkpoints/     旧 Round11 A/B/C，各一个相同 seed 0 的 checkpoint
src/             模型、数据接口和流体推进代码
scripts/         训练与比较入口
docs/            数据说明、路线原理和项目扫盲，按需查阅
tests/           小型检查
```

数据是完整 `t=0–80` 的 `[n,u,p,E]` 和 `dq/dx`，128 个空间点、时间间隔 0.02。
保留原来的训练/验证划分和精度，无损压缩后整个实验包约 1.1 GB。
默认使用全部 139 条训练轨迹，按连续时间块加载到 GPU，适合先在 3060 笔记本上尝试。

## 1. 安装一次

建议使用独立 Python 3.11 环境。如果已经有可用的 GPU PyTorch，直接执行第二条即可。

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e .
```

这组 PyTorch 安装命令来自[官方旧版本安装页](https://pytorch.org/get-started/previous-versions/)。
无须安装 Gkeyll 或额外的 Mamba/SSM CUDA 扩展。

## 2. 开始训练

```bash
python scripts/train_pc_ssm.py
```

默认自动选 GPU、使用完整训练数据，训练小型 FNO + 对角 SSM + 非线性残差闭合。
先跑 3 个 epoch；输出在 `runs/ssm/`，其中 `best.pt` 按离线 validation 标签误差选取。
它是可改造的起始实现，不是已验证达到物理精度目标的模型。
若显存不足，先加 `--chunk 32`；增加训练可加 `--epochs 10`。
重跑时通过 `--output-dir runs/另一个名称` 保留之前结果。

## 3. 和旧模型比较

```bash
python scripts/compare_pc_closures.py --checkpoint runs/ssm/best.pt
```

同一命令会逐个评价新模型、线性 HP 基线和旧 A/B/C，对完整 20 条 validation 自由推进到 t=80。
结果在 `runs/comparison/`：看 `comparison.csv`、`field_energy.png` 和 `paired.json`。
首次想快速检查入口，可在命令末尾加 `--horizon 2 --limit 1`；这只是短程检查。

下一步只改 `--variant` 就能做消融：`fno`、`fno_ssm`、`fno_residual`、`fno_ssm_residual`。
例如 `python scripts/train_pc_ssm.py --variant fno_residual --output-dir runs/no_memory`。

理解路线时记住：FNO 处理空间，SSM 保存过去状态，非线性头修正线性热流闭合。
训练按案例从 t=0 连续递推；自由推进时只喂自己的预测，不补真实历史。
先比较完成率与误差；旧 A/B/C 本身也没有通过 Round11 的完整精度门槛。

需要改模型时，从 `src/landau_surrogate/models/pc_fno_ssm.py` 开始。
需要理解数组和比较口径时，再看 [详细说明](05_PC_FNO_SSM_TASK_zh-CN.md)。
旧测试数据已放在 `data/cases/test/`，默认训练和比较均不会使用；它不是新的盲测集。
