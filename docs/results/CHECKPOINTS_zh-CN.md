# Checkpoint 下载、校验与使用

[先看关键结果](KEY_RESULTS_zh-CN.md) · [环境安装](../operations/COLLABORATION_zh-CN.md)

权重发布在 [GitHub Release：research-assets-2026-09-09](https://github.com/pizidu007/landau-damping-surrogate-standardized/releases/tag/research-assets-2026-09-09)。
共 16 个原始 checkpoint，约 205 MiB；没有重新训练、裁剪权重或转换精度。
模型不进入 Git 历史，`git clone` 只拉取代码、精选图表与指标。
需要哪一组就下载哪一组，正式训练数据继续从共享服务器访问。

## 1. 应该下载哪个

| 下载 ID / 分组 | 数量及体积 | 用途与边界 |
|---|---|---|
| `snapshot` | 1，32.71 MiB | 第一次跑推理；旧版参数到相空间快照 |
| `rollout` | 1，65.42 MiB | 旧版完整分布递归；包含 Snapshot anchor，使用历史精确缓存 |
| `pic_closure_v1` | 1，1.74 MiB | PIC 闭合；按模型卡使用 75% HP 混合，不能把纯 FNO 当稳定部署默认值 |
| `continuum_baseline_seed1` | 1，16.33 MiB | Gkeyll 瞬时闭合与 Round 11 父初始化，研究基线 |
| `round10` 组 | 3，共约 9.14 MiB | 单帧 FNO seed 1、四帧 FNO seed 2、四帧 U-Net seed 1，既有旧测试代表 |
| `round11` 组 | 9，共约 79.7 MiB | A/B/C 各三种子；全程验证选定版本，全部未通过精度门，用于复查与研究 |

完整文件名、目标路径、大小、SHA256 和对应报告见
[机器可读清单](../../artifacts/manifests/public_release_2026-09-09.json)。
同一组中的多个种子不是多个已达标产品；`selected_full_validation.pt` 的含义是
“按既定 validation 规则选中的候选”，不等于“通过质量门”。

## 2. 按需下载并自动核对

在自己的 clone 根目录，用 Python 3.10 或更新版本运行以下命令。
下载器只用标准库，不要求先安装 PyTorch，也不要求 GitHub 登录。

```bash
# 只列出资产，不下载。
python scripts/download_checkpoints.py --list

# 第一次建议只取 Snapshot。
python scripts/download_checkpoints.py --asset snapshot

# 也可只取一个当前研究模型。
python scripts/download_checkpoints.py --asset round11_B_seed1

# 或取完整对照组。
python scripts/download_checkpoints.py --group round11
```

其他分组为 `legacy`（Snapshot、Rollout、PIC v1）、`continuum_baseline`、`round10`，以及 `all`。
默认按清单恢复到 `models/...` 或对应 `results/...` 路径，图表和报告不需要再次下载。

下载完成后核对字节数与 SHA256，再安装到目标位置。
重复运行会跳过已经校验一致的文件；若已有不同内容，直接报错并保留原文件。
下载中断或校验失败不会留下一个被当作有效 checkpoint 的目标文件。

如果自己的 clone 已用符号链接连接共享模型目录，下载器可能拒绝向仓库外写入。
可改用独立目录，不要删除或强行覆盖共享链接：

```bash
python scripts/download_checkpoints.py --asset round11_B_seed1 \
  --output-root /path/to/my/local-assets
```

文件将位于该目录下清单规定的相对路径。上面的 `/path/to/...` 是需要自行替换的示意路径。
Release 页面也支持浏览器逐个下载；其资产名与服务器原文件名不同，恢复路径以清单为准。
例如 `round11_B_seed1.pt` 对应
`results/continuum_v1_closure_round11/formal/B/seed1/selected_full_validation.pt`。

## 3. 第一次运行：无需训练数据的 Snapshot

在自己的环境安装项目后：

```bash
python scripts/download_checkpoints.py --asset snapshot
python scripts/inspect_checkpoint.py models/snapshot/snapshot_best.pt
python scripts/infer_snapshot.py \
  --checkpoint models/snapshot/snapshot_best.pt \
  --k 0.55 --alpha 0.040 --time 7.5 \
  --output results/runs/new_member_release/snapshot.npz \
  --device cpu
```

这验证权重可以加载并生成一个预测。它没有读取该条件下的真值，因而不是一次新的准确度验证。
原始训练尺度、网格和背景已经在 checkpoint 内，不要再自行按这个查询重新归一化。
PyTorch checkpoint 使用 pickle 容器；此处只加载本仓库发布且校验一致的可信文件。

旧 Rollout 需要合适的初始分布。要复现已有缓存上的结果，按
[协作指南](../operations/COLLABORATION_zh-CN.md)连接 `historical_exact` 缓存，
再使用[快速开始](../operations/QUICKSTART.md)中的 Rollout 命令。
`conservative_m02` 缓存不是它的直接替代品。

## 4. Round 11：先检查一次网络调用

以下例子只做 CPU 模型接口检查，无需 Gkeyll 数据，不生成新的物理结论：

```bash
python scripts/download_checkpoints.py --asset round11_B_seed1
python - <<'PY'
import torch
from landau_surrogate.training.continuum_history_closure import construct

torch.set_num_threads(2)
path = 'results/continuum_v1_closure_round11/formal/B/seed1/selected_full_validation.pt'
checkpoint = torch.load(path, map_location='cpu', weights_only=False)
model = construct(checkpoint['config'], checkpoint['normalization'], torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'], strict=True)
model.eval()

K, alpha = torch.tensor([0.35]), torch.tensor([0.10])
theta = 2 * torch.pi * (torch.arange(96) + 0.5) / 96
n = 1 + alpha[0] * torch.cos(theta)
u = torch.zeros_like(n)
p = n.clone()
E = -alpha[0] / K[0] * torch.sin(theta)
state = torch.stack((n, u, p, E))[None]
history = state[:, None].expand(-1, 8, -1, -1).clone()
valid = torch.zeros(1, 8, dtype=torch.bool)
valid[:, -1] = True
with torch.no_grad():
    gradient = model(history, valid, K, alpha)
print('shape:', tuple(gradient.shape))
print('finite:', bool(torch.isfinite(gradient).all()))
print('spatial mean:', float(gradient.mean()))
PY
```

预期形状为 `(1,96)`、有限性为 True、空间均值接近数值零。
这一步没有把网络输出当作真值；初态 `q=0` 也不意味着学习模型在该输入下必须返回精确零。
完整物理误差要以参考轨迹和公开报告为准。

## 5. 在共享服务器复查完整 validation

完整重评估使用 CUDA 和已有 `continuum_v1` 数据，可能需要较长时间。
先确认有可用设备，在自己的输出与缓存目录运行。不要为初次浏览而启动完整训练控制器。

```bash
python scripts/evaluate_round11_checkpoint.py \
  --checkpoint results/continuum_v1_closure_round11/formal/B/seed1/selected_full_validation.pt \
  --data-root /rydata/duxinxu/landau-damping-surrogate-standardized/continuum_v1 \
  --cache-dir results/runs/new_member_round11/cache \
  --output-dir results/runs/new_member_round11/validation \
  --device cuda:0
```

`cuda:0` 指当前进程可见的第一个设备，应按共享服务器的设备分配设置可见范围。
`--data-root` 与 `--cache-dir` 是为个人 clone 提供的路径覆盖，
它们不会改写下载的 checkpoint，实际训练与评价配置分别写入 provenance。
不同模型可复用相同数值设置的个人派生缓存，但不能混用不同频带或数据版本的缓存。

已有成功或失败的完整评价可直接从[公开指标](../../results/published/2026-09-09/round11/)读取，
无需重跑才能理解结论。不要通过调小终点、删掉失败案例或回灌真值来“复现成功”。

## 6. 发布内容与数据放在哪里

| 内容 | 位置 |
|---|---|
| 代码、指南、精选图表、JSON/CSV 指标 | 当前 Git 仓库 |
| 16 个 checkpoint、manifest、SHA256SUMS | 本次 GitHub Release |
| 完整训练数据、全部 rollout 数组、日志与中间候选 | 共享服务器原目录 |
| 下载后的模型与个人新输出 | 各自 clone 中被 `.gitignore` 排除的路径 |

完整 [assets.json](../../artifacts/manifests/assets.json) 仍描述历史服务器资产。
下载本次 16 个权重不会自动补齐其中全部数据和历史运行输出，
因此不要在新 clone 上把 `verify_assets.py --require-all` 当作本次下载成功的标准。
本次下载器按独立的公开 Release 清单核验，不会修改旧清单的口径。
