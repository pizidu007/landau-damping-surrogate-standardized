# 新成员协作与共享服务器入门

本仓库公开维护代码，正式数据与实验产物通过已有共享服务器访问。
公开仓库本身不授予服务器访问权限；以下路径适用于已经有访问权限的组内成员。

如果还不清楚项目研究什么，请先读[新成员学习路线](../onboarding/README.md)。
其中的背景、项目逻辑与任务定义介绍物理概念和工作目标，本篇负责安装与服务器操作。

## 1. 先认识当前项目

- `src/landau_surrogate/`：数据接口、模型、流体/PIC 求解器、训练和评估实现。
- `scripts/`：命令行入口；`configs/`：数据生成与训练配置。
- `tests/`：使用小型合成数据的测试，不要求正式数据或 GPU。
- `docs/experiments/LANDAU_CLOSURE_ROUND11_PROTOCOL_zh-CN.md`：当前有记忆热流闭合实验的协议、质量门和运行入口。
- `docs/reports/`：各阶段的研究报告；请区分历史结果与当前实验结论。
- `reference/`：固定上游快照和历史材料，只读保留；修改应进入当前源码。

完整数据、checkpoint、运行日志和日常生成的产物不进入 Git。
精选图表与指标保存在 `results/published/`，选定权重通过 GitHub Release 下载。
资产清单、模型卡、冻结配置和文字报告用于说明来源与复现口径。
历史文档中的服务器绝对路径是已有实验记录，跨机器运行时需要调整。

## 2. 在自己的目录和环境中开发

在 GitHub 仓库页面复制克隆地址，在自己的工作目录执行 `git clone`，
然后进入克隆目录。不要在共享的正式实验目录中直接修改代码或安装个人分支。

在同一服务器上，可克隆已有环境后安装自己的源码：

```bash
conda create -n landau-dev --clone /wangx/home/duxinxu/miniconda3/envs/landau-pic-surrogate
conda activate landau-dev
python -m pip install -e '.[dev]'
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q
```

本次发布准备使用的环境为 Python 3.11.13、PyTorch 2.5.1+cu121。
异机安装可从 `configs/environment/conda.yml` 创建环境，再安装适配本机
CUDA/CPU 的 PyTorch 和本项目。不要把 `pip install -e .` 指向另一位成员的共享环境。

## 3. 连接共享数据

```bash
export LANDAU_DATA_ROOT=/wangx/home/duxinxu/datasets/landau-damping-surrogate-standardized
export LANDAU_SHARED_PROJECT=/wangx/home/duxinxu/projects/landau-damping-surrogate-standardized
test -r "$LANDAU_DATA_ROOT/processed/historical_exact/landau_deltaf_cache_x128_v193_v1.h5"
```

`LANDAU_DATA_ROOT` 控制使用统一数据路径模块的入口；部分历史配置和脚本仍有
绝对路径，它不会替换所有 JSON、Shell 脚本中的路径。新实验先复制配置为
自己的文件，检查数据路径、checkpoint、输出目录和设备编号。

旧推理命令使用 `data/processed/...`，新克隆目录可以建立兼容链接：

```bash
test -e data/processed || test -L data/processed || ln -s "$LANDAU_DATA_ROOT/processed" data/processed
```

Snapshot/原始 Rollout 模型使用 `historical_exact` 缓存。
`conservative_m02` 用于保守矩审计，不能作为现有模型的等价缓存替换。
Gkeyll 和 Round 11 使用独立的数据合同及冻结划分，请按相应配置读取。

## 4. 连接已有模型并试运行

也可以按[公开 checkpoint 指南](../results/CHECKPOINTS_zh-CN.md)直接下载校验后的模型。
只看结果无需连接服务器，参见[关键结果与可视化](../results/KEY_RESULTS_zh-CN.md)。
下面的符号链接方式适用于已经能读取共享模型的同机成员。

以下命令只为缺失的模型建立符号链接，保留本地已有文件：

```bash
python - <<'PY'
import os
from pathlib import Path

shared = Path(os.environ['LANDAU_SHARED_PROJECT'])
for source in sorted((shared / 'models').rglob('*')):
    if source.suffix not in {'.pt', '.pth', '.ckpt'} or not source.is_file():
        continue
    target = Path('models') / source.relative_to(shared / 'models')
    if target.exists() or target.is_symlink():
        continue
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source)
    print(target, '->', source)
PY

python scripts/inspect_checkpoint.py --help
python scripts/infer_snapshot.py \
  --checkpoint models/snapshot/snapshot_best.pt \
  --k 0.55 --alpha 0.040 --time 7.5 \
  --output results/runs/new_member_smoke/snapshot.npz \
  --device cpu
```

模型链接用于读取；新训练的 checkpoint 写入自己的 `results/runs/<run_id>/`。
只在可信的组内 checkpoint 上使用 PyTorch 加载。

`python scripts/verify_assets.py` 会逐项报告 OK/MISSING/MISMATCH。
刚克隆的源码仓库没有历史运行产物，出现对应 MISSING 是预期情况；
`--require-all` 仅适用于已经备齐清单内全部数据、模型和历史输出的工作目录。

## 5. 查看或复现当前实验

共享结果位于 `$LANDAU_SHARED_PROJECT/results/`，Round 11 状态页为
`continuum_v1_closure_round11/RUN_STATUS.md`。
公开仓库中的报告可能引用共享目录下的图表和数值产物；这些链接在纯源码
克隆中没有对应文件，需要到共享结果目录查看。

Round 11 的 `formal_overrides.json` 是正式实验配置的一部分，随代码保存；
监督模型、归一化产物、冻结划分和训练输出仍需从共享服务器读取。
重跑前先阅读协议和 CLI 参数，不要直接启动会等待/续写正式任务的控制器。
新实验指定自己的输出目录和空闲设备，保留 train/validation/test 的划分与质量门。
服务器上的绝对路径、既有设备编号和正在更新的状态页不是通用运行默认值。

## 6. 提交改动

每位成员使用自己的 clone 和功能分支，通过 Pull Request 评审代码。
公开仓库可直接克隆；写权限需由仓库管理员添加协作者，或通过个人 fork 提交 PR。

提交前运行相关测试并查看 `git diff --cached --stat`。
`.gitignore` 已排除数据、权重、输出、缓存、日志和常见本地凭据文件；
不要使用 `git add -f` 把这些资产强行加入仓库。
涉及数据或训练的改动同时记录配置、随机种子、数据版本和可复现命令。

保留根目录 GNU GPL v3 许可证和 `THIRD_PARTY_NOTICES.md` 中的上游署名。
