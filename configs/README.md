# 配置目录

- `environment/conda.yml`：可移植的参考 Conda 环境。
- `paths.example.toml`：项目内标准资产路径。
- `data/nonlinear_pic_v1.json`：强非线性 CUDA-PIC 的 smoke/pilot/formal 合同。
- `cli_presets/`：已验证命令的参数参考。

当前训练和推理程序仍以命令行参数为正式接口，`cli_presets/*.json` 不会被脚本自动读取，避免出现配置文件与 CLI 隐式覆盖。使用时请将其中参数显式传给对应脚本，并把实际命令、环境和日志保存在同一个 `results/runs/<run_id>` 目录中。

大型数据例外：PIC 生成输出统一写入 `$LANDAU_DATA_ROOT/nonlinear/runs/`，不写入源码树。
