# 结果目录

- `smoke/`：从原项目保留的已验证推理样例，不应被新运行覆盖。
- `runs/<run_id>/`：新训练、评估或推理产生的结果。

一个运行目录应尽量包含：

```text
command.txt
config.json
environment.json
metrics.csv / metrics.json
*.npz
plots/
artifact_sha256.txt
```

控制台日志放入 `logs/<run_id>/`，不要与科学结果混放。
