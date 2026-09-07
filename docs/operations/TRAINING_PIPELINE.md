# 训练流程与当前边界

## 科学训练顺序

```text
Snapshot FNO
  -> one-step Stepper（teacher forcing）
  -> multi-step Stepper（2/4/8 curriculum）
  -> mean-anchor 候选选择
  -> positivity fine-tuning
  -> 最终 checkpoint 封装
```

公开入口：

```text
scripts/train_snapshot.py
scripts/train_one_step.py
scripts/train_multistep.py
scripts/train_positivity.py
```

## 重要限制

训练脚本保存的是研究候选 checkpoint，而后续阶段要求的是经过历史选择程序封装的 `final` checkpoint：

- Snapshot 候选：`stage8d1d_fno_ablation`；
- 一步候选：`stage8d2a_stepper`；
- 多步候选：`stage8d2b_multistep_candidate`；
- 正性候选：`stage9b_positivity_candidate`。

现有正式推理 checkpoint 使用 `stage8d1d_final_baseline` 和 `stage9b_final_positivity_rollout`。候选比较、mean-anchor 选择和 final schema 封装代码没有完整包含在当前发布范围，因此不能宣称公开脚本已经形成无人工步骤的端到端重训练流水线。

若重新训练，必须保存：实际命令、Git revision、环境、输入资产 SHA256、随机种子、训练历史、验证选择依据和最终模型卡。不要用测试集选择候选。

## PIC 热流闭合分支

新的闭合分支独立于上述相空间 Snapshot/Stepper 路线：

```text
audit_closure_labels.py
  -> evaluate_closure_baselines.py
  -> train_closure.py (4 variants x 3 seeds)
  -> select_closure_model.py
  -> train_closure_rollout.py
  -> run_fluid_closure.py
  -> plot_closure_results.py
  -> summarize_closure_study.py
```

正式配置为 `configs/training/closure_fno_v1.json`。冻结 checkpoint 位于
`models/closure/closure_fno_pic_v1.pt`，部署时必须同时使用模型卡记录的
HP blend、谱截止和 RK4 子步，不能把 checkpoint 单独视为稳定的自由
rollout 模型。
