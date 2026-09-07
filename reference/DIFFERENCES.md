# 当前改进版与 GitHub 上游项目的边界

上游项目：`kaneman777/landau-damping-surrogate`，固定参考 commit `9490dbf`。

## 谱系

```text
Antoine Tavant 1D electrostatic PIC
  -> kaneman777/landau-damping-surrogate
       PIC + (Te, Lx) 到场能曲线的 MLP/FNO 实验
    -> 当前标准化改进版
       x-v 相空间数据 + 2D FNO + 递归 Rollout + 物理闭合
```

## 主要差异

| 方面 | GitHub 上游 | 当前改进版 |
|---|---|---|
| 主要预测量 | 场能或 `log10(E)` 时间曲线 | `delta_f(x,v,t)` 相空间场 |
| 条件变量 | 主要为 `(T_e, L_x)` 或 `(T_e,L_x,t)` | `(k, alpha, t)`，Stepper 还输入当前 x-v 状态 |
| 数据组织 | `.npy` sweep 文件 | case-major HDF5，110 case × 31 时刻 |
| 模型 | MLP 与曲线型 FNO 实验 | 条件二维 FNO、一步与多步 Stepper |
| 时间推进 | 直接拟合曲线 | 自回归自由 Rollout，`dt=0.5` |
| 物理结构 | 衰减趋势等轻量约束 | mean/nonzero 分解、正性损失、密度模、Poisson、电场、能量闭合 |
| 资产管理 | 研究目录式模型/图像/数据 | 数据、模型、结果、日志、配置和哈希清单分层 |
| 测试与打包 | WIP 脚本与旧 PIC 测试 | `src/` 包结构、CLI、合同测试、真实 artifact smoke 验证 |

## 仍然继承的部分

- 1D electrostatic PIC 教学实现及其基本数据生成思路；
- Landau 阻尼代理建模的研究方向；
- 部分历史 notebook、案例脚本和命名语境。

当前项目中的 `src/landau_surrogate/pic/` 是经过包路径适配的派生代码，不等同于 `reference/upstream/.../pic/` 的原始快照。未来修改只进入当前代码，原快照保持不变。
