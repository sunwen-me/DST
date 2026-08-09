# `both` 双路径修复记录

记录日期：2026-07-28

适用项目：DST-Calib 的 Livox Mid-360 + Orbbec Gemini 335 工程实现。

## 1. 背景

项目的 `both` 模式组合了两条路径：

1. 评估模块（SB）从初始外参快速估计较好的先验；
2. 位姿估计器（PE）使用式 (19) 的自监督损失继续优化；
3. 对逐帧候选打分并执行多帧融合。

修复前的慢测中，`fast` 可以把旋转误差从 `1.115°` 降到 `0.470°`，但
`both` 最终结果反而退化到 `1.149°`，未通过项目规定的
`e_r < 1°、e_t < 0.1 m` 验收门槛。

## 2. 根因

修复前，PE 完成全局优化和逐帧微调后，代码会针对每个 PE 候选调用一次
`eva_refine()`：

```text
PE 候选 T_i ──SB──> 精化结果 T_ref
```

但 `T_ref` 只被用于计算式 (20) 的分数，最终多帧平均仍使用原始 PE 候选
`T_i`。因此 SB 产生的外参修正被丢弃：

```text
旧流程：
PE 候选 ──SB──> T_ref ──只用于打分
   │
   └──────────────────────> 最终多帧融合
```

这与 PE+SB 的组合含义不一致，也是 `both` 可能劣于 `fast` 的主要原因。

## 3. 修复后的流程

现在 PE 候选会真正作为 SB 的先验，SB 精化结果进入最终融合：

```text
初始外参
   │
   ▼
fast / SB 快速估计
   │  T_eva，同时作为式 (18) 的评估先验
   ▼
PE 自监督优化（L_t_ini + L_CD + L'_eva）
   │
   ▼
逐帧 PE 候选
   │
   ▼
SB 最终精化
   │
   ├──> 再前向一次，按式 (20) 计算一致性分数
   ▼
式 (22)(23) 多帧选择与加权平均
   │
   ▼
最终外参 T*
```

核心实现位于 `dst_calib/calibrate.py` 的 `_calibrate_with_eva()`。

## 4. 精化轮数解耦

新增配置：

```yaml
inference:
  eva_iters: 3
  both_eva_iters: 3
```

- `eva_iters`：控制 fast 从名义外参或外部先验开始的 SB 迭代次数。
- `both_eva_iters`：控制 PE 已进入较小邻域后，SB 最终精化的迭代次数。

两者需要分开，是因为欠训练的评估模块从较粗初值反复更新时可能累积偏置，而
PE 后的候选已经位于更小的收敛邻域，可以使用至少 3 次 SB 精化消除残余误差。

旧配置没有 `both_eva_iters` 时，代码使用：

```python
max(eva_iters, 3)
```

作为兼容缺省值。

诊断使用的新训练断点中，不同最终精化次数的结果如下：

| `both_eva_iters` | 旋转误差 | 平移误差 |
|---:|---:|---:|
| 1 | 1.022° | 0.0422 m |
| 2 | 1.001° | 0.0472 m |
| 3 | 0.356° | 0.0572 m |
| 4 | 0.386° | 0.0560 m |

因此默认取 3。

## 5. fast 非退化保护

`both` 从 fast 结果出发，正常情况下不应在相同的自监督主目标上变差。最终融合后，
代码会使用同一批帧、末段体素大小、截断距离和确定性下采样，分别计算：

```text
CD_both = mean_final_cd(T_both)
CD_fast = mean_final_cd(T_fast)
```

当 `CD_both` 非有限或严格大于 `CD_fast` 时，最终结果自动回退到 fast：

```text
score_method: <原打分方式>_fast_guard
```

正常完成 PE→SB 时输出：

```text
score_method: full_supervised_after_pe_sb
```

该字段会写入 `extrinsic.yaml`，可用于确认实际选择的路径。

## 6. 验证结果

### 针对原失败数据

同一组 12 帧、同一断点和同一初始外参：

| 版本/阶段 | 旋转误差 | 平移误差 | Chamfer |
|---|---:|---:|---:|
| 初始外参 | 1.115° | 0.0347 m | — |
| fast | 0.470° | 0.0440 m | 0.01914 |
| 修复前 both | 1.149° | 0.0438 m | 0.01976 |
| 修复后 both | 0.413° | 0.0587 m | 0.01887 |

### 从零重训验收

完整慢测会重新生成 3 个场景、训练新的 SB 断点，再分别运行 fast 和 both。
最终结果：

```text
3 passed, 76 deselected in 280.94s
```

该轮 `both` 实际指标：

```text
rotation error    = 0.4588°
translation error = 0.0572 m
final_cd          = 0.01912
score_method      = full_supervised_after_pe_sb
both_eva_iters    = 3
```

基础测试结果：

```text
76 passed, 3 deselected in 50.76s
```

总计 `79/79` 项测试通过。

## 7. 涉及文件

| 文件 | 修改内容 |
|---|---|
| `dst_calib/calibrate.py` | PE 后真正执行并采用 SB 精化结果；增加式 (20) 打分和 fast 非退化保护 |
| `config/default.yaml` | 新增 `inference.both_eva_iters` |
| `tests/test_full_paper_form.py` | 验证输出确实采用 `full_supervised_after_pe_sb` |
| `README.md` | 更新 both 数据流、配置和回退语义 |
| `INTERFACES.md` | 更新 A3 模式契约 |

## 8. 后续排查

如果真实数据上的 `both` 结果异常，依次检查：

1. 查看 `extrinsic.yaml` 的 `score_method`：
   - `full_supervised_after_pe_sb`：完整 PE→SB 路径；
   - 以 `_fast_guard` 结尾：组合结果被判定劣于 fast；
   - `self_supervised_fallback (...)`：SB 精化或式 (20) 打分发生异常。
2. 比较 `final_cd` 与同数据 `fast` 模式的结果。
3. 查看 `loss_curve.png`，确认 PE 最后阶段没有发散。
4. 检查评估模块断点是否来自相同虚拟相机参数和相近传感器配置。
5. 若 fast 正常但 PE→SB 不稳定，优先尝试：
   - `both_eva_iters: 3` 或 `4`；
   - 增加结构丰富的训练场景；
   - 检查训练 session 使用的基准外参是否可靠。

建议不要通过放宽 `<1° / <0.1 m` 测试门槛掩盖回归。
