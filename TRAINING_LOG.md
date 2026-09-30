# ChestMem-AD RSNA 训练实验日志

记录时间：2026-09-30（全部训练跑于 2026-09-29）
项目根目录：仓库根目录
所有实验产物在 `outputs/<实验名>/`（检查点、metrics.json、可视化）。

---

## 0. 基础设置

| 项目 | 内容 |
|---|---|
| 数据集 | Med-AD v1 的 RSNA 部分（Zenodo 完整版，解压于 `datasets/Med-AD_v1/RSNA`，26,684 张图） |
| 训练划分 | 已知正常 3,851 张；无标签 4,000 张（异常率 0.6 → 2,400 异常 + 1,600 正常，按 manifest 顺序，`shuffle_unlabeled_pool: false`） |
| 测试划分 | 正常 1,000 张 + 异常 1,000 张 |
| 硬件 | 单卡 RTX 4060（8GB），CUDA |
| 优化器 | Adam，beta=(0.5, 0.999)（论文复现设置） |
| 评测指标 | 重建误差、A/B 差异（inter）、B 模型内差异（intra）的 AUROC / AP |
| 评分口径 | `scoring.pool`：先对图像与重建做 N×N 平均池化再算像素差（N=1 为原始逐像素口径）。最终采用 N=8 |
| 集成 | 每个模块 K 个独立 AE；inter 用两模块均值图的差，intra 用 B 模块各成员的逐像素标准差 |

一般命令（训练/评估）：

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -u main.py --config configs\<实验>.yaml train --module a
.\.venv\Scripts\python.exe -u main.py --config configs\<实验>.yaml train --module b
.\.venv\Scripts\python.exe -u main.py --config configs\<实验>.yaml evaluate
```

---

## 1. 起点：复现与早期模型

| 实验 | 配置 | 结构要点 | 结果（重建 / inter / intra） | 结论 |
|---|---|---|---|---|
| rsna-smoke | 测试用 | 微型合成数据 | 自动测试 | `pytest` 冒烟测试 |
| rsna-train | configs/rsna-train.yaml | compact AE，latent 16，宽度 0.5 | 0.606 / 0.806 / 0.721 | 早期小模型 |
| rsna-large | configs/rsna-large.yaml | compact AE，latent 32，宽度 1.0（238万参数） | 0.537 / 0.650 / 0.613 | 加大容量反而变差（重建更强→差异被稀释） |
| **rsna-paper** | configs/rsna-paper.yaml | 论文原版 AE：64px、latent 16、449万参数、3 成员、250 轮 | **0.670 / 0.832 / 0.706** | 对上论文（0.669 / 0.815 / 0.694），作为基线 |
| rsna-skip128 | configs/rsna-skip128.yaml | 128px + 3 层跳跃连接，latent 128（3561万参数），100 轮 | 0.431 / 0.372 / 0.368 | **失败**：模型近似恒等复制，检测信号反转 |
| rsna-noskip128 | configs/rsna-noskip128.yaml | 由 skip128 权重去掉跳跃后热启动，100 轮 | 0.572 / 0.627 / 0.517 | 方向恢复但仍弱 |
| rsna-noskip64 | configs/rsna-noskip64.yaml | 同上去掉跳跃、输入改回 64（1023万参数） | 0.502 / 0.574 / 0.533 | 仍弱于论文版（瓶颈不够紧） |
| rsna-resample-pilot | configs/rsna-resample-pilot.yaml | 反卷积换“上采样+3×3卷积”，B 单成员 30 轮试点 | 未定论 | 30 轮内未显现消除网格纹理的收益，暂停该方向 |

失败机理（skip128）：特征级评分与记忆距离评分全部 < 0.5，证明跳跃连接让解码器直接复制输入，编码器未学到判别性特征。

---

## 2. 记忆库系列（核心工作）

思路：在隐向量后加 MemAE 式记忆库（注意力加权原型组合 + 熵正则），用更紧的“正常原型约束”压制底噪。
所有实验均从上一档权重热启动（仅记忆库重新初始化），64px、100 轮、3 成员（每个模块），`entropy_loss_weight=0.0002`。

热启动命令示例：

```powershell
.\.venv\Scripts\python.exe scripts\warmstart_memory.py `
  --config configs\rsna-noskip64-mem16.yaml `
  --source outputs\rsna-noskip64-mem25\checkpoints `
  --destination outputs\rsna-noskip64-mem16\checkpoints
```

| 实验 | 记忆槽 | 热启动来源 | 评分池化 | 重建 | A/B差异 | B内差异 | 完成时间 |
|---|---|---|---|---|---|---|---|
| rsna-noskip64-mem | 64 | noskip64 | 1 | 0.581 | 0.647 | 0.580 | 9/29 20:06 |
| 〃（仅评测） | 64 | — | 4 | 0.670 | 0.690 | 0.615 | — |
| rsna-noskip64-mem25 | 25 | mem64 | 1 | 0.641 | 0.751 | 0.661 | 9/29 20:17 |
| 〃（仅评测） | 25 | — | 4 | 0.711 | 0.777 | 0.685 | — |
| rsna-noskip64-mem16 | 16 | mem25 | 4 | 0.737 | 0.810 | 0.695 | 9/29 20:28 |
| rsna-noskip64-mem8（K=3） | 8 | mem16 | 4 | 0.739 | 0.813 | 0.730 | 9/29 20:42 |
| 〃（仅评测） | 8 | — | 8 | 0.767 | 0.814 | 0.732 | — |
| rsna-noskip64-mem4 | 4 | mem8 | 4 | 0.733 | 0.806 | 0.746 | 9/29 20:54 |
| rsna-noskip64-mem8-shrink | 8 + shrink 0.0025 | mem8 | 4 | 0.737 | 0.807 | 0.730 | 9/29 21:06 |
| **rsna-noskip64-mem8（K=5，最终）** | 8 | +新增成员3/4（从零） | 8 | **0.770** | **0.826** | **0.753** | 9/29 21:20 |

结论：

- **记忆库越紧，正常/异常的分离越明显**（见下表），inter 从 0.647 → 0.826；
- 但过紧（4 槽）会让 inter 回落、异常标记率下降，**8 槽是甜点**；
- shrink 硬稀疏（0.0025）没有收益（0.807 < 0.813），维持 0；
- 论文版 AE 也出现过同款“重建越强→差异越弱”的规律，记忆库的作用正是把“重建能力”锁回正常原型上。

正常/异常均值分离度（原始逐像素口径，`scripts/class_stats.py`）：

| 配置 | 正常 inter 均值 | 异常 inter 均值 | 分离度 |
|---|---|---|---|
| 论文版 | 0.0460 | 0.0666 | +45% |
| mem64 | 0.0388 | 0.0437 | +13% |
| mem25 | 0.0434 | 0.0553 | +27% |
| mem16 | 0.0491 | 0.0676 | +38% |
| mem8 | 0.0550 | 0.0827 | +50% |
| mem4 | 0.0682 | 0.1067 | +56% |
| mem8-shrink | 0.0601 | 0.0886 | +47% |

---

## 3. 集成规模扫描（mem8，pool=8）

`scripts/ensemble_probe.py`，同一测试集：

| K | 正常 inter 均值 | 异常 inter 均值 | inter AUROC |
|---|---|---|---|
| 1 | 0.0605 | 0.0915 | 0.795 |
| 2 | 0.0484 | 0.0784 | 0.819 |
| 3 | 0.0428 | 0.0692 | 0.814 |
| 4 | 0.0384 | 0.0622 | 0.819 |
| **5** | **0.0349** | **0.0585** | **0.826** |

结论：增加集成成员可以持续压低正常样本的随机差异（K=1→5 降 42%），同时 AUROC 上升。最终将两个模块各扩到 5 个成员（新增成员从零训练 100 轮）。

---

## 4. 纯评测实验（不训练，用来否定/确认方案）

| 实验 | 脚本 | 结果 | 结论 |
|---|---|---|---|
| 池化尺度扫描 | scripts/pooled_scores.py | 4×/8×/16× → 0.813 / **0.814** / 0.802（inter）；重建 0.739/0.767/0.785 | 采用 8× |
| 局部化评分（中央裁剪 15%、取最热 5%） | scripts/localized_scores.py | inter 0.751→0.715/0.718 | 信号分布在肺野而非单点，放弃 |
| 环带屏蔽（5–15%） | scripts/mask_experiment.py | AUROC 基本不变或微降 | 误标并非只来自最外圈，放弃 |
| 逐像素校准（500 张训练正常的均值/方差） | scripts/calibrated_scores.py | 0.8129 → 0.8142 | 无实质提升，放弃 |
| 特征级 / 记忆距离评分（skip128 诊断） | scripts/feature_scores.py、memory_scores.py | 全部 <0.5 | 证明跳跃版编码器失效 |

定性验证工具（每个实验都可重复生成）：

- `scripts/render_median_samples.py`：中位数正常/异常样本的 6 联图；
- `scripts/render_grid.py`：固定随机种子的 6 正常 + 6 异常网格（共享色标）；
- `scripts/render_flagged_normals.py`：误标最严重的正常样本；
- `scripts/localization_stats.py`：热点面积口径的正常干净率/异常标记率；
- `scripts/grid_analysis.py`：网格纹理/频谱检查。

最终配置（K=5、pool=8）的热点统计（阈值=正常像素 99 分位）：

| 口径（热点面积） | 正常被标 | 异常被标 |
|---|---|---|
| >0.1% | 29.2% | 74.9% |
| >1% | **18.5%** | **63.3%** |
| >2% | 14.7% | 55.7% |
| 论文版（>1%） | 24.1% | 67.7% |

---

## 5. 最终推荐配置

`configs/rsna-noskip64-mem8.yaml`（当前最佳）：

- 64×64 输入、无跳跃 AE、latent 128、宽度 1.0（1024 万参数）；
- 记忆库 8 槽（shrink=0、entropy 权重 2e-4）；
- 每模块 5 个集成成员（A：3 个热启动链 + 2 个从零；B 同理）；
- 评分：8× 平均池化；Adam，lr 2e-4，100 轮/成员。

最终指标（`outputs/rsna-noskip64-mem8/metrics.json`）：

| 指标 | 数值 |
|---|---|
| 重建 AUROC | 0.7699 |
| A/B 差异 AUROC | 0.8255 |
| B 内差异 AUROC | 0.7526 |

定性结果：

- 中位数正常样本（00973）：峰值 0.122，低于热点阈值 0.141，整图无热点；
- 中位数异常样本（01119）：峰值 0.167，出现 2–3 处独立红色热点；
- 正常/异常平均分离 +77%（0.0315 vs 0.0559）；
- 全类别统计（1000 张/类的逐图峰值中位数）：正常 0.116 vs 异常 0.192。

关键产物：

- 可视化总目录：`outputs/rsna-noskip64-mem8/visualizations/`（每个实验批次 8 张 6 联图）
- 中位数样本：`outputs/rsna-noskip64-mem8/median-samples/`
- 随机网格：`outputs/rsna-noskip64-mem8/grids/`
- 检查点：`outputs/rsna-noskip64-mem8/checkpoints/{a,b}/member-00..04.pt`

---

## 6. 一轮实验的完整流程（复刻用）

```powershell
# 1) 热启动（只初始化新记忆库，其余权重继承）
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe scripts\warmstart_memory.py `
  --config configs\rsna-noskip64-mem8.yaml `
  --source outputs\rsna-noskip64-mem16\checkpoints `
  --destination outputs\rsna-noskip64-mem8\checkpoints

# 2) 训练两个模块（resume=true 时已完成的成员自动跳过）
.\.venv\Scripts\python.exe -u main.py --config configs\rsna-noskip64-mem8.yaml train --module a
.\.venv\Scripts\python.exe -u main.py --config configs\rsna-noskip64-mem8.yaml train --module b

# 3) 评估 + 定性验证
.\.venv\Scripts\python.exe -u main.py --config configs\rsna-noskip64-mem8.yaml evaluate
.\.venv\Scripts\python.exe scripts\localization_stats.py configs\rsna-noskip64-mem8.yaml --area 0.01
.\.venv\Scripts\python.exe scripts\render_grid.py configs\rsna-noskip64-mem8.yaml --count 6
.\.venv\Scripts\python.exe scripts\render_median_samples.py configs\rsna-noskip64-mem8.yaml
```

---

## 7. 关键代码变更（相对初始重构版）

| 文件 | 变更 |
|---|---|
| `src/ddad/model.py` | 新增 `PaperAutoencoder`（论文原版 AE）、`SkipAutoencoder`（128px/可开关跳跃/可选 resample 解码器）、`MemoryBank`（MemAE 注意力记忆库 + 熵） |
| `src/ddad/scoring.py` | 新增统一评分函数 `anomaly_maps`（支持 N×N 池化与显示上采样） |
| `src/ddad/engine.py` | 训练支持熵正则、优化器/热启动续训；评估接入 `scoring.pool`、三张热图 + 双重建面板 |
| `src/ddad/config.py` | 新增 `model.memory_size/shrink_threshold/architecture`、`train.entropy_loss_weight/checkpoint_interval`、`scoring.pool` |
| `src/ddad/visualize.py` | 面板重构（Input/Recon A/Recon B/误差/差异/B 方差），平滑上采样，色标从测试集分位数估计 |
| `scripts/` | 热启动转换与上述全部评测/渲染脚本（见第 4 节） |

---

## 8. 已知问题与后续方向

1. 仍有约 18–29% 的正常样本被标出小热点（口径不同），集中在构图异常（大黑边、高亮条带）的非典型图像；
2. 极端 OOD 样本（小胸片悬在大黑框内）必然误标——属数据质量问题；
3. 进一步降噪可继续扩集成（K=8 预计正常差异再降约 15%），或对新成员加长训练轮数；
4. 128px 跳跃连接路线已确认不可用（检测信号反转），相关检查点保留仅作对比；
5. resample 解码器的抗网格伪影结论未定（试点仅 30 轮、单成员），如需要可完整重训验证。

---

## 9. 改进计划执行结果（2026-09-30）

### 9.1 封存集验证

从无标签池中已经留出的样本里，用 seed=2026 固定抽取 1,000 张正常和 1,000 张异常。它们与训练集和原始测试集均无文件重叠。结果：

| 指标 | AUROC | AP |
|---|---:|---:|
| 重建误差 | 0.7783 | 0.7507 |
| A/B 差异 | 0.8440 | 0.8242 |
| B 内差异 | 0.7547 | 0.7143 |

产物：`outputs/rsna-noskip64-mem8/heldout/{manifest.json,scores.csv,metrics.json}`。

按图像分层 bootstrap 2,000 次得到 inter AUROC 的 95% CI 为 0.8268–0.8617，AP 的 95% CI 为 0.7997–0.8502，保存于 `outputs/rsna-noskip64-mem8/heldout/bootstrap_ci.json`。

### 9.2 ASR 热图修正试验

冻结现有 5+5 个重建模型，用正常训练图像随机混合局部块产生伪异常，训练三层卷积修正网络。初版 ASR 容易输出接近常数的热图，因此修正为在原始 inter 图的 logit 上预测残差，以保留空间证据。

第一封存集用于完成这次修正；随后从剩余、与训练集、原测试集及第一封存集均无重叠的样本中再抽取正常/异常各 1,000 张，组成 `heldout-final`，只做一次最终评估：

| 评分（heldout-final） | AUROC | AP |
|---|---:|---:|
| 原始 inter（pool=1） | 0.8348 | 0.8237 |
| ASR 残差修正 | **0.8574** | **0.8681** |

ASR AUROC 的 bootstrap 95% CI 为 0.8403–0.8728，AP 的 95% CI 为 0.8506–0.8847。pool=8 的原始 inter 在同一最终集上为 0.8407 / 0.8333。ASR 现作为最终候选后处理模块保存于 `outputs/rsna-noskip64-mem8/asr/`。

### 9.3 误报分析

在原测试集上，正常图热点阈值为 0.14149；热点面积超过 1% 的比例为正常 18.5%、异常 63.3%。排名最高的正常误报包含胸片悬在大黑框中的构图，说明主要误报来自视野和构图异常，而不是肺野中的局部病灶。修正后的脚本为 `scripts/render_flagged_normals.py`。

当前 Med-AD 压缩包只包含 PNG 和 `data.json`，没有 RSNA 原始病灶框 CSV，因此本轮不能声称完成病灶框命中率验证；后续拿到标注文件后可用同一批保存的逐图分数复核。

### 9.4 128×128 高分辨率试点

使用无跳跃连接、latent=128、8 槽记忆库，A/B 各训练 1 个成员 20 轮。封存集结果：

| 指标 | 64×64 mem8（5+5） | 128×128 试点（1+1） |
|---|---:|---:|
| 重建 AUROC | 0.7783 | 0.7188 |
| inter AUROC | **0.8440** | 0.5696 |
| intra AUROC | 0.7547 | 0.5000 |

当前高分辨率路线没有收益，试点检查点保留在 `outputs/rsna-noskip128-mem8-pilot-single/`，不进入最终配置。

### 9.5 当前结论

最终重建配置仍为 `configs/rsna-noskip64-mem8.yaml`。图像级主结果可以采用残差 ASR 分数，同时保留原始 pool=8 inter 作为无需后处理的基线。真实病灶框标注仍是判断热图能否用于定位的必要条件。
