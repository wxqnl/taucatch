# 复演、完整运行与训练

## 1. 先复演保存结果（无需 GPU）

```bash
python tools/handoff.py verify
python tools/handoff.py results
CUDA_VISIBLE_DEVICES= python scripts/replay_fusion.py
```

前两项只需 Python 3.10+ 标准库。融合头复演需 CPU PyTorch：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/replay.txt
```

复演读取精选的 TRAIN 特征、固定基础 logits 和当前融合层权重，执行 `tanh(base_logits + fusion(features,context))`，核对保存的 12×3 动作输出。它验证融合层计算和动作接口；**没有运行完整 SpikeVLA、事件感知或物理接球，不能产生新的接球率或推理时延。**

本次整理另保留并运行了 13 项现有动作协议/时间网格检查：仿真环境 `unittest` 的 10 项，以及模型环境 `pytest` 的 3 项，均通过。验证记录见 `artifacts/validation.json`。安装 `requirements/visualization.txt` 后，`python tools/plot_results.py` 可从完整 6 组原始数据重绘成绩图，`python tools/render_saved_trajectory.py` 可重绘轨迹 GIF。主动修改代码或图后，用 `python tools/handoff.py refresh` 更新交接校验清单，再 `verify` 并重新构建源码包；原始实验报告不应随之改写。

## 2. 完整闭环需要哪些内容

| 部分 | 当前状态与获取位置 |
|---|---|
| 本项目核心源码 | 默认交接包包含，完整文件清单在 `artifacts/source_manifest.json` |
| 当前融合适配权重 | 默认包包含原始 `fusion-617` adapter；约 21 MB，不是完整基座 |
| 本项目基础权重与适配器祖先 | 单独的 `spikevla-catch-checkpoints.tar.gz`，展开后保持 `outputs/...` 原路径；含 rich 基座、finder 及当前加载器所校验的父权重 |
| SpikeVLA 架构代码 | 需要配套 `Spike-driven-VLA` 仓库版本，参见 `artifacts/external_dependencies.json` |
| CTS 训练/推理框架 | 需要配套 `cts-v1-pretrain` 仓库，提供 `experiments.pretrain.common` 与 `turbovla` |
| 预训练初始化和文本投影 | 加载器初始化仍需要原预训练 checkpoint 与文本投影，未随本项目包分发；本机路径及 SHA 在外部依赖清单 |
| 视觉/语言模型资产 | 对应预训练框架的本地资产目录与 recipe；本包不包含这些外部大模型资产 |
| MuJoCo 模型资产 | 默认包附当前仿真使用的 FR3 与 Shadow Hand 资产及其原许可证 |

历史完整运行分为两个 Python 环境：仿真环境含 MuJoCo、Ruckig、SciPy、Numba、PyTorch；模型环境含 PyTorch、Transformers 与上游 SpikeVLA/CTS 依赖。`requirements/` 和 `artifacts/environment.json` 记录本机实际版本，前者是最小依赖范围，后者用于对齐已有运行条件，不保证所有平台二进制兼容。

### 完整运行的准备与入口

1. 展开源码包与权重包到同一根目录；按外部依赖清单补齐两个上游仓库、预训练 checkpoint 和资产。
2. 原始构建器 `taucatch/chooser.py` 中的 `CTS_ROOT / SPIKEVLA_ROOT / ASSET_ROOT / PRETRAINED / TEXT_PROJECTION` 是本机路径。**迁移机器时需显式修改这五项及上游 recipe 内的资产路径**，然后重新做数值一致性验收；它们不是可直接照抄的通用路径。
3. 从项目根目录设置 `PYTHONPATH=.:scripts`。`scripts/serve_xyz_fusion_v603.py --configuration <新的服务配置>` 在模型环境运行，原入口限定 GPU5；`scripts/collect_xyz_fusion_v605.py --configuration <新的采集配置>` 在仿真环境运行，finder 原入口限定 GPU0。模型服务使用本地认证 socket，部署时保持仅本机可访问。
4. 配置模板结构可参考 `artifacts/configuration_examples/`。它们是**历史结构样例**，内含本机路径和历史源 hash；不能直接当作一键新实验配置。完整配置引用的数据、数值 gate、运行选项须先对照外部依赖准备并重新验收。
5. 完整运行之前，用 `check_xyz_fusion_RPC_v608.py` 检查实际时延，再用 `check_xyz_fusion_actual_v604.py` 独立重算请求。不能跳过时延验收后宣称达到历史接球率。

上述完整闭环源码用于算法复现与接续开发，**尚未改造成跨机器一键启动器**。本次实际验证范围是独立解包的证据校验与 CPU 融合层复演。巡检已经关闭，源码中的旧训练入口会受 STOP 保护；不要为迁移方便盲删 STOP。

## 3. 应当如何训练

| 阶段 | 数据与优化 | 验收依据 |
|---|---|---|
| 对齐机械约束与快环 | 标定机器人/手、延迟、相机和球模型；先固定球型建立快环基线 | 感知、到达、闭合、保持分别可测；同条件可重复 |
| 原生动作预训练/SFT | 多几何、多速度、多初始手位与噪声；保留原生 12×16 ms 时间网格，适配 DVS 编码器与动作解码器 | 离线动作误差下降，并验证动作在闭环中的可执行性 |
| 残差动作训练 | 当前实现见 `fit_xyz_fusion_hardnegative_v615.py`；失败几何＋相邻成功几何均衡采样，成功状态保持项防止过度修正 | 必须按几何划分，训练数据不能混入冻结留出几何 |
| 修正执行标签 | 使用真实送达时间、接触前实际插值支持区间；失败方向由可行教师给出，保留当前策略自身成功状态 | 缺失端点必须有完整教师/成功动作依据，不能补 mask 而不补标签 |
| 独立验证 | 固定权重，交替运行 VLA 开/关；保留全部失败和时延超限 | 留出集接稳率净提升、过期率不升、收益在速度/几何上稳定 |
| 可选 RL | 从已验证的 SFT 模型继续，用新的 on-policy 数据和真实交付时延，加入保持奖励与动作代价 | 只能按新策略重新采样；收益必须在冻结留出集成立 |

最近一次标签修正后的 CPU 融合层更新未通过保持约束，已经恢复父模型。因此优先事项是数据覆盖、监督目标与稳定延迟，并非简单延长训练或直接扩大 RL。真机训练需先替换当前理想化 `agile` 动力学及轻量手模型，详见[硬件验收](hardware.md)。
