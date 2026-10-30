# 核心代码导航

版本号保留是为了追溯现有权重与实验。推荐按下表顺序阅读。

| 环节 | 源码 | 作用 |
|---|---|---|
| 仿真与机器人 | [scene.py](../taucatch/scene.py)、[volley.py](../taucatch/volley.py) | FR3 资产与 Shadow Hand 资产组合、球与接稳判据；`agile` 是放大约束的理想化配置 |
| DVS 输入 | [fused_observation_v232.py](../taucatch/fused_observation_v232.py)、[fused_event_tracker_v231.py](../taucatch/fused_event_tracker_v231.py) | 学习定位、事件特征、因果交付与三维跟踪 |
| 快环规划 | [multiscale_plane_controller_v13.py](../taucatch/multiscale_plane_controller_v13.py) | 拦截位置/姿态与连续目标更新 |
| 快环闭合与运控 | [native_close_timing_v44.py](../taucatch/native_close_timing_v44.py)、[differential_servo_v72.py](../taucatch/differential_servo_v72.py) | 闭合时机、局部 IK、关节执行 |
| SpikeVLA＋DVS | [action_policy.py](../taucatch/action_policy.py) | 事件 token 与原生 SpikeVLA 动作头结合 |
| 动作融合头 | [xyz_fusion_policy_v596.py](../taucatch/xyz_fusion_policy_v596.py) | 256D 原生特征＋22D 上下文生成 XYZ chunk |
| 上下文 | [xyz_context_softsign_v273.py](../taucatch/xyz_context_softsign_v273.py) | 已交付快环动作、本体、误差与观测年龄 |
| 真实推理运行时 | [xyz_fusion_runtime_v601.py](../taucatch/xyz_fusion_runtime_v601.py)、[xyz_prethrow_runtime_v199.py](../taucatch/xyz_prethrow_runtime_v199.py) | 加载模型、释放前条件缓存、CUDA 图推理 |
| 执行限幅与过期 | [xyz_coarse_execution_v309.py](../taucatch/xyz_coarse_execution_v309.py)、[native_residual_v26.py](../taucatch/native_residual_v26.py) | 6 cm / 0.8 m/s、观测时间插值与过期处理 |
| 完整闭环采集 | [collect_xyz_fusion_v605.py](../scripts/collect_xyz_fusion_v605.py)、[serve_xyz_fusion_v603.py](../scripts/serve_xyz_fusion_v603.py) | 仿真进程与 VLA 服务进程连接，记录真实交付时间 |
| 当前权重的训练算法 | [fit_xyz_fusion_hardnegative_v615.py](../scripts/fit_xyz_fusion_hardnegative_v615.py) | 失败几何与相邻成功几何均衡采样，联合适配 DVS、原生动作头、融合头 |
| 训练支持区间 | [xyz_observed_support_v700.py](../taucatch/xyz_observed_support_v700.py) | 按实际执行插值，补齐有监督依据的端点 |
| 独立数值与执行审计 | [check_xyz_fusion_actual_v604.py](../scripts/check_xyz_fusion_actual_v604.py)、[audit_xyz_fusion_v606.py](../scripts/audit_xyz_fusion_v606.py) | 重算保存请求、检查输出及真实执行 |
| 轻量交接复演 | [replay_fusion.py](../scripts/replay_fusion.py)、[handoff.py](../tools/handoff.py) | 无 GPU 的融合头复演、证据摘要与交接校验 |

`chooser.py`/`choice.py` 属于更早的语义编号选择分支；当前动作路径仍复用其中部分初始化工具。保留在依赖闭包中的文件不一定是当前顶层算法入口。历史运维编排与一次性探针已归档，默认交接包不含它们。
