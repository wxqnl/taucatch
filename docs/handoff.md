# 交接清单与整理范围

对外发送 `dist/spikevla-catch-source.tar.gz` 即可阅读算法、查看证据并做 CPU 复演；需要继续完整模型开发时，加发 `dist/spikevla-catch-checkpoints.tar.gz`，以及另行提供配套的两个上游仓库和模型资产。

| 内容 | 源码交接包 | 本机保留位置 |
|---|---|---|
| README、算法、结果、代码导航、复现/训练、硬件表 | 包含 | 根目录与 `docs/` |
| 依赖闭包内的核心源码与相关实验入口 | 包含 | `taucatch/`、`scripts/` |
| 原始配对完成报告、时延审计、逐次 CSV | 包含 | `artifacts/evidence/` |
| 当前 adapter、CPU 融合层复演样本 | 包含 | `artifacts/models/`、`artifacts/examples/` |
| 本项目完整运行所需本地基础/父权重 | 单独权重包 | `outputs/`，保留原 hash |
| 上游模型、资产与仓库 | 不包含；列明版本与路径/hash | 外部依赖清单 |
| 旧语义选择分支、一次性探针、巡检操作文档 | 不包含 | `archive/20261028/` |
| 原始全量训练数据、历史运行 tape、完整实验日志 | 不包含 | 本机 `data/`、`outputs/`、归档目录 |
| 账号配置、缓存、临时进程状态 | 不包含 | 不进入白名单打包 |

`artifacts/MANIFEST.sha256` 校验精选资料，`artifacts/source_manifest.json` 记录源码，`artifacts/checkpoints.json` 列出权重身份。旧绝对路径、报告中的 `outputs/...` 字符串仅是来源标识；精选资料中的相应副本可离线检查。原始字节保留，未为美化文档改写实验报告。

解包两个文件时使用同一个目录，它们都会生成 `spikevla-catch/` 根目录。源码包可单独复演，完整模型依赖仍按[复现说明](reproduction.md)补齐。重新构建命令为 `python tools/handoff.py build` 与 `python tools/handoff.py build-weights`；工具会拒绝覆盖同名已有包。

本机整理采用归档与内容相同文件去重；清单在 `archive/20261028/cleanup_manifest.json`。同内容模型快照可能共享 inode，历史权重不得原地覆盖，后续训练应另存新文件。恢复原路径时运行 `python tools/restore_archive.py --dry-run` 查看计划，再按需执行 `--apply`。恢复旧脚本不等于授权恢复实验或巡检。

模型与本项目研究代码用于协作交接，未额外声明公共开源许可；第三方资产保留各自许可证。若准备公开发布，应由项目所有者另行确认许可与上游分发条件。
