# COM9501_Group20_Project

**COMP9501 · Group 20 · 面向人形机器人鲁棒控制的深度强化学习**

Chen Yuxuan · Deng Zihan · Han Yue · Jia Feiyu

[English](README.md) · [仿真演示](#simulation-demos) · [环境安装](docs/SETUP.md)

<a id="overview"></a>

## 项目概述

本项目研究基于 **混合内部模型（Hybrid Internal Model，HIM）** 与 **近端策略优化（Proximal Policy Optimization，PPO）** 的人形机器人下半身控制。策略需要在上半身姿态独立变化时，跟踪移动与身体高度指令并维持平衡。

仓库包含 G1 训练任务、Isaac Gym 环境、学习算法和实验工具。实现重点是强化学习与仿真，硬件构建、机器人部署和模仿学习不在本仓库的实现范围内。

<a id="method"></a>

## 三项训练设计

1. **上半身姿态课程：** 根据移动跟踪表现逐步增加姿态扰动难度。
2. **高度与膝关节奖励：** 将目标高度跟踪与膝关节相关奖励结合，支持下蹲与平衡控制。
3. **对称性：** 利用镜像经验和动作策略／价值网络的一致性损失，并提供独立开关用于对照实验。

HIM 编码器从六帧观测中估计显式速度与潜在响应表示，PPO 利用这些特征学习十二维下半身关节目标。详见 [方法说明](docs/METHOD.md) 和 [实现改动](docs/CHANGES.md)。

<a id="simulation-demos"></a>

## 仿真演示

补充视频展示仿真中的行走、高度调整与移动操作。5 个 MP4 文件均保存在 `media/videos/`，解压后可直接在本地播放。

| 演示内容 | 仿真器 | 视频 |
|---|---|---|
| Unitree G1：行走与高度控制 | Isaac Gym | [播放视频](media/videos/g1.mp4) |
| Fourier GR-1：行走与高度控制 | Isaac Gym | [播放视频](media/videos/gr1.mp4) |
| Unitree G1：场景演示 | GRUtopia | [播放视频](media/videos/g1sim.mp4) |
| Fourier GR-1：场景演示 | GRUtopia | [播放视频](media/videos/gr1sim.mp4) |
| 移动操作任务演示 | GRUtopia | [播放视频](media/videos/g1tasksim.mp4) |

本仓库提供的可运行任务是 **Isaac Gym 中的 G1**。GR-1 和 GRUtopia 视频补充展示其他机器人与场景，其训练配置和仿真器集成未包含在本仓库中。

<a id="quick-start"></a>

## 快速开始

目标环境：Ubuntu 20.04/22.04、Python 3.8、Isaac Gym Preview 4 和 NVIDIA GPU。先完成 [环境安装](docs/SETUP.md)，再从仓库根目录运行：

```bash
# 不启动 Isaac Gym 或 GPU，先预览训练配置。
python scripts/run.py train --variant full --seed 0 --dry-run

# 先用少量环境检查仿真是否能正常运行。
python scripts/run.py train --variant full --num-envs 32 --iterations 2 --headless

# 训练完整方法。
python scripts/run.py train --variant full --seed 0 --num-envs 4096 --iterations 5000 --headless

# 评估模型，并使用与训练一致的方法配置和随机种子。
python scripts/run.py evaluate --variant full --seed 0 --checkpoint runs/YOUR_RUN/model_4999.pt --num-envs 1000 --horizon 20

# 在仿真中观看策略表现。
python scripts/run.py play --variant full --checkpoint runs/YOUR_RUN/model_4999.pt
```

将 `YOUR_RUN` 替换为实际训练目录。`--output` 可指定新的运行目录，已有目录不会被覆盖。训练保存 TensorBoard 日志、模型和 `run.json`；评估需要与模型匹配的训练记录，并使用确定性的动作均值。

<a id="experiments"></a>

## 消融实验

| 配置 | 姿态课程 | 高度／膝关节权重 | 镜像数据 | 对称损失 |
|---|---|---|---|---|
| `full` | 完整范围支持 | 2 / -0.75 | 启用 | 启用 |
| `no_curriculum` | 初始即为完整难度 | 2 / -0.75 | 启用 | 启用 |
| `range_curriculum` | 范围受限 | 2 / -0.75 | 启用 | 启用 |
| `no_knee` | 完整范围支持 | 2 / 0 | 启用 | 启用 |
| `no_knee_high_height` | 完整范围支持 | 4 / 0 | 启用 | 启用 |
| `augmentation_only` | 完整范围支持 | 2 / -0.75 | 启用 | 关闭 |
| `loss_only` | 完整范围支持 | 2 / -0.75 | 关闭 | 启用 |
| `no_symmetry` | 完整范围支持 | 2 / -0.75 | 关闭 | 关闭 |
| `baseline` | 初始即为完整难度 | 0 / 0 | 关闭 | 关闭 |

`baseline` 保留相同的 G1 任务和 HIM+PPO 结构，并关闭三项训练设计；提高高度奖励的对照组将该权重设为 4。默认训练种子为 `0、1、2`；评估使用 1000 个环境、20 秒时长，所有配置统一采用完整难度的上半身姿态采样。

```bash
# 打印全部 27 个任务：9 组配置，每组 3 个训练种子。
python scripts/sweep.py

# 依次运行选定的实验。
python scripts/sweep.py --variants full no_curriculum no_knee --seeds 0 1 2 --execute

# 汇总实际评估文件并生成图表。
python scripts/summarize.py runs/EVAL_0/metrics.json runs/EVAL_1/metrics.json runs/EVAL_2/metrics.json --output results/summary.csv --plot results/summary.png
```

汇总结果给出不同训练种子之间的均值和样本标准差。指标定义与模型选择方式详见 [评估协议](docs/EXPERIMENTS.md)。

<a id="structure"></a>

## 仓库结构

| 路径 | 内容 |
|---|---|
| `configs/experiments.json` | 实验配置、随机种子与评估设置 |
| `project/` | 配置管理与评估统计 |
| `scripts/` | 训练、评估、播放、批量实验与汇总 |
| `legged_gym/` | G1 Isaac Gym 环境与机器人模型 |
| `rsl_rl/` | HIM、PPO、轨迹存储与对称变换 |
| `tests/` | 算法与评估单元测试 |
| `results/` | 实验状态与实测结果汇总 |
| `media/` | 仿真补充视频索引 |
| `docs/` | 安装、方法与实验说明 |
| `licenses/` | 许可证声明与实现历史记录 |

<a id="teleoperation"></a>

## 遥操作背景

预期的操作端使用现有外骨骼／手套与 Apple Vision Pro 设备。本仿真研究通过采样上半身目标及移动／高度指令来表示操作员输入，仓库不实现设备数据采集、图像传输或真实机器人通信。详见 [接口范围](docs/TELEOPERATION.md)。

<a id="validation"></a>

## 验证与运行状态

**10 项 CPU 单元测试已通过。** 测试覆盖对称变换、PPO 更新、轨迹回报、观测布局、评估统计与模型元数据。

```bash
python -m unittest discover -s tests -v
python scripts/run.py train --dry-run
```

本次整理后的版本尚未重新训练或完成 Isaac Gym 端到端验证，未附带新生成的模型或定量评估文件。上方视频作为补充演示。详见 [验证说明](docs/VALIDATION.md) 和 [运行状态](results/status.json)。

<a id="acknowledgements"></a>

## 致谢与许可证

学习框架使用 [HIMLoco](https://github.com/Sage-PhysicalAI/HIMLoco)、[RSL-RL](https://github.com/leggedrobotics/rsl_rl)、[legged_gym](https://github.com/leggedrobotics/legged_gym)、NVIDIA Isaac Gym 和 Unitree G1 机器人模型。各组件的作者声明与许可证予以保留，详见 [致谢说明](ATTRIBUTION.md) 和 [LICENSE](LICENSE)。
