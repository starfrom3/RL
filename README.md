
# AI-Driven Design of Ligands for Am(III)/Eu(III) Based on Coordinative Stoichiometry Differences

A unified AI framework that combines Graph Neural Networks (GNNs), Variational Autoencoders (VAEs), and Reinforcement Learning (RL) to de novo design and optimize extractant molecules for Am(III)/Eu(III) separation using a physically meaningful descriptor Δn (coordination stoichiometry difference).

AI‑driven design of ligands targets improving Am(III)/Eu(III) selectivity by exploring chemical space efficiently, guided by a Δn-based objective, while ensuring chemical validity and structural diversity.

---

## 目录 / Table of Contents
- [项目概要 / Project Overview](#project-overview)
- [快速开始 / Quick Start](#quick-start)
- [安装与依赖 / Installation & Dependencies](#installation--dependencies)
- [数据集与任务 / Data & Tasks](#data--tasks)
- [方法与实现 / Methods & Implementation](#methods--implementation)
- [运行实验 / Running Experiments](#running-experiments)
- [结果与分析 / Results & Analysis](#results--analysis)
- [可重复性与评估 / Reproducibility & Evaluation](#reproducibility--evaluation)
- [API 与代码结构 / API & Code Structure](#api--code-structure)
- [贡献指南 / Contributing](#contributing)
- [许可证 / License](#license)
- [联系方式 / Contact](#contact)

---

## 项目概要 / Project Overview

- 研究背景：锕系/镧系元素的分离在核燃料再处理和高放射性废液管理中至关重要，Am(III)与 Eu(III) 的差异微弱，传统设计效率低下。本文提出将 GNN、VAE、RL 融合，利用 Δn（Am(III) 与 Eu(III) 在特定溶剂环境下偏好的配位数差）作为目标描述符进行 de novo 萃取剂分子设计与优化。  
- 核心创新：
  - 将 Δn 作为可解释且与分离过程直接相关的目标属性；
  - 多模态 GNN 预测不同金属–溶剂环境下的配位行为；
  - VAE 在连续潜在化学空间中生成化学合理且多样的分子；
  - 基于 RL 的目标导向搜索，在潜在空间中提升 Δn 值对应的分离潜力；
  - 迁移学习与嵌套交叉验证在极少样本条件下的鲁棒性分析。
- 产出概览：生成的候选分子数量级（如 9,464）及其核心骨架分布、以及多组 Murcko 骨架与 Δn 的关联分析。

---

## 快速开始 / Quick Start

1) 克隆仓库
- git clone git@github.com:starfrom3/RL.git
- cd ai-ligand-design

2) 设置环境
- 推荐使用 Conda：
  - conda create -n ligand-ai python=3.9
  - conda activate ligand-ai
  - conda install -c rdkit rdkit

3) 数据准备
- 将数据放置在 data/ 目录，包含：
  - ligand SMILES 列表
  - Δn 标签（或用于训练的代理任务标签）
  - 溶剂系统信息（最多 4 种溶剂的特征）
- 数据需要符合 config.yaml/ configs/*.yaml 所指定的字段结构

4) 启动训练/推理
- 示例（请根据实际配置文件调整路径和参数）：
  - python train.py --config configs/am_eu_config.yaml
  - python generate.py --config configs/generation_config.yaml

5) 评估与可重复性
- 评估指标包括基于 GNN 的 Δn 预测准确性、生成分子的化学有效性、Latent Space 的多样性（PCA/T-SNE 等），以及 Murcko 骨架与 Δn 的统计关联。

---

## 安装与依赖 / Installation & Dependencies

- 语言/环境：Python 3.9+  
- 关键依赖与组件：
  - PyTorch（或其他你们团队使用的深度学习框架）
  - RDKit（化学信息学工具，用于 SMILES ↔ 图结构转换、指纹计算）
  - OpenBabel（可选，用于分子描述符扩展）
  - scikit-learn、numpy、pandas、tqdm、matplotlib（可选用于可视化）
- 数据版本控制：若使用 Git LFS 或大文件存储，请确保配额与设置合规
- 运行环境示例（conda 方式）：
  - conda create -n ligand-ai python=3.9
  - conda activate ligand-ai
  - conda install -c conda-forge rdkit

---

## 数据集与任务 / Data & Tasks

- 数据来源：文献中的 Am(III)/Eu(III) 萃取系统数据，以及 Supporting Information（SI）中的配体结构、SMILES、溶剂组分等。论文引入 Δn 作为核心目标描述符，结合多模态特征用于模型训练。  
- 数据结构要点：
  - 分子图信息（SMILES → 图结构）
  - 金属离子特征（原子质量、离子化能、离子半径、电负性等）
  - 溶剂系统特征（最多 4 种溶剂及其描述符）
  - Δn 标签（目标属性）及与之相关的分离潜力评分
- 数据划分：训练/验证/测试集，80%/10%/10% 的比例，且避免同一配体在不同数据子集间数据泄漏。

---

## 方法与实现 / Methods & Implementation

- 总体框架：GNN（配位预测） + VAE（分子生成） + RL（目标导向优化）
- Δn 描述符：表示 Am(III) 与 Eu(III) 的配位数差，作为奖励信号的核心组分。
- GNN 模块：多模态编码与深融合，将配体、金属离子和溶剂环境的特征进行联合建模。
- VAE 模块：基于 SMILES 的序列编码/解码，进入连续潜在空间以便高效搜索。
- RL 模块：基于 Actor-Critic 的策略学习，在潜在空间中进行目标导向优化，结合相似性约束和化学有效性惩罚。
- 迁移学习与验证：先在大规模结构数据上进行预训练（如 CSD），再在目标任务数据上微调；嵌套交叉验证以提高小样本条件下的鲁棒性。

---

## 运行实验 / Running Experiments

- 配置文件示例：configs/config.yaml
- 关键参数（示例，可根据实际实现调整）：
  - data_path: data/
  - model:
    - gnn_layers: 4
    - vae_latent_dim: 256
    - rl_algorithm: "ActorCritic"
  - training:
    - epochs: 100
    - batch_size: 8
    - seed: 42
  - evaluation:
    - metrics: ["Delta_n_prediction_accuracy", "diversity", "scaffold_analysis"]

- 一键执行模板（bash 脚本示例，需按实际脚本名称调整）
  - bash run_all.sh configs/am_eu_config.yaml

---

## 结果与分析 / Results & Analysis

- Δn 的物理化学意义：通过配位数差来反映分离潜力，理论上影响分配比与选择性。
- 模型性能：GNN 对 Δn 的预测鲁棒性、VAE 生成分子的化学有效性、RL 优化后分子在 Latent Space 的分布与多样性。
- 化学空间分析：Murcko 核心骨架的分布、Δn 的分布以及在 t-SNE/PCA 映射中的分布特征。
- 何时需要进一步工作：小样本下的鲁棒性提升、多目标优化、实验反馈回路的整合等。

---

## 可重复性与评估 / Reproducibility & Evaluation

- 复现实验要点：
  - 固定随机种子，使用嵌套交叉验证（如 10 次重复的 5 折交叉验证）
  - 严格的数据分组，防止同一配体的不同溶剂组导致信息泄漏
  - 对生成分子进行化学有效性校验（RDKit 等）
- LFS 与大文件传输注意事项：若涉及大分子数据，确保 LFS 配额和网络条件足以支撑训练与推断过程。

---

## API 与代码结构 / API & Code Structure

- 主要模块（示意，实际项目中可能不同）：
  - data/：数据处理与数据加载
  - models/：GNN 与 VAE 的实现
  - rl/：强化学习策略与训练循环
  - train.py、generate.py、evaluate.py：训练、生成、评估入口
  - configs/：配置文件目录
  - scripts/：辅助脚本，如数据准备、可视化
- 主要可执行接口（示例）：
  - python train.py --config configs/am_eu_config.yaml
  - python generate.py --config configs/generation_config.yaml
  - python evaluate.py --config configs/eval_config.yaml

---

## 作者与联系 / Authors & Contact

- 作者：阙宇龙、杨栋嵊、张智渊、刘冲等（示例名称）
- 联系方式：2023223070042@stu.scu.edu.cn

---

## 参考与引用 / References

- 文章核心思想：Que et al. AI‑Driven Design of Ligands for Am(III)/Eu(III) Based on Coordinative Stoichiometry Differences
- 相关领域方法与背景文献：GNN、VAE、RL、Murcko scaffolds、t-SNE、PCA、LFS 等

---

## 参考演示与文档入口

- 详细技术细节、数据表与图谱，请参阅论文的 Supporting Information（SI）与 Figure 1–5 的图示。

---

## 联系与反馈 / Contact & Feedback

- 如需帮助定制特定的 README 版本（英文/双语、针对你的代码结构的具体模板），请告诉我你的仓库结构、语言偏好和目标受众。

