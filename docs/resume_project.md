# MultiWorld Causal Subsidy — 简历项目描述

> **用途**：研究生申请（数据科学/统计学/商业分析硕士项目）
> **原则**：STAR法则三点展开，不硬性划分技术板块；突出学术潜力与方法论深度
> **注意**：本项目为个人独立完成，不提及竞赛背景

---

## 🇨🇳 中文版

**MultiWorld Causal Subsidy** ｜ 个人研究项目 ｜ Python, CausalML, DoWhy, Mesa ABM, NetworkX
*GitHub: github.com/Crescent-Starling/multiworld-causal-subsidy*

- **构建因果推断与AI仿真融合框架**：设计T/X/DR/S-Learner四重Meta-Learner竞选管道估计异质处理效应（HTE），结合PSM、IPW与双重鲁棒估计消除观测数据混淆偏倚；Bootstrap置信区间与E-value敏感性分析量化因果结论对未测量混淆的稳健性；形成从CATE估计→CATE驱动策略→仿真验证的闭环管道（CausalSimulationPipeline），正uplift过滤确保"负效应用户不补贴"的因果语义一致性
- **提出多平行世界仿真评估机制**：基于Mesa ABM构建500智能体×30轮仿真环境，通过共享Agent配置（AgentConfig）与增量记账实现策略效应与随机噪声的解耦量化——同一组用户画像在相同初始条件下运行5种策略，差异仅源于策略本身；5-seed Monte Carlo验证认知型策略排名一致（CV=0.14），参数扰动（±20%）确认策略排序对关键行为参数不敏感
- **内嵌行为经济学理论的认知型Agent模型**：将前景理论价值函数（Kahneman & Tversky, 1979; α=0.88, λ=2.25）、心理账户参考点更新（Thaler, 1985; 4种账户类型+windfall→income迁移机制）、有限理性折扣（Simon, 1955）与疲劳脱敏统一嵌入决策流程，使Agent行为具备可解释的认知科学基础；认知型策略ROI达2.95，较随机基线提升26.5%；SIR社会传染模型在4种网络拓扑上揭示BA网络正向溢出（+27.78）与WS网络抑制效应（−19.08）的质性差异

---

## 🇬🇧 English Version

**MultiWorld Causal Subsidy** ｜ Individual Research Project ｜ Python, CausalML, DoWhy, Mesa ABM, NetworkX
*GitHub: github.com/Crescent-Starling/multiworld-causal-subsidy*

- **Developed a causal inference–AI simulation framework** integrating T/X/DR/S-Learner meta-learners for heterogeneous treatment effect (HTE) estimation, with PSM, IPW, and doubly-robust estimation to deconfound observational data; Bootstrap CIs and E-value sensitivity analysis quantify robustness to unmeasured confounding; implemented CausalSimulationPipeline (CATE estimation → CATE-driven strategy → simulation validation), where positive-uplift filtering ensures causal semantic consistency (negative-effect users receive no subsidy)
- **Proposed a multi-world evaluation mechanism** on a 500-agent × 30-round Mesa ABM, employing shared AgentConfig and incremental accounting to decouple strategy effects from random noise—the same user profiles run under 5 strategies with identical initial conditions, so observed differences arise solely from policy design; 5-seed Monte Carlo confirmed cognitive strategy's ranking consistency (CV=0.14), and ±20% parameter perturbation verified insensitivity of strategy ordering to key behavioral parameters
- **Embedded behavioral economics in cognitively-grounded agents**: prospect theory value function (Kahneman & Tversky, 1979; α=0.88, λ=2.25), mental accounting reference-point updates (Thaler, 1985; 4 account types + windfall→income migration), bounded rationality discounting (Simon, 1955), and fatigue desensitization—yielding interpretable, cognitively-grounded agent decisions; cognitive strategy achieved ROI=2.95, a 26.5% improvement over random baseline; SIR social contagion across 4 network topologies revealed qualitatively distinct spillover patterns—positive in BA scale-free networks (+27.78) vs. inhibitory in WS small-world networks (−19.08)

---

## 附：备选精简版（1-2 bullet，适合版面受限场景）

### 中文精简版

**MultiWorld Causal Subsidy** ｜ 个人研究项目 ｜ Python, CausalML, Mesa ABM
- 构建因果推断与AI仿真融合框架：T/X/DR/S-Learner四重Meta-Learner + PSM/IPW/双重鲁棒估计消除混淆偏倚，CausalSimulationPipeline实现CATE→策略→验证闭环；多平行世界仿真（500 Agent×30轮）通过共享AgentConfig解耦策略效应与噪声，5-seed Monte Carlo验证认知型策略排名稳健（CV=0.14），ROI达2.95（较随机基线+26.5%）；认知型Agent内嵌前景理论/心理账户/有限理性折扣，SIR传染模型揭示BA网络正向溢出（+27.78）与WS网络抑制效应（−19.08）

### English Concise Version

**MultiWorld Causal Subsidy** ｜ Individual Research Project ｜ Python, CausalML, Mesa ABM
- Built a causal inference–AI simulation framework: T/X/DR/S-Learner ensemble + PSM/IPW/doubly-robust deconfounding with CausalSimulationPipeline (CATE→strategy→validation closed loop); multi-world evaluation (500 agents × 30 rounds) decouples strategy effects from noise via shared AgentConfig, 5-seed Monte Carlo confirms cognitive strategy robustness (CV=0.14, ROI=2.95, +26.5% over random); cognitively-grounded agents embed prospect theory/mental accounting/bounded rationality; SIR contagion reveals positive spillover in BA networks (+27.78) vs. inhibitory effects in WS networks (−19.08)

---

## 附：项目技术规格（简历外参考，不写入简历）

| 维度 | 规格 |
|------|------|
| 源码规模 | 16模块 / 5,581行 / 43公共API符号 |
| 测试覆盖 | 32个单元测试（全通过） |
| 论文文档 | 1,324行技术论文（中英摘要+7章+3附录+17篇参考文献） |
| 因果推断 | T/X/DR/S-Learner, PSM(3种匹配算法含optimal), IPW, DR Policy Value, Bootstrap CI, E-value |
| 仿真引擎 | Mesa 3.x ABM, 5种策略(random/static/dynamic/cognitive/cate_driven) |
| Agent模型 | 前景理论+心理账户(4类型+迁移)+有限理性+疲劳脱敏 |
| 网络拓扑 | BA无标度/WS小世界/属性相似度/POI共现 |
| 稳健性 | Monte Carlo多种子 + 参数扰动(±20%) + 效应量敏感性 |
| 基础设施 | MIT LICENSE, pyproject.toml, CI-ready |
