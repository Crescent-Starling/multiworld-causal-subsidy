# MultiWorld Causal Subsidy — 简历项目描述

> **用途**：研究生申请（数据科学/统计学/商业分析硕士项目）
> **原则**：格式对齐简历其他项目；逻辑优先，数字克制；突出因果推断合理性与仿真有效性
> **注意**：本项目为个人独立完成，不提及竞赛背景

---

## 🇨🇳 中文版（当前使用版）

**2025.09-2026.06 MultiWorld Causal Subsidy  独立研究**
背景：外卖平台补贴策略面临因果评估难题——ML预测模型无法回答反事实问题，A/B测试成本高周期长。项目构建因果推断与ABM仿真融合框架，实现补贴政策的离线评估与策略排名。
因果推断层：设计T/X/DR/S四重Meta-Learner估计个体处理效应（CATE），结合PSM、IPW与双重鲁棒估计消除选择偏差；AUUC指标验证CATE排序的补贴策略价值（X-Learner AUUC=0.657，高于随机基线0.5）；Bootstrap置信区间与E-value检验因果结论是否受未观测因素影响。
仿真评估层：多平行世界框架使同一组用户画像在完全相同初始条件下运行多种策略，不同策略的结果差异只来自策略本身，排除随机噪声的干扰；Monte Carlo多seed与参数扰动敏感性分析验证策略排名稳健性。
主要成果：认知型策略ROI达2.95，较随机基线提升26.5%；10次Monte Carlo实验策略排名一致（CV=0.24）；三层Agent架构（规则型/认知型/LLM-driven）嵌入前景理论与心理账户机制，LLM实验验证不同心理账户类型在递增门槛下的差异化理性拒绝行为，与行为经济学理论预测一致。

---

## 🇬🇧 English Version（当前使用版）

**2025.09-2026.06 MultiWorld Causal Subsidy  Independent Research**
Context:  Subsidy policy evaluation faces a causal inference challenge—ML prediction cannot answer counterfactual questions, and A/B testing is costly and slow. This project builds a causal inference–ABM simulation fusion framework for offline subsidy policy evaluation and strategy ranking.
Causal inference layer:  Designed a T/X/DR/S four-way Meta-Learner ensemble to estimate Conditional Average Treatment Effects (CATE); combined PSM, IPW, and doubly-robust estimation to deconfound selection bias; AUUC score validated the ranking value of CATE for subsidy targeting (X-Learner AUUC=0.657 vs. random baseline ≈0.5); Bootstrap CIs and E-value tested whether causal conclusions are sensitive to unmeasured confounders.
Simulation evaluation layer:  Multi-world framework runs the same user profiles under multiple strategies with identical initial conditions, so observed differences arise solely from policy design, isolating strategy effects from random noise; Monte Carlo multi-seed and parameter perturbation sensitivity analysis verified strategy ranking robustness.
Key results:  Cognitive strategy achieved ROI=2.95, +26.5% over random baseline; strategy ranking was consistent across 10 Monte Carlo seeds (CV=0.24); three-tier agent architecture (rule-based / cognitive / LLM-driven) embeds prospect theory and mental accounting mechanisms; LLM experiments verified differentiated rational rejection behaviors across mental accounting types under increasing redemption thresholds, consistent with behavioral economics theory predictions.

---

## 附：项目技术规格（简历外参考，不写入简历）

| 维度 | 规格 |
|------|------|
| 源码规模 | 16模块 / 5,581行 / 43公共API符号 |
| 测试覆盖 | 32个单元测试（全通过） |
| 论文文档 | 1,324行技术论文（中英摘要+7章+3附录+17篇参考文献） |
| 因果推断 | T/X/DR/S-Learner, PSM(3种匹配算法含optimal), IPW, DR Policy Value, Bootstrap CI, E-value, AUUC/Qini |
| 仿真引擎 | Mesa 3.x ABM, 5种策略(random/static/dynamic/cognitive/cate_driven) |
| Agent模型 | 三层架构：规则型/认知型（前景理论+心理账户+有限理性+疲劳脱敏）/LLM-driven（DeepSeek API, CoT推理, 已真实验证） |
| 网络拓扑 | BA无标度/WS小世界/属性相似度/POI共现 |
| 稳健性 | Monte Carlo多种子 + 参数扰动(±20%) + 效应量敏感性 |
| 基础设施 | MIT LICENSE, pyproject.toml, CI-ready |

## 版本历史

| 版本 | 说明 |
|------|------|
| v1（原版） | 三段并列，信息密度高，数字堆砌 |
| v2（逻辑版） | 每段以"为什么→方法→价值"展开，精简数字 |
| v3（格式对齐版） | 对齐简历其他项目格式（时间/角色/背景+分点结构） |
| **v4（当前版）** | 简洁版，压缩约24%字数，删除"解耦量化"等不自然学术表述 |
