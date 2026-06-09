# MultiWorld Causal Subsidy — 简历项目描述（v5 架构升级版）

> 本版更新要点：(1) 突出"概率建模为函数"的核心方法论创新；(2) 融入行为链核提取与上下文化概率建模；(3) 涵盖已实现+规划中的三类Agent交互；(4) 量化数字使用修复后真实数据

---

## 🇨🇳 中文版

**2026.03-2026.05 因果驱动的单用户ABM仿真框架及补贴策略优化 第六届美团商务分析精英大赛**

背景：平台补贴长期以"用户包+统一券面额"批量投放，核销率低且边际效益递减；优化方向是走向个性化分发，但面临ML无法回答反事实、A/B测试成本高、传统仿真忽视个体异质性三重瓶颈。构建因果推断与行为经济学驱动的ABM仿真融合框架，设计从批量到个性化的完整策略光谱，实现离线因果评估与最优剂量优化。

因果推断层：设计T/X/DR/S四类Meta-Learner估计CATE，结合PSM+IPW+双重鲁棒消除选择偏差；修复DR-Learner倾向得分极端值导致的数值爆炸（CATE相关性-0.03→0.84）；在美团MT-LIFT数据集（550万RCT）上交叉验证，X-Learner AUUC=0.575；DoWhy因果图+Bootstrap CI+E-value检验未观测混杂敏感性。

行为建模层：①**行为链核提取**——从用户行为链提取参数化UserKernel(θ基础率,β敏感度,γ认知调制)，替代硬编码全局参数；层级贝叶斯James-Stein收缩解决小样本估计，差分隐私保护参数发布。②**上下文化概率建模**——决策概率建模为上下文函数P(step|θ,W_θ·z_c)而非固定标量，同一用户跨场景行为概率差异达7.2%；剂量-响应函数g(d)=log(1+d/scale)捕捉边际递减（0→5元+2.5%，10→15元仅+0.7%），实现连续剂量优化。

仿真模拟层：构建多平行世界框架，同组Agent相同初始条件对比策略差异，解耦假设风险与随机噪声。实现规则型、认知型（前景理论/心理账户/有限理性）、LLM驱动三类Agent，验证不同心理账户在递增门槛下的差异化拒绝行为。

成果：认知策略ROI=2.95（较随机基线+26.5%）；5-seed Monte Carlo策略排名一致CV=0.14。规划：商家核φ_m与用户核θ_i上下文耦合，实现P(行为|θ_i,φ_m,z_c,d)三方博弈仿真。

---

## 🇬🇧 English Version

**2026.03-2026.05 Causal-Driven Individual-Level ABM Simulation for Subsidy Optimization 6th Meituan Business Analytics Elite Competition**

Background: Platform subsidies have long operated in a "user segment + flat coupon" batch mode with low redemption rates and diminishing returns. The industry is shifting toward personalized distribution, but faces three bottlenecks—ML cannot answer counterfactuals, A/B testing is costly, and traditional simulation ignores individual heterogeneity. This project builds a causal inference and behavioral-economics-driven ABM simulation framework, designing a full strategy spectrum from batch to personalized allocation for offline causal evaluation and optimal dosage optimization.

Causal inference layer: Designed T/X/DR/S four-way Meta-Learner ensemble to estimate CATE; combined PSM, IPW, and doubly-robust estimation to deconfound selection bias; fixed DR-Learner numerical explosion from extreme propensity scores (CATE correlation: -0.03→0.84); cross-validated on Meituan MT-LIFT dataset (5.5M RCT), X-Learner AUUC=0.575; DoWhy causal DAG + Bootstrap CI + E-value sensitivity analysis.

Behavioral modeling layer: ①**Behavioral kernel extraction** — Extracted parametric UserKernel(θ base rate, β sensitivity, γ cognitive modulation) from behavioral chains, replacing hardcoded global parameters; hierarchical Bayesian James-Stein shrinkage for small-sample estimation; differential privacy for kernel publication. ②**Contextual probability modeling** — Decision probability modeled as context function P(step|θ,W_θ·z_c) rather than fixed scalar; same user exhibits 7.2% behavior probability difference across scenarios; dose-response g(d)=log(1+d/scale) captures diminishing returns (0→5 yuan +2.5%, 10→15 only +0.7%), enabling continuous dosage optimization.

Simulation layer: Multi-world framework runs same agent cohort under multiple strategies with identical initial conditions, decoupling assumption risk from random noise. Three-tier agent architecture: rule-based / cognitive (prospect theory / mental accounting / bounded rationality) / LLM-driven; experiments verified differentiated rejection behaviors across mental accounting types under increasing thresholds.

Results: Cognitive strategy ROI=2.95 (+26.5% over random baseline); 5-seed Monte Carlo consistent ranking (CV=0.14). Future: Merchant kernel φ_m coupled with user kernel θ_i via shared context, enabling P(behavior|θ_i,φ_m,z_c,d) tripartite game simulation.

---

## 字数统计

| 版本 | 中文含空格 | 英文单词数 | 预估A4排版行数 |
|------|------------|------------|----------------|
| v4（上版） | ~440字 | ~120词 | ~14行 |
| **本版（v5.1）** | ~540字 | ~170词 | ~17行 |
| 目标（一页简历） | <600字 | <180词 | <18行 |

> v5.1基于用户润色版四层结构（背景/因果/行为建模/仿真），融合v5的批量→个性化洞察和量化数字。规划段包含待实现内容，后续需真正落地。

---

## 技术规格参考表（不在简历中展示，仅供面试准备）

| 维度 | 技术细节 |
|------|----------|
| 因果推断 | CausalML (T/X/DR/S-Learner)、DoWhy (DAG+反驳)、PSM (nearest/caliper/optimal)、IPW、双重鲁棒 |
| 核提取 | BehavioralKernelExtractor (层级贝叶斯+DP)、KernelPopulationSampler (多变量正态)、ContextualKernelExtractor (Ridge logistic) |
| 上下文 | ContextConfig (5维→17维one-hot: 品类5+价格3+时段4+意图4+竞争1) |
| 剂量-响应 | g(d)=log(1+d/scale), optimal_dosage() 网格搜索 |
| 认知理论 | 前景理论(α=0.88,λ=2.25)、心理账户(4类+迁移)、有限理性(cognitive_load)、疲劳脱敏 |
| 仿真框架 | Mesa 3.x、MultiWorldModel (6种策略)、Monte Carlo + 参数扰动 + 效应量敏感性 |
| LLM | DeepSeek API (deepseek-v4-flash)、8Agent×3轮、PromptTemplate (4种心理账户) |
| 社会 | SocialNetwork (BA/WS/属性相似度/POI共现)、SocialContagion (SIR) |
| 评估 | Bootstrap CI、E-value、AUUC/Qini、ROI (净利润率)、ΔGTV、策略对比 |
| 数据 | 竞赛950条+478条、MT-LIFT 550万RCT (10%采样验证) |
| 规划中（不在简历中） | 商家Agent (MerchantKernel φ_m)、三方交互、在线学习闭环、SEGMENT_COHORT策略 |
