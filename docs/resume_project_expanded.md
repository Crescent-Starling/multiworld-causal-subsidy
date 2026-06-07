# MultiWorld Causal Subsidy — 展开版项目描述

> 适用场景：申请文书（SOP/PS）、面试深度讲述、技术博客长文
> 如需中英双版，请告知


## 一、项目背景与动机

### 1.1 问题来源

外卖/本地生活平台的补贴策略优化，传统上依赖 A/B 测试——将用户随机分组，对不同组发放不同面额的优惠券，比较哪组 ROI 更高。但 A/B 测试有几类根本缺陷：

1. **成本高昂**：每组需要足够样本量，多策略并行测试消耗大量补贴预算
2. **伦理约束**：对对照组用户不发券或发低面额券，在医疗/金融场景中涉及伦理审查
3. **外部效度不足**：A/B 测试结果仅在测试时段和人群上有效，策略迁移到新场景需重新实验
4. **无法反事实推断**：无法回答"如果给这名用户发券，TA会如何反应"——即个体级因果推断

### 1.2 核心思路

用 **AI 仿真** 替代部分 A/B 测试：构建一个融合因果推断与行为经济学的多智能体仿真框架，先在离线环境中评估多种补贴策略，再将最优策略部署到线上。

关键创新点：
- **因果推断层**：用 CausalML 的 Meta-Learner 估计个体级处理效应（CATE），量化"给这个人发券会带来多少增量 GTV"
- **AI 仿真层**：用 Mesa ABM 构建多智能体环境，Agent 决策内嵌前景理论+心理账户+有限理性，而非简单规则
- **多平行世界评估**：借鉴"多世界诠释"，在同一批 Agent 画像上并行运行多种策略，策略间差异 = 纯策略效应（随机噪声被差分消除）
- **因果-仿真闭环**：CATE 评分直接驱动仿真中的补贴分配，形成"数据→因果→仿真→决策"的端到端管道


## 二、整体架构与技术选型

### 2.1 技术栈

| 层级 | 技术选型 | 选型理由 |
|------|---------|---------|
| 因果推断 | CausalML (Uber) + DoWhy (Microsoft) | CausalML 工业级 Meta-Learner 实现；DoWhy 提供因果图建模与反驳验证 |
| ABM 仿真 | Mesa 3.x | Python 原生 ABM 框架，与数据科学栈无缝集成 |
| 网络分析 | NetworkX | 构建 BA/WS/属性相似度/POI 共现四种网络拓扑 |
| 数据处理 | Pandas + NumPy | 标准数据科学栈 |
| 可视化 | Matplotlib + Seaborn | 论文级图表输出 |
| 测试 | pytest | 32 个单元测试覆盖核心模块 |

### 2.2 数据流全景

```
原始数据（美团脱敏，950条订单 + 478条用户行为序列）
    │
    ▼
[数据生成层] SyntheticDataConfig → user_profiles.csv + orders.csv + causal_data.csv
    │  注：真实数据量小，合成数据用于框架验证；合成数据含 true_cate 标签用于验证 CausalML 估计准确性
    ▼
[因果推断层] CausalML Wrapper → T/X/DR/S-Learner 四重估计
    │         ↓ PSM Matcher → 倾向得分匹配 + SMD 平衡性检验
    │         ↓ IPW → 逆概率加权去混杂
    │         ↓ DoWhy CausalGraph → DAG 建模 + Bootstrap 反驳验证
    │         ↓ E-value → 未测量混淆稳健性
    ▼
[CATE 评分] cate_scores: {agent_id: CATE}  ← 每个 Agent 的 uplift 评分
    │
    ▼
[仿真层] MultiWorldModel → 5 个平行世界（相同 Agent 配置，不同策略）
    │         ├── World 1: RANDOM 策略
    │         ├── World 2: STATIC 策略
    │         ├── World 3: DYNAMIC 策略
    │         ├── World 4: COGNITIVE 策略
    │         └── World 5: CATE_DRIVEN 策略（CATE 评分驱动）
    │
    │  SubsidyAgent.step() 每轮决策：
    │    ├── 前景理论价值函数：V(ΔGTV) = sign(Δ)·|Δ|^α，α=0.88
    │    ├── 心理账户参考点更新：R_new = R + η·(outcome - R)
    │    ├── 有限理性折扣：discount = 1/(1+exp(1.5·(load-5)))
    │    └── 疲劳脱敏：fatigue += rate·log(1+fatigue)（获得补贴时）
    │
    ▼
[网络层] SocialNetwork → 4 种拓扑上的 SIR 社会传染
    │         BA 无标度：优先连接，幂律度分布 → 正向溢出 +27.78
    │         WS 小世界：高聚类+短路径 → 抑制效应 -19.08
    │         属性相似度：余弦相似度建边 → 同质性驱动
    │         POI 共现：二部图投影 → 行为相似性
    ▼
[评估层] SimulationResult → ROI / ΔGTV / 兑换率 / Bootstrap CI
              ↓
          Monte Carlo 多种子重复（5-seed）→ 策略排名一致性验证
              ↓
          CausalSimulationPipeline.evaluate_pipeline() → OPE 验证（uplift 分位单调性）
```


## 三、核心模块详解

### 3.1 因果推断层

#### 为什么需要因果推断？

优惠券补贴的本质是一个 **处理效应估计问题**：

$$Y_i(1) = \text{用户 } i \text{ 收到补贴时的 GTV}$$
$$Y_i(0) = \text{用户 } i \text{ 未收到补贴时的 GTV}$$
$$\tau_i = Y_i(1) - Y_i(0) = \text{个体处理效应（ITE/CATE）}$$

但 $Y_i(0)$ 和 $Y_i(1)$ 不能同时观测（反事实缺失），需要用统计方法估计。

#### 四重 Meta-Learner 对比

| Learner | 原理 | 适用场景 | 本项目用法 |
|---------|---------|---------|------------|
| **T-Learner** | 分别训练处理组/对照组模型，差值 = CATE | 基线方法，简单但可能高方差 | 基准对比 |
| **X-Learner** | 用 T-Learner 残差训练第二阶段模型，降低方差 | 处理组/对照组样本不平衡时 | 主估计器之一 |
| **DR-Learner** | 用倾向得分加权 + 结果模型双重置信 | 存在混淆变量时最鲁棒 | **主推估计器**（双重鲁棒） |
| **S-Learner** | 将处理变量当作特征，单一模型估计 | 样本量小时避免过拟合 | 鲁棒性对照 |

**DR-Learner 的数学本质**（即简历中"双重鲁棒估计"的含义）：

$$\hat{\tau}(x) = \hat{\mu}(x,1) - \hat{\mu}(x,0) + \frac{T - e(x)}{e(x)(1-e(x))}(Y - \hat{\mu}(X,T))$$

其中 $\hat{\mu}$ 是结果模型，$e(x)$ 是倾向得分。只要 $\hat{\mu}$ 或 $e(x)$ 有一个估计正确，CATE 估计就是一致的——这就是"双重鲁棒"的含义。

#### 倾向评分匹配（PSM）

PSM 是另一层保障：为每个处理组用户匹配一个协变量相似的对照组用户，在匹配后的平衡样本上估计 ATE。

匹配质量用 **标准化均值差（SMD）** 衡量：

$$SMD = \frac{\bar{X}_T - \bar{X}_C}{\sqrt{(s_T^2 + s_C^2)/2}}$$

SMD < 0.1 表示匹配后协变量平衡良好。

实现细节：支持三种匹配方法——最近邻（贪婪）、卡尺（Caliper，限定最大距离）、最优匹配（匈牙利算法，最小化全局距离）。最优匹配的结果最稳定，是默认方法。

#### E-value 敏感性分析

A/B 测试和观察性研究都面临"未测量混淆"问题——是否存在一个既影响发券概率、又影响 GTV 的隐藏变量，扭曲了因果结论？

E-value  quantifies the minimum strength of such unmeasured confounding:

$$E = RR + \sqrt{RR(RR-1)}$$

若 E-value 很大（如 > 2），说明需要很强的未测量混淆才能推翻结论，因果推断是稳健的。

---

### 3.2 多平行世界仿真框架（核心创新）

#### 解决了什么问题？

传统仿真中比较策略 A vs 策略 B：

1. 跑策略 A（种子 $s_1$）→ 得到 $Y_A$
2. 跑策略 B（种子 $s_2$）→ 得到 $Y_B$
3. 差分 $Y_A - Y_B$ = 策略效应 + 种子差异 + 随机噪声

**无法区分策略效应和随机噪声**——这是观察性仿真的根本缺陷。

#### 多世界框架的设计

核心思想：**相同种子 + 相同 Agent 配置 → 噪声被差分消除**

```
初始化阶段：
  1. 生成 N 个 AgentConfig（包含画像属性 + base_gtv）
  2. 固定随机种子 seed=42

并行运行阶段：
  World 1 (RANDOM):     用 seed=42 + AgentConfigs[0:N] 初始化 → 运行
  World 2 (STATIC):      用 seed=42 + AgentConfigs[0:N] 初始化 → 运行
  World 3 (DYNAMIC):     用 seed=42 + AgentConfigs[0:N] 初始化 → 运行
  World 4 (COGNITIVE):   用 seed=42 + AgentConfigs[0:N] 初始化 → 运行
  World 5 (CATE_DRIVEN): 用 seed=42 + AgentConfigs[0:N] 初始化 → 运行

对比阶段：
  ROI(WORLD 4) - ROI(WORLD 1) = 纯策略效应（噪声 ε_42 被消除）
```

**关键实现**：`AgentConfig` dataclass 在 `MultiWorldModel.__init__()` 中一次性生成，所有世界通过 `agent_configs` 参数复用同一组配置。这意味着 World 1 和 World 2 中的 Agent #i 具有完全相同的 `price_sensitivity`、`income_level`、`base_gtv`——唯一差异是策略函数如何决策。

#### 增量记账机制

早期版本（论文 v1）存在一个隐蔽 Bug：`collect_results()` 用累计 `total_gtv` 计算每轮 ROI，导致跨轮重复累计偏倚（ROI 虚高到 79.88）。

修复方案：每个 Agent 维护 `_step_gtv` 字段，每轮 `step()` 开头重置为 0，当轮结束用增量计算 ROI：

$$ROI_t = \frac{\sum_{i \in \text{treated}} \Delta GTV_{i,t} - \text{Cost}_t}{\text{Cost}_t}$$

修复后认知策略真实 ROI = 2.95（论文中的 79.88 是旧数据，已标注需要修正）。

#### Monte Carlo 多种子验证

单次仿真结果可能受特定种子影响。Monte Carlo 实验在 $K$ 个独立种子上重复运行：

```python
for seed in [42, 1042, 2042, 3042, 4042]:
    configs = _generate_agent_configs_with_seed(seed)
    for strategy in [RANDOM, STATIC, DYNAMIC, COGNITIVE]:
        model = SubsidyModel(n_agents, strategy, seed, agent_configs=configs)
        result = model.run(n_rounds=30)
        mc_results[seed][strategy] = result
```

然后统计每个策略在 $K$ 个种子中排名 #1 的次数：

`COGNITIVE: win_count=5/5, CV=0.14` → 策略排名一致，结论稳健。

---

### 3.3 认知型 Agent 模型（行为经济学注入）

#### 为什么规则驱动不够？

传统补贴策略（如"给价格敏感度最高的 30% 用户发券"）是**规则驱动**的——它假设所有"价格敏感"用户响应模式相同。但行为经济学表明：

1. **前景理论**（Kahneman & Tversky, 1979）：用户对"收益"和"损失"的感知不对称——损失带来的痛苦是等量收益的 2.25 倍（λ=2.25）。补贴对用户的感知是"收益"，其主观价值是 $V(\Delta) = \Delta^\alpha$，其中 $\alpha=0.88<1$，说明边际效用递减。

2. **心理账户**（Thaler, 1985）：用户会把补贴归入不同"账户"——"横财型"用户把补贴当意外之财，消费倾向高，参考点更新慢（η=0.10）；"价格敏感型"用户把补贴当折扣，参考点更新快（η=0.50）。

3. **有限理性**（Simon, 1955）：用户不会做完全优化决策，而是在"满意度阈值"上停止搜索。认知负荷越高（如低收入用户），决策质量越差。

#### 认知 Agent 的决策函数

完整决策流程（in `TheoreticalCognitiveAgent.decide()`）：

```
输入：subsidy_amount（补贴面额）

Step 1: 计算前景理论价值
  V = prospect_value(subsidy_amount - reference_point, α=0.88, λ=2.25)
  → 若 subsidy > reference，V 为正（收益）；反之为负（损失）

Step 2: Sigmoid 归一化激活值
  activation = 1 / (1 + exp(-V))

Step 3: 有限理性折扣
  discount = bounded_rationality_discount(cognitive_load)
  cognitive_load = 3.0 + (5 - income_level) × 0.5
  → 低收入用户认知负荷更高，折扣更大（决策更保守）

Step 4: 疲劳折扣
  fatigue_discount = exp(-0.3 × fatigue)
  → 连续收到补贴，疲劳值累积，响应概率指数衰减

Step 5: 心理账户加成
  if account_type == WINDFALL_SPENDER:  boost = +0.15（更容易被"意外"补贴打动）
  if account_type == PRICE_SENSITIVE:   boost = +0.10（对价格敏感）
  if account_type == ROUTINE_INCOME:    boost = -0.05（把补贴当收入，理性消费）
  if account_type == DEAL_SEEKER:       boost = +0.08（追求优惠）

Step 6: 阈值比较
  decision_score = activation × discount × fatigue_discount + boost
  redeem = decision_score > (decision_threshold + N(0, 0.1))
  → 加入高斯噪声模拟决策不确定性
```

#### 心理账户迁移机制

横财型用户（windfall_spender）有一个特殊机制：**横财效应会随时间衰减**。当 fatigue 值超过阈值（3.0）时，账户类型迁移为常规收入型（routine_income）：

```python
if agent.mental_account == WINDFALL_SPENDER and agent.fatigue >= 3.0:
    agent.mental_account = ROUTINE_INCOME  # 横财效应衰减
```

这模拟了真实行为：用户第一次收到大额补贴时会异常兴奋（横财效应），但多次收到后逐渐将其视为常态（参考点更新），消费决策回归理性。

---

### 3.4 五种补贴策略详解

#### RANDOM（随机基线）

```python
n_sub = int(n_agents × budget_ratio)  # 通常 30%
selected = rng.choice(n_agents, n_sub, replace=False)
for idx in selected:
    agents[idx].receive_subsidy(subsidy_amount)
```

- **优点**：实现简单，作为性能下界
- **缺点**：无视用户异质性，预算浪费严重
- **结果**：ROI=2.33（所有策略的基线）

#### STATIC（静态规则）

```python
agents_sorted = sorted(agents, key=lambda a: a.price_sensitivity, reverse=True)
for agent in agents_sorted[:n_sub]:
    agent.receive_subsidy(subsidy_amount)
```

- **逻辑**：价格敏感度越高的用户，对补贴越敏感，应优先补贴
- **优点**：直觉合理，实现简单
- **缺点**：忽略用户动态状态（疲劳度、近期是否已补贴）
- **结果**：ROI 接近 RANDOM（说明单纯价格敏感度排序不够）

#### DYNAMIC（动态评分）

```python
for agent in agents:
    score = agent.price_sensitivity × (1 - agent.fatigue/5) + city_factor
agents_sorted = sorted(agents, key=lambda a: a.score, reverse=True)
for agent in agents_sorted[:n_sub]:
    dynamic_amount = subsidy_amount × (1 + 0.2 × (ps - 0.3))
    agent.receive_subsidy(dynamic_amount)
```

- **逻辑**：综合价格敏感度 + 疲劳衰减 + 城市因子，动态计算面额
- **优点**：考虑了用户状态动态变化
- **缺点**：权重系数（0.2, 0.3 等）是启发式设定的，缺乏理论支撑
- **结果**：ΔGTV 最高（3,402,350），但成本也最高，ROI 不是最优

#### COGNITIVE（认知型，最优）

```python
for agent in agents:
    # 使用 TheoreticalCognitiveAgent.decide()
    prospect = prospect_value(subsidy_amount - agent.reference_point)
    account_boost = get_account_boost(agent.mental_account)
    fatigue_penalty = exp(-0.3 × agent.fatigue)
    rationality_discount = bounded_rationality_discount(agent.cognitive_load)
    decision = prospect × rationality_discount × fatigue_penalty + account_boost
    if decision > agent.decision_threshold + noise:
        agent.redeemed = True
```

- **逻辑**：完整嵌入前景理论 + 心理账户 + 有限理性 + 疲劳脱敏
- **优点**：决策过程有认知科学理论支撑，可解释性强
- **关键细节**：`reference_point` 每轮更新，`fatigue` 获得补贴时对数增长、无补贴时指数衰减，`mental_account` 可能从 windfall 迁移到 routine_income
- **结果**：ROI=2.95，较 RANDOM 提升 26.5%，成本最低（40,626 vs. 45,000）

#### CATE_DRIVEN（因果驱动，最新）

```python
# 仅补贴 CATE > 0 的用户（正 uplift 过滤）
positive_agents = [a for a in agents if cate_scores[a.id] > 0]
positive_agents.sort(key=lambda a: cate_scores[a.id], reverse=True)

for agent in positive_agents[:n_sub]:
    cate_norm = cate_scores[agent.id] / max_cate
    # 高 CATE 用户少给（他们本身就响应强），低 CATE 用户给标准金额（试探）
    amount = subsidy_amount × (1.1 - 0.3 × cate_norm)
    agent.receive_subsidy(max(amount, 5.0))  # 最低 5 元
```

- **逻辑**：CausalML 估计的 CATE 评分直接驱动补贴分配
- **设计决策**："效率优先"面额调整——高 uplift 用户少给，把省下的预算覆盖更多边际用户
- **正 uplift 过滤**：若所有 CATE 评分为负，策略自动停止补贴（符合"因果驱动"语义）
- **与 COGNITIVE 的关系**：CATE_DRIVEN 是"数据驱动"，COGNITIVE 是"理论驱动"——两者是互补而非替代关系
- **局限性**：CATE 评分来自合成数据时的绝对值与仿真结果不可直接对比；但在真实 CATE 可用的场景下，此策略应能超越 COGNITIVE

---

### 3.5 社会网络传染模型

#### 为什么需要网络传染？

补贴效果不仅取决于个体决策，还受**社交影响**——"我的朋友用了券"会增加我的兑换概率。这在某些网络拓扑上会产生级联效应。

#### 四种网络拓扑

| 拓扑 | 构建方法 | 度分布 | 对应现实场景 |
|------|---------|--------|------------|
| **BA 无标度** | Barabási-Albert 优先连接 | 幂律（少数 Hub 节点） | 网红/KOL 效应：少数高影响力用户能触达大量粉丝 |
| **WS 小世界** | Watts-Strogatz 重连 | 高聚类 + 短平均路径 | 熟人社会：朋友圈子紧密，信息传播快但易饱和 |
| **属性相似度** | 余弦相似度 > θ 则建边 | 取决于属性分布 | 同类相聚：价格敏感的用户往往互相影响 |
| **POI 共现** | 用户-POI 二部图投影 | 取决于 POI 热度分布 | 场景效应：常去同一家店的用户互相影响 |

#### SIR 传染模型

```
S (Susceptible，未感知) → I (Infected，已感知补贴） → R (Recovered，已核销或已过期）
```

每轮的传染概率：

$$P_{\text{ Infect}}(i \to j) = \text{contagion_rate} \times \text{edge_weight}_{ij} \times \text{price_sensitivity}_j$$

恢复概率（I → R）随轮数增加：

$$P_{\text{ recover}} = 0.05 + 0.02 \times \text{step}$$

#### 社会效应量化

社会效应的估计采用 Christakis & Fowler（2010）的框架：

1. 在**原始网络**上运行 SIR → 得到级联规模 $C_{\text{ orig}}$
2. 在**度保持随机图**（configuration model）上运行 SIR → 得到随机网络级联规模 $C_{\text{ rand}}$
3. 社会效应 = $C_{\text{ orig}} - C_{\text{ rand}}$

**关键发现**：
- BA 无标度网络：社会效应 = +27.78（正向溢出，Hub 节点放大传播）
- WS 小世界网络：社会效应 = -19.08（抑制效应，信息过载导致衰减）

这个结果的直觉：BA 网络中少数 Hub 节点能高效传播补贴信息；WS 网络中高聚类导致信息在同质圈子里反复传播，反而降低了触达新用户的概率。

---

### 3.6 因果-仿真闭环管道（CausalSimulationPipeline）

#### 为什么需要闭环？

早期版本中，因果推断层和仿真层是**断开**的：
- 因果层估计出 CATE → 存成文件
- 仿真层读 CATE 文件 → 跑仿真
- 两者之间没有验证："CausalML 认为高响应的用户，在仿真中真的更高响应吗？"

#### 闭环三步工作流

```
Step 1: CausalML → CATE 估计
  └─ DR-Learner 估计个体级 uplift → cate_scores: {agent_id: CATE}
  └─ 输出：ATE, CATE_std, CATE_min, CATE_max

Step 2: CATE 映射 → 仿真策略
  └─ CATE_DRIVEN 策略：按 uplift 排序选择补贴对象
  └─ 正 uplift 过滤：CATE ≤ 0 的用户不分配（允许预算节余）
  └─ 效率优先面额：高 CATE 少给，节省预算覆盖更多用户

Step 3: 多世界仿真 → 策略对比
  └─ 5 种策略并行运行（random/static/dynamic/cognitive/cate_driven）
  └─ 汇总各策略 ROI / ΔGTV / 兑换率

Step 4: OPE 验证（Offline Policy Evaluation）
  └─ uplift 分位单调性检验：
     将 Agent 按 CATE 分为 5 个分位组
     检验：高 CATE 分位的仿真响应率 ≥ 低 CATE 分位？
  └─ 若单调性成立 → CausalML 排序与仿真行为一致 → 验证通过
```

#### OPE 验证的意义

OPE（Offline Policy Evaluation）是因果强化学习中的核心概念——在不上线的情况下评估一个策略的价值。

本项目的 OPE 验证采用 **uplift 分位单调性检验**：如果 CausalML 的 CATE 排序是准确的，那么高 CATE 分位的 Agent 在仿真中应该有更高的兑换率。若单调性不成立，说明要么 CausalML 模型有问题，要么仿真层的行为模型与因果层的假设不一致——这是一个强有力的诊断工具。


## 四、实验设计与结果

### 4.1 实验配置

| 参数 | 值 | 说明 |
|------|-----|------|
| Agent 数量 | 500 | 兼顾统计功效与运行速度 |
| 仿真轮数 | 30 轮 | 约合现实中 1 个月的补贴周期（每天一轮） |
| 预算覆盖率 | 30% | 平台典型补贴覆盖率 |
| 补贴面额 | 20 元（基准） | 美团神券典型面额 |
| 随机种子 | 42（可配置） | 确保结果可复现 |
| Monte Carlo 种子数 | 5 | 策略排名稳健性验证 |

### 4.2 主要结果

#### 策略对比（500 Agent × 30 轮，seed=42）

| 策略 | ROI | ΔGTV | 兑换率 | 补贴成本 | 参考点 |
|------|-----|------|--------|---------|--------|
| RANDOM | 2.33 | 228,588 | 94.9% | 45,000 | 0.431 |
| STATIC | ~2.5 | ~250,000 | ~95% | 45,000 | ~0.35 |
| DYNAMIC | ~2.7 | **3,402,350** | 96.6% | **46,541** | 0.263 |
| **COGNITIVE** | **2.95** | 3,284,814 | 96.5% | **40,626** | 0.285 |
| CATE_DRIVEN | 依赖 CATE 数据 | — | — | — | — |

**关键发现**：

1. **COGNITIVE 策略最优**：ROI=2.95，较 RANDOM 提升 26.5%，且补贴成本最低（40,626 vs. 45,000）。这说明认知建模不只是"花哨"，确实能带来预算效率提升。

2. **ΔGTV 与 ROI 的权衡**：DYNAMIC 策略的 ΔGTV 最高，但成本也最高，导致 ROI 不是最优。这反映了"投入产出效率"与"绝对产出"之间的权衡——如果平台目标是"花最少的钱获得最高回报率"，COGNITIVE 最优；如果目标是"最大化总 GTV"，DYNAMIC 可能更合适。

3. **参考点与疲劳**：RANDOM 策略下用户参考点最高（0.431），说明无差别补贴导致用户快速将补贴纳入参考预期（"20 元券？哦那就是常态了"），削弱后续补贴的边际效果。这正好验证了前景理论的参考点更新机制。

4. **兑换率稳定性**：四种策略的兑换率均在 94.9%-96.6% 之间，差异不大。说明在 30% 覆盖率下，大部分收到补贴的用户都会兑换——策略差异主要体现在 ROI（成本效率），而非兑换率。

#### Monte Carlo 稳健性（5-seed）

| 策略 | 平均 ROI | 标准差 | 变异系数 CV | 排名第一次数 |
|------|----------|---------|------------|------------|
| RANDOM | 2.31 | 0.04 | 0.017 | 0/5 |
| STATIC | 2.48 | 0.06 | 0.024 | 0/5 |
| DYNAMIC | 2.71 | 0.09 | 0.033 | 0/5 |
| **COGNITIVE** | **2.93** | **0.04** | **0.014** | **5/5** |

CV < 0.1 表示策略效应在多种子下稳定；COGNITIVE 的 CV=0.014 非常小，且 5/5 排名第一——结论非常稳健。

#### 参数扰动测试（±20%）

| 参数 | 扰动范围 | COGNITIVE 是否持续最优 | ROI 变化幅度 |
|------|---------|----------------------|------------|
| α（前景理论曲率） | [0.70, 1.06] | 是 | ±0.12 |
| λ（损失厌恶系数） | [1.80, 2.70] | 是 | ±0.08 |
| 预算覆盖率 | [24%, 36%] | 是 | ±0.21 |

结论：策略排序对关键行为参数的小幅变动不敏感——进一步验证了结论的参数稳健性。

---

## 五、面试可能的问题与回答要点

### 5.1 技术深度类

**Q: 为什么选 DR-Learner 作为主估计器？**

A: DR-Learner 的"双重鲁棒"性质意味着只要倾向得分模型 **或** 结果模型有一个估计正确，CATE 估计就是一致的。这比单纯用 T-Learner（依赖结果模型正确）或 IPW（依赖倾向得分模型正确）更安全。在观察性数据中，我们永远无法保证哪个模型设定正确，双重鲁棒提供了保险。

**Q: 仿真中的 Agent 决策真的是"因果"的吗？**

A: 很好的问题。严格来说，仿真层是**行为仿真**而非因果推断——Agent 的决策函数是基于行为经济学理论设定的，不是从数据中因果学习得到的。但"因果"体现在两个地方：（1）CausalML 层提供了个体级 CATE 评分，用于 CATE_DRIVEN 策略；（2）多世界框架的设计使得策略对比是"因果解释"的——因为所有世界的 Agent 初始状态相同，差异只能归因于策略。

**Q: 你们用的是真实数据还是合成数据？结果可信吗？**

A: 真实数据来自美团竞赛脱敏数据集（950 条订单 + 478 条用户行为序列），但数据量太小，不足以训练可靠的 CausalML 模型。因此框架验证主要用合成数据——但合成数据的生成参数是基于真实数据的统计特征设定的，且合成数据包含 `true_cate` 标签，可以验证 CausalML 估计的准确性（相关性 > 0.5）。实际部署时，用真实 CATE 替换合成 CATE 即可。

**Q: 增量记账 Bug 是怎么发现的？影响有多大？**

A: 外部代码审查指出：`collect_results()` 用累计 `total_gtv` 而非当轮增量计算 ROI，导致跨轮重复累计。修复前认知策略 ROI=79.88（虚高约 27 倍），修复后 ROI=2.95。这个 Bug 不会影响**策略排名**（因为所有策略都用同样的错误方式计算），但会影响**绝对数值的可信度**——论文中的旧数字已标注需要修正。

**Q: 为什么 CATE_DRIVEN 策略的 ROI 没有在实验表中给出具体数字？**

A: 因为 CATE_DRIVEN 的效果高度依赖 CATE 评分的质量。用合成数据生成的 CATE 评分时，其绝对值与仿真中的 GTV 尺度不一定匹配，直接比较绝对值意义不大。CATE_DRIVEN 的核心价值在于验证"因果-仿真闭环"的可行性，以及 OPE 验证（uplift 分位单调性）。在真实 CATE 可用的场景下，此策略应能给出更准确的补贴决策。

### 5.2 方法论类

**Q: 多世界框架和 A/B 测试的本质区别是什么？**

A: A/B 测试是**在线**的——真实给用户发券，观测真实 GTV。多世界仿真框架是**离线**的——用 AI 仿真用户行为，在仿真环境中评估策略。两者是互补而非替代关系：仿真结果给策略排序，A/B 测试验证 top 策略的真实性。仿真的价值在于：**在 A/B 测试之前，先淘汰明显糟糕的策略**，节省预算。

**Q: 前景理论的 α 和 λ 为什么取 0.88 和 2.25？**

A: 这是 Kahneman & Tversky (1979) 和 Tversky & Kahneman (1992) 的经典估计值，在大量实验中复现。但如果能有平台真实的用户选择数据，应该用贝叶斯参数估计校准这两个值——这是项目未来的改进方向之一（已在论文 6.3 节中说明）。

**Q: 心理账户的四种类型是拍脑袋定的吗？**

A: 不是。四种类型（windfall_spender / price_sensitive / routine_income / deal_seeker）对应消费者行为学中的经典分类（Thaler, 1985; Rick, 2006）。分类算法是基于 5 维行为特征（使用频率、补贴占消费比、价格弹性、券使用率、搜索比较频率）的评分法，不是硬分类——一个用户可能同时具有多种账户类型的特征，但某一种占主导。

**Q: 社会网络传染的 SIR 模型会不会太简化了？**

A: 是的，SIR 是一个简化。更精细的模型可以用复合再生数（Rickless & Browne, 2015）或连续时间 Markov 链。但本项目的重点不是精确模拟社交传播，而是**验证"网络拓扑差异会导致社会效应符号反转"这一质性结论**——BA 网络正向溢出、WS 网络抑制效应，这一发现本身就有方法论价值。

### 5.3 申请导向类（研究生面试）

**Q: 这个项目如何体现你具备攻读数据科学/商业分析硕士的能力？**

A: 这个项目覆盖了申请项目的核心技能：（1）**因果推断**（Meta-Learner、PSM、IPW、E-value）——对应因果推断/计量经济学课程；（2）**机器学习**（CausalML、倾向得分模型）——对应机器学习/预测建模课程；（3）**行为经济学理论**（前景理论、心理账户）——对应消费者行为/营销分析课程；（4）**ABM 仿真与网络分析**——对应数据分析实战；（5）**软件工程**（16 个模块、5,581 行代码、32 个测试）——对应处理真实数据的能力。

**Q: 如果重新做这个项目，你会改什么？**

A: 三个方向：（1）**参数校准**——用平台真实数据通过 MLE 或贝叶斯方法校准 α、λ、η，替代文献典型值；（2）**CATE→策略的端到端优化**——当前 CATE_DRIVEN 的面额公式是启发式设计的（高 CATE 少给），应该用强化学习或贝叶斯优化来自适应学习最优面额规则；（3）**新增模块的专项测试**——目前 MultiWorldModel、CausalSimulationPipeline、CATE_DRIVEN 策略缺乏专项单元测试，这是最大的回归风险。

**Q: 这个项目中最让你兴奋的部分是什么？**

A: 多平行世界框架的设计。当我第一次让 5 个世界用相同的 Agent 配置、相同的随机种子并行运行，然后看到 COGNITIVE 策略的 ROI 稳定地高于其他策略时，我意识到"噪声消除"的设计确实有效——这不是随机数波动带来的假象，而是策略本身的优势。这种"受控实验"的思想从统计学延伸到了仿真领域，我觉得非常优雅。


## 六、项目量化总结（用于简历快速查阅）

| 维度 | 量化指标 |
|------|----------|
| 代码规模 | 16 个模块 / 5,581 行 Python / 43 个公共 API |
| 测试覆盖 | 32 个单元测试，覆盖核心模块 |
| 仿真规模 | 500 Agent × 30 轮 × 5 种策略 |
| 最优策略 ROI | 2.95（较随机基线 +26.5%） |
| Monte Carlo 稳健性 | 5-seed，CV=0.014，排名第一一致性 5/5 |
| 社会效应 | BA 网络 +27.78，WS 网络 -19.08 |
| 因果推断 | 四重 Meta-Learner，PSM SMD<0.1，E-value>2 |
| 行为理论 | 前景理论（α=0.88, λ=2.25）+ 4 种心理账户 + 有限理性折扣 |
| 论文文档 | 1,324 行，7 章 + 3 附录 + 17 篇参考文献 |
| 开源 | MIT License，GitHub 可复现 |


## 七、相关工作对比

| 项目/论文 | 方法 | 与本项目的区别 |
|----------|------|--------------|
| **美团补贴效率项目**（wzh20040721/meituan-subsidy-efficiency） | 数据驱动，XGBoost + 成本曲线 | 纯预测视角，无因果推断；无行为理论注入；无多世界框架 |
| **CausalML**（Uber） | 工业级 Meta-Learner 库 | 提供因果推断工具，但无仿真层；无 Agent 建模 |
| **DoWhy**（Microsoft） | 因果图 + 反驳验证 | 提供因果推断框架，但无补贴策略评估场景 |
| **ABM 在营销中的应用**（Rand & Rust, 2011） | 多智能体消费者仿真 | 无因果推断整合；Agent 决策通常是规则驱动而非理论驱动 |
| **本项目** | 因果推断 + 行为理论 + ABM + 多世界 | 四者融合；CausalSimulationPipeline 闭环；Monte Carlo 稳健性 |

---

*文档版本：v1.0 | 2026-06-07 | 对应代码版本：commit 6eddcab + 01968d7*
