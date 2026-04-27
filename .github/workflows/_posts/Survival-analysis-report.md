---
layout: default
title: "生存分析报告: Telco Customer Churn 生存分析"
author: 陈思琪
date: 2026-04-26
---

## 1. 生存分析理论基础

生存分析（Survival Analysis）是统计学中专门研究"从起始时刻到某一特定事件发生所持续时间"的方法论。在本项目中：

- **起始时刻：** 客户开始订阅电信服务的时刻（tenure = 0）。
- **事件（Event）：** 客户流失（Churn = 1）。
- **时间变量（Time）：** 客户已使用月数（tenure）。
- **删失（Censoring）：** 部分客户在观测期结束时仍未流失（Churn = 0），属于右删失数据。

本案例采用以下核心模型：

1. **Kaplan-Meier（KM）非参数估计：** 用于描述整体客户留存概率随时间的变化，不引入任何协变量。其生存函数公式为：
   $$S(t) = \prod_{t_1 \leq t} \left(1 - \frac{d_i}{n_i}\right) \quad (1)$$
   其中 $d_i$ 为 $t_i$ 时刻的流失事件数，$n_i$ 为 $t_i$ 时刻仍处于风险集的客户数。

2. **Cox比例风险模型（Cox Proportional Hazards Model）：** 半参数模型，引入协变量（如家庭情况、互联网服务类型等），估计各因素对流失风险的倍数影响。其风险函数为：
   $$h(t \mid X) = h_0(t) \exp (\beta_1 X_1 + \beta_2 X_2 + \dots + \beta_p X_p) \quad (2)$$

3. **加速失效时间模型（AFT，Accelerated Failure Time Model）：** 当Cox假设不成立或更关注"时间延长/缩短"效应时采用的参数化方法，本项目中使用对数逻辑分布（Log-Logistic）拟合，以分析各因素如何直接改变客户留存时间。

4. **客户终身价值（CLV）计算：** 将生存概率与月均利润（假设30美元）结合，通过月化IRR进行折现，得到累计净现值（Cumulative NPV），为精准营销提供量化依据。

**数据处理策略：** 使用Spark DataFrame API完成大规模ETL（Extract-Transform-Load），生成Silver层数据；生存建模与可视化则在Pandas + lifelines层面完成，实现"大数据处理 + 精准统计建模"的高效结合。

## 2. 实验环境初始化

- 创建SparkSession（appName: Telco_Customer_Survival_Analysis）。
- 安装并导入生存分析核心库（lifelines）、可视化库（matplotlib、seaborn）以及数据处理库（pandas、numpy）。
- 配置全局绘图参数，屏蔽不必要的警告。

## 3. 数据加载与清洗（Silver层）

**数据来源：** 直接从IBM官方GitHub链接加载Telco-Customer-Churn.csv（7043条原始记录，21个字段）。

**清洗逻辑（使用pandas链式操作）：**

1. 将Churn列（Yes/No）转换为数值型（0/1）作为事件指示变量。
2. 严格筛选：仅保留月付合同（Contract == 'Month-to-month'）且有互联网服务（InternetService != 'No'）的客户。

清洗后得到3351条高质量Silver层数据，用于后续建模。清洗后数据的前5行关键字段已在代码输出中预览。

<img src="images/Q2.1.png" alt="图1：数据清洗后前5行数据" style="max-width:100%; display:block; margin:20px auto;">
<p class="figure-caption">图1：数据清洗后前5行数据</p>

## 4. Kaplan-Meier 生存分析

代码使用 lifelines.KaplanMeierFitter 拟合 KM 模型，并绘制整体生存曲线。

<img src="images/Q2.2.png" alt="图 2: Kaplan-Meier Survival Curve: Population Level" style="max-width:100%; display:block; margin:20px auto;">
<p class="figure-caption">图 2: Kaplan-Meier Survival Curve: Population Level</p>

**核心结果：**

- 横轴：合同月数（tenure）；纵轴：生存概率（客户仍未流失的概率）。
- 趋势分析：前 6 个月流失速率最快（曲线陡峭下降），之后曲线逐渐平缓，符合电信行业"早期高流失、后期趋稳"的经典模式。
- 中位生存时间（Median Survival Time）：34.0 个月。这意味着在月付且使用互联网服务的客户群中，有一半的客户会在 34 个月内流失。

**业务洞见：** 前 6 个月是客户挽留的关键窗口，企业应在此阶段加大优惠或服务干预力度。

### 4.1 维度细分下的生存差异分析 (Log-rank Test)

为了识别不同服务特征对流失的影响，代码进一步按 OnlineSecurity（是否开通网络安全服务）分组绘制 KM 曲线，并使用 Log-rank 检验判断组间差异的统计显著性。在线安全服务的 KM 曲线清楚地显示，开通了网络安全服务（Yes）的客户生存概率显著高于未开通（No）的客户。

<img src="images/q2.3.png" alt="图 3: Kaplan-Meier by OnlineSecurity" style="max-width:100%; display:block; margin:20px auto;">
<p class="figure-caption">图 3: Kaplan-Meier by OnlineSecurity</p>

**Log-rank检验结果：**

<table>
    <tr><th>组别</th><th>test_statistic</th><th>p-value</th></tr>
    <tr><td>OnlineSecurity (No vs. Yes)</td><td>141.60</td><td>1.19×10⁻³²</td></tr>
</table>
<p><strong>表1：OnlineSecurity的Log-rank检验结果</strong></p>

p值远小于0.001，说明是否开通在线安全服务对客户流失时间存在极显著的差异。该结论为后续将OnlineSecurity纳入Cox模型提供了统计依据。

### 4.2 DSL客户生存概率预测

代码还提取了DSL互联网服务子集的生存概率矩阵。前10个月的预测值如表2所示。

<table>
    <tr><th>月份</th><th>生存概率</th></tr>
    <tr><td>0</td><td>1.000000</td></tr>
    <tr><td>1</td><td>0.902698</td></tr>
    <tr><td>2</td><td>0.864380</td></tr>
    <tr><td>3</td><td>0.834702</td></tr>
    <tr><td>4</td><td>0.810522</td></tr>
    <tr><td>5</td><td>0.794352</td></tr>
    <tr><td>6</td><td>0.783900</td></tr>
    <tr><td>7</td><td>0.776362</td></tr>
    <tr><td>8</td><td>0.768486</td></tr>
    <tr><td>9</td><td>0.750833</td></tr>
</table>
<p><strong>表2：DSL客户前10个月生存概率</strong></p>

该表格精确反映了DSL用户的逐月留存衰减，可用于客户关怀节奏的制定。

## 5. Cox比例风险模型与AFT模型

### 5.1 Cox模型建模与特征工程

代码进行了One-hot编码处理分类变量，最终选择以下协变量进入Cox比例风险模型：Dependents_Yes（是否有家属）、InternetService_DSL（是否DSL用户）、OnlineBackup_Yes（是否开通在线备份）、TechSupport_Yes（是否开通技术支持）。模型拟合完成后，打印了详细摘要。

### 5.2 Cox模型结果解读

<table>
    <tr><th>指标</th><th>值</th></tr>
    <tr><td>样本量（observations）</td><td>3351</td></tr>
    <tr><td>事件数（events observed）</td><td>1556</td></tr>
    <tr><td>偏对数似然（partial log-likelihood）</td><td>-11315.95</td></tr>
    <tr><td>一致性指数（Concordance）</td><td>0.64</td></tr>
    <tr><td>对数似然比检验（-log2(p) of ll-ratio test）</td><td>236.24</td></tr>
</table>
<p><strong>表3：Cox模型基本信息</strong></p>

<table>
    <tr><th>协变量</th><th>coef</th><th>exp(coef)</th><th>p-value</th><th>-log2(p)</th><th>风险比解读</th></tr>
    <tr><td>Dependents_Yes</td><td>-0.33</td><td>0.72</td><td>&lt;0.005</td><td>18.12</td><td>有家属的客户风险降低 28%</td></tr>
    <tr><td>InternetService_DSL</td><td>-0.22</td><td>0.80</td><td>&lt;0.005</td><td>12.07</td><td>DSL 用户风险降低 20%</td></tr>
    <tr><td>OnlineBackup_Yes</td><td>-0.78</td><td>0.46</td><td>&lt;0.005</td><td>128.37</td><td>开通在线备份的客户风险降低 54%</td></tr>
    <tr><td>TechSupport_Yes</td><td>-0.64</td><td>0.53</td><td>&lt;0.005</td><td>55.36</td><td>开通技术支持的客户风险降低 47%</td></tr>
</table>
<p><strong>表4：Cox模型系数估计</strong></p>

所有变量均显著 (p &lt; 0.005)。其中OnlineBackup的保护效应最强，HR为0.46；TechSupport次之，HR为0.53。该结果表明：提供附加服务和家庭关怀可大幅降低流失风险。

<img src="images/Q2.4.png" alt="图4：CoxPH Model - Hazard Ratios (95% CI)" style="max-width:100%; display:block; margin:20px auto;">
<p class="figure-caption">图4：CoxPH Model - Hazard Ratios (95% CI)</p>

Cox 模型的风险比图 (Hazard Ratios with 95% CI) 直观展示了各变量的效应方向和置信区间，所有变量的风险比均小于 1，属于保护因素。

### 5.3 比例风险假设检验 (PH Assumption Test)

代码对 Cox 模型进行了比例风险假设 (PH assumption) 的 Schoenfeld 残差检验。检验结果见表 5。

<table>
    <tr><th>变量</th><th>km (p-value)</th><th>rank (p-value)</th><th>PH 假设是否满足？</th></tr>
    <tr><td>Dependents_Yes</td><td>0.2232</td><td>0.3680</td><td>满足</td></tr>
    <tr><td>InternetService_DSL</td><td>&lt;0.005</td><td>&lt;0.005</td><td>违反</td></tr>
    <tr><td>OnlineBackup_Yes</td><td>&lt;0.005</td><td>&lt;0.005</td><td>违反</td></tr>
    <tr><td>TechSupport_Yes</td><td>0.0044</td><td>0.0002</td><td>违反 (p &lt; 0.05)</td></tr>
</table>
<p><strong>表5：比例风险假设检验结果</strong></p>

检验输出的图示（含 rank-transformed time 和 km-transformed time 的平滑曲线）也提示部分变量的回归系数随时间不是完全恒定的。对于 InternetService、OnlineBackup 和 TechSupport 三个变量，模型建议采用分层（stratification）或引入时变协变量等方法进行修正。这也是引入 AFT 模型的重要原因之一。

<img src="images/Q2.5.png" alt="图5: Scaled Schoenfeld residuals of 'Dependents_Yes'" style="max-width:100%; display:block; margin:20px auto;">
<p class="figure-caption">图5: Scaled Schoenfeld residuals of 'Dependents_Yes'</p>

<img src="images/Q2.6.png" alt="图6: Scaled Schoenfeld residuals of 'InternetService_DSL'" style="max-width:100%; display:block; margin:20px auto;">
<p class="figure-caption">图6: Scaled Schoenfeld residuals of 'InternetService_DSL'</p>

<img src="images/Q2.7.png" alt="图7: Scaled Schoenfeld residuals of 'OnlineBackup_Yes'" style="max-width:100%; display:block; margin:20px auto;">
<p class="figure-caption">图7: Scaled Schoenfeld residuals of 'OnlineBackup_Yes'</p>

<img src="images/Q2.8.png" alt="图8: Scaled Schoenfeld residuals of 'TechSupport_Yes'" style="max-width:100%; display:block; margin:20px auto;">
<p class="figure-caption">图8: Scaled Schoenfeld residuals of 'TechSupport_Yes'</p>

### 5.4 AFT（加速失效时间）模型

当Cox模型的PH假设被违反，且更关心"时间延长/缩短"效应时，可使用参数化AFT模型。本项目采用Log-Logistic AFT模型，并引入了更多维度的支付与服务信息。模型拟合结果摘要见表6。

<table>
    <tr><th>参数</th><th>协变量</th><th>coef</th><th>exp(coef)</th><th>p-value</th><th>解读</th></tr>
    <tr><td>alpha_</td><td>Intercept</td><td>1.59</td><td>4.91</td><td>&lt;0.005</td><td>基准时间尺度</td></tr>
    <tr><td>alpha_</td><td>OnlineBackup_Yes</td><td>0.81</td><td>2.25</td><td>&lt;0.005</td><td>开通在线备份使生存时间延长约2.25倍</td></tr>
    <tr><td>alpha_</td><td>TechSupport_Yes</td><td>0.69</td><td>1.99</td><td>&lt;0.005</td><td>开通技术支持使生存时间延长约2倍</td></tr>
    <tr><td>alpha_</td><td>OnlineSecurity_Yes</td><td>0.86</td><td>2.37</td><td>&lt;0.005</td><td>开通在线安全服务时间延长约2.4倍</td></tr>
    <tr><td>alpha_</td><td>MultipleLines_Yes</td><td>0.66</td><td>1.94</td><td>&lt;0.005</td><td>开通多线路服务使时间延长约1.9倍</td></tr>
    <tr><td>alpha_</td><td>Partner_Yes</td><td>0.68</td><td>1.97</td><td>&lt;0.005</td><td>有伴侣的客户时间延长约2倍</td></tr>
    <tr><td>alpha_</td><td>DeviceProtection_Yes</td><td>0.48</td><td>1.62</td><td>&lt;0.005</td><td>设备保护延长约1.6倍</td></tr>
    <tr><td>alpha_</td><td>InternetService_DSL</td><td>0.38</td><td>1.47</td><td>&lt;0.005</td><td>DSL用户时间延长约1.5倍</td></tr>
    <tr><td>alpha_</td><td>PaymentMethod_Bank transfer</td><td>0.74</td><td>2.10</td><td>&lt;0.005</td><td>采用银行自动转账时间延长</td></tr>
    <tr><td>alpha_</td><td>PaymentMethod_Credit card</td><td>0.80</td><td>2.22</td><td>&lt;0.005</td><td>采用信用卡自动支付时间延长</td></tr>
    <tr><td>beta_</td><td>Intercept</td><td>0.12</td><td>1.13</td><td>&lt;0.005</td><td>对数逻辑分布的尺度参数</td></tr>
</table>
<p><strong>表6：Log-Logistic AFT模型拟合结果（部分变量）</strong></p>

**模型性能：** Concordance = 0.73，高于Cox模型的0.64，说明该AFT模型具有更好的预测区分能力。对数似然比检验的 -log2(p) = 605.78，模型整体高度显著。

AFT模型的结果进一步印证了附加服务（尤其是OnlineSecurity、OnlineBackup、TechSupport）对延长客户留存时间的显著正向作用。同时，对数-逻辑分布假设的检查图（Log-Logistic Check）显示数据的log-failure odds与log-time大致呈线性关系，说明分布假设基本合理。

<img src="images/Q2.10.png" alt="图9：Log-Logistic Check" style="max-width:100%; display:block; margin:20px auto;">
<p class="figure-caption">图9：Log-Logistic Check</p>

### 5.5 客户终身价值（CLV）交互式仪表盘

为了将生存分析成果转化为商业决策，代码构建了一个基于ipywidgets的交互式仪表盘。用户可实时选择客户画像并输入年化IRR（默认10%），仪表盘将：

- 预测该客户群体未来36个月的个性化生存曲线。
- 按CLV公式计算各月净现值（NPV）：

$$\mathrm{Expected~Monthly~Profit}_{t} = S(t)\times 30$$

$$\mathrm{NPV}_{t} = \frac{\mathrm{Expected~Monthly~Profit}_{t}}{(1 + \frac{\mathrm{IRR}}{12})^{t}}$$

$$\mathrm{Cumulative~NPV} = \sum_{t = 1}^{36}\mathrm{NPV}_{t}$$

- 以柱状图（Cumulative NPV）和折线图（Predicted Retention Curve）展示。默认画像，全部特征 = 0，Annual IRR = 0.10 的前12个月计算结果见表7。

<img src="images/Q2.11.png" alt="图10：Cumulative NPV & Predicted Retention Curve" style="max-width:100%; display:block; margin:20px auto;">
<p class="figure-caption">图10：Cumulative NPV & Predicted Retention Curve</p>

<table>
    <tr><th>Contract Month</th><th>Survival Prob.</th><th>Avg Expected Monthly Profit</th><th>NPV</th><th>Cumulative NPV</th></tr>
    <tr><td>1</td><td>0.865866</td><td>25.975983</td><td>25.761306</td><td>25.761306</td></tr>
    <tr><td>2</td><td>0.813649</td><td>24.409476</td><td>24.007681</td><td>49.768987</td></tr>
    <tr><td>3</td><td>0.773411</td><td>23.202339</td><td>22.631816</td><td>72.400802</td></tr>
    <tr><td>4</td><td>0.736666</td><td>22.099971</td><td>21.378400</td><td>93.779203</td></tr>
    <tr><td>5</td><td>0.708739</td><td>21.262185</td><td>20.397985</td><td>114.177188</td></tr>
    <tr><td>6</td><td>0.690009</td><td>20.700285</td><td>19.694800</td><td>133.871987</td></tr>
    <tr><td>7</td><td>0.667327</td><td>20.019802</td><td>18.889954</td><td>152.761942</td></tr>
    <tr><td>8</td><td>0.648066</td><td>19.441970</td><td>18.193123</td><td>170.955065</td></tr>
    <tr><td>9</td><td>0.626485</td><td>18.794551</td><td>17.441942</td><td>188.397007</td></tr>
    <tr><td>10</td><td>0.603472</td><td>18.104167</td><td>16.662390</td><td>205.059398</td></tr>
    <tr><td>11</td><td>0.589942</td><td>17.698259</td><td>16.154190</td><td>221.213588</td></tr>
    <tr><td>12</td><td>0.572987</td><td>17.189616</td><td>15.560254</td><td>236.773842</td></tr>
</table>
<p><strong>表7：默认画像下前12个月的CLV计算结果</strong></p>

12个月的累计NPV达236.77美元；随着时间延长，36个月累计NPV更高。右侧折线图清晰反映该客户群体的逐月留存衰减，与整体KM曲线趋势一致，但受协变量影响更精准。

## 6. 完整生存分析流程总结

1. **数据准备阶段（Spark ETL）：** 清洗得到3351条高质量数据，聚焦月付且有互联网服务的客户。
2. **探索性分析阶段（KM模型）：** 揭示整体留存规律，中位生存时间34个月，且前6个月为高流失期。
3. **分组比较与假设检验：** 通过Log-rank检验确认不同服务维度（如OnlineSecurity）对留存时间的显著影响。
4. **多变量建模阶段（Cox PH + AFT）：** Cox模型识别出OnlineBackup、TechSupport等变量是强保护因素，但发现部分变量违反比例风险假设。AFT模型进一步量化了各因素对生存时间的"延长"效应，模型区分能力（Concordance 0.73）优于Cox模型。
5. **业务应用与价值评估（CLV Dashboard）：** 将生存预测与财务指标结合，计算个性化累计NPV，企业可根据Dashboard结果对高风险客群采取针对性干预，最大化终身价值。
