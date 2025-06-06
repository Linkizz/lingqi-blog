---
title: 强化学习（5）：蒙特卡洛方法
author: Lingqi Zeng
date: '2025-04-21'
slug: rl-5-monte-carlo-learning
categories:
  - Reinforcement Learning
tags: []
---

## Model-Based to Model-Free

上一节介绍的[策略迭代算法](/blog/rl-4-value-iteration-and-policy-iteration/)是基于明确的环境模型（model-based）来进行策略评估和改进的。然而在许多现实问题中，模型 \(p(s'|s,a), p(r|s,a)\) 不能够轻易得到，那么可以使用基于蒙特卡洛（Monte Carlo，MC）的无模型（model-free）方法进行策略的评估和改进。

回顾策略迭代算法的两个步骤：

- Policy evaluation：

$$\mathbf{v}_{\pi_{k}}=\mathbf{r}_{\pi_{k}}+\gamma \mathbf{P}_{\pi_{k}}\mathbf{v}_k.$$

- Policy improvement： 

$$\pi_{k+1}=\arg\max_{\pi}(\mathbf{r}_{\pi}+\gamma \mathbf{P}_{\pi}\mathbf{v}_{\pi_{k}}).$$

其中 policy improvement 的 elementwise form 为

$$\begin{aligned}
\pi_{k+1}(s) &= \arg\max_{\pi} \sum_{a \in \mathcal{A}(s)} \pi(a|s) \left( \sum_{r \in \mathcal{R}(s,a)} p(r|s,a)r + \gamma \sum_{s' \in \mathcal{S}} p(s'|s,a)v_{\pi_k}(s') \right) \\
&= \arg\max_{\pi} \sum_{a \in \mathcal{A}(s)} \pi(a|s) q_{\pi_k}(s,a), \quad \forall s \in \mathcal{S}.
\end{aligned}$$

策略迭代算法的第一步 policy evaluation 计算 state value 的目的是用于第二步计算 action value \(q_{\pi_k}(s,a)\)，因此这里最关键的部分就是求解 \(q_{\pi_k}(s,a)\)，有两种方法：

- Model-based：

$$q_{\pi_k}(s,a)=\sum_{r \in \mathcal{R}(s,a)} p(r|s,a)r + \gamma \sum_{s' \in \mathcal{S}} p(s'|s,a)v_{\pi_k}(s').$$

- Model-free：

$$q_{\pi_k}(s,a)=\mathbb{E}[G_t|S_t=s,A_t=a].$$

使用 model-free 方法时，从任意 state \(s\) 出发，采取 action \(a\)，根据 policy \(\pi_k\)，采样多个 episodes，然后使用这些 episodes 的 return 均值来估计 \(q_{\pi_k}(s,a)\)：

$$q_{\pi_k}(s,a)=\mathbb{E}[G_t|S_t=s,A_t=a] \approx \frac{1}{N} \sum_{i=1}^N g^{(i)}(s,a).$$

其中，\(g^{(i)}(s,a)\) 表示第 \(i\) 个 episode 的 return。

总的来说，**没有模型就得有数据**。

## MC Basic

根据刚刚介绍的基于蒙特卡洛的 model-free 方法，我们可以得到 MC Basic 算法的步骤如下：

- Policy evaluation：对于每一个 state-action pair \((s,a)\)，基于给定的 policy \(\pi\)，生成多个 episodes。进而用所有 episodes 的平均 return 来估计 \(q_{\pi_k}(s,a)\)。

- Policy improvement：这一步求解 \(\pi_{k+1}(s) = \arg\max_{\pi} \sum_{a \in \mathcal{A}(s)} \pi(a|s) q_{\pi_k}(s,a), \forall s \in \mathcal{S}\)，最优 policy 为 \(\pi_{k+1}(a_k^*|s)=1,a_k^*=\arg\max_a q_{\pi_k}(s,a)\)。

可以看到，MC Basic 根据蒙特卡洛方法直接计算 \(q_{\pi_k}(s,a)\)，而非通过计算 state value\(v_{\pi_k}(s)\) 后再计算 \(q_{\pi_k}(s,a)\)，其余过程 MC Basic 和策略迭代算法相同。

综上，MC Basic 算法的伪代码如下：

![MC Basic](images/MC_Basic.png)

一些有趣的现象：

- Episode 如果太短，那么模型探索的就不够，只有接近 target 的 state 才有非零的 state value。

- 理论上 Episode 越长，那么 \(G_t\) 的估计就越准确，但一般实际情况我们取一个充分长的长度即可。

## MC Exploring Starts

MC Basic 算法考虑的信息较少，且需要采样大量 episodes，因此效率太低。举个简单例子，考虑如下 episode：

$$s_1 \xrightarrow {a_2} s_2 \xrightarrow {a_4} s_1 \xrightarrow {a_2} s_2 \xrightarrow {a_3} s_5 \xrightarrow {a_1} \cdots$$

MC Basic 算法的信息仅用来计算 \(q_{\pi}(s_1,a_2)\)，但实际上这个 episode 也访问到很多其他的 state-action pairs（\(q_{\pi}(s_2,a_4),q_{\pi}(s_2,a_3),q_{\pi}(s_5,a_1)\)）：

$$\begin{align*}
s_1 \xrightarrow{a_2} s_2 \xrightarrow{a_4} s_1 \xrightarrow{a_2} s_2 \xrightarrow{a_3} &s_5 \xrightarrow{a_1} \cdots \quad \text{[original episode]} \\
s_2 \xrightarrow{a_4} s_1 \xrightarrow{a_2} s_2 \xrightarrow{a_3} &s_5 \xrightarrow{a_1} \cdots \quad \text{[episode starting from $(s_2, a_4)$]} \\
s_1 \xrightarrow{a_2} s_2 \xrightarrow{a_3} &s_5 \xrightarrow{a_1} \cdots \quad \text{[episode starting from $(s_1, a_2)$]} \\
s_2 \xrightarrow{a_3} &s_5 \xrightarrow{a_1} \cdots \quad \text{[episode starting from $(s_2, a_3)$]} \\
&s_5 \xrightarrow{a_1} \cdots \quad \text{[episode starting from $(s_5, a_1)$]}
\end{align*}$$

因此一个 episode 中的所有 state-action 都可以作为起点而被充分利用，这就是 Exploring Starts 。其中包括两种方法：

- First-visit：对于 episode 中的某个 state-value pair \((s_t,a_t)\)，只考虑首次访问时的 return。

- Every-visit：对于 episode 中的某个 state-value pair \((s_t,a_t)\)，考虑每次访问的累计 return 的均值。

除了上述让数据被更加高效利用之外，还可以更加高效地更新 policy。MC Basic 算法需要 agent 将所有的 episodes 都收集后再计算更新，我们可以利用单个 episode 去估计 action value，这样就可以更新 policy episode-by-episode。这个思想在学习[截断策略迭代算法](/blog/rl-4-value-iteration-and-policy-iteration/)时介绍过，是一种用不精确的中间结果估计来提高算法效率的方法，可以统称为 Generalized Policy Iteration（GPI）。

基于 MC Basic 算法，利用上述的高效利用数据和高效更新 policy 的方法，我们得到 MC Exploring Starts 算法，它的伪代码如下：

![MC Exploring Starts](images/MC_Exploring_Starts.png)

总的来说，exploring starts 表示我们需要为每一个 state-action pair 生成足够多的 episodes，换句话说，每一个 state-value pair 被深度探索后，我们才能够精确地估计 action value，进而找到最优 policy。

## MC \(\varepsilon\)-Greedy

在实际应用中，经常难以收集到每一个 state-value pair 的 episode，因为某些 policy 可能永远不会访问到某些 state 或采取某些 action，我们可以引入 soft policy 来去除 exploring starts 的条件。

Soft policy 是指每一个 state 都有概率采取所有不同的 action，依赖于随机性，只要 episode 足够长，那么所有的 \((s,a)\) 都会被访问到很多次，进而 exploring starts 的条件就可以去除。

常用的 soft policy 是 \(\varepsilon\)-greedy policy：

$$\pi(a|s) = 
\begin{cases} 
1 - \dfrac{\varepsilon}{|\mathcal{A}(s)|}(|\mathcal{A}(s)| - 1), & \text{for the greedy action,} \\
\dfrac{\varepsilon}{|\mathcal{A}(s)|}, & \text{for the other } |\mathcal{A}(s)| - 1 \text{ actions.}
\end{cases}$$

其中，\(\varepsilon \in [0,1]\)，\(|\mathcal{A}(s)|\) 是 state \(s\) 可以采取的 action 的数量。

可以看到，greedy action 的概率一定大于其他 action：

$$1 - \frac{\varepsilon}{|\mathcal{A}(s)|}(|\mathcal{A}(s)| - 1) = 1 - \varepsilon + \frac{\varepsilon}{|\mathcal{A}(s)|} \geq \frac{\varepsilon}{|\mathcal{A}(s)|}.$$

也就是说，采取最优 action 的概率依然是的最大的，不过其他 action 都有采取的可能。\(\varepsilon\) 越小就越贪心，即最优 action 概率相比其他 action 大很多；\(\varepsilon\) 越大就越倾向于探索，即各个 action 的概率趋向于均匀分布。

把 \(\varepsilon\)-greedy policy 嵌入到 MC-based 的强化学习算法，具体地，将 policy improvement 改为求解

$$\pi_{k+1}(s)=\arg\max_{\pi \in \Pi_{\varepsilon}} \sum_{a \in \mathcal{A}(s)} \pi(a|s) q_{\pi_k}(s,a),$$

其中 \(\Pi_{\varepsilon}\) 表示所有的 \(\varepsilon\)-greedy policy。最优 policy 为

$$\pi_{k+1}(a|s) = 
\begin{cases} 
1 - \dfrac{\varepsilon}{|\mathcal{A}(s)|}(|\mathcal{A}(s)| - 1), & a=a_k^* \\
\dfrac{\varepsilon}{|\mathcal{A}(s)|}, & a \neq a_k^*.
\end{cases}$$

MC \(\varepsilon\)-Greedy 除了使用 \(\varepsilon\)-greedy policy 进行 policy improvement 之外，其余步骤与 MC Exploring Starts 完全相同。它不需要 exploring starts，但是以另一种方式访问了所有的 state-action pairs。

![MC epsilon Greedy](images/MC_epsilon_Greedy.png)

MC \(\varepsilon\)-Greedy 算法的优势是探索性很强，所有不需要 exploring starts 这个条件。它的劣势是仅仅在 \(\Pi_{\varepsilon}\) 这个 policy 集合中是最优的，但在 \(\Pi\) 中并不是最优的。当 \(\varepsilon\) 非常小，那么 \(\Pi_{\epsilon}\) 中的最优 policy 与 \(\Pi\) 的最优 policy 就非常接近。一个常用的方法是先设置较大的 \(\varepsilon\)，让它有较强的探索能力，随着 policy 的更新，逐步减小 \(\varepsilon\)，让其趋于 0，以得到一个最优的策略。






