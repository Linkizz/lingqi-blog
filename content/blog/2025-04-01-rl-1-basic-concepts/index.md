---
title: 强化学习（1）：基础概念
author: Lingqi Zeng
date: '2025-04-01'
slug: rl-1-basic-concepts
categories:
  - Reinforcement Learning
tags: []
---

## Introduction

本强化学习系列（1-10）笔记参考西湖大学赵世钰老师的强化学习课程，[github 链接](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning)。

## Grid World

课程始终以 grid world 作为例子介绍强化学习，具体而言，一个机器人（agent）在网格世界中自由地移动，目的是从一个给定的初始位置，找到一条“最好”的路径达到目标位置（蓝色），中途会遇到一些障碍物（黄色）。

![grid world](images/grid_world.png)

那么如何定义路径的好坏呢？一条好的路径应该避免撞墙，碰到障碍物，以及不要重复之前走过的路。如果 agent 知道网格的地图，那么它找到最优路径是非常容易的，但实际情况往往相反，agent 难以直接对周围环境的全貌有清晰的认识。因此 agent 必须与环境进行交互，通过试验不断地感知和了解环境，进而做出较好的判断。为此，我们介绍 agent 与环境交互的一些概念。

## State and Action

**State：** 表示 agent 在环境中的状态，在 grid world 中指的是它的位置，记为 \(s_i\)。

**State Space：** 所有的状态集合 \(\mathcal{S}=\{s_i\}_{i=1}^{9}\)。

![grid world state](images/grid_world_state.png)

**Action：** 表示 agent 在某个状态下采取的行动，记为 \(a_i\)。

- \(a_1\)：向上
- \(a_2\)：向右
- \(a_3\)：向下
- \(a_4\)：向左
- \(a_5\)：原地不动

![grid world action](images/grid_world_action.png)

**Action Space of a State:** 某个状态下所有可能的行动集合，记为 \(\mathcal{A}(s_i)=\{a_i\}_{i=1}^5\)。

不同的 state 有不同的 action space，例如在 state \(s_1\)，agent 只能向右，向下或者原地不动，那么 \(\mathcal{A}(s_1)=\{a_2,a_3,a_5\}\)。本笔记考虑一般情形，\(\mathcal{A}(s_i)=\{a_i\}_{i=1}^5\)。

## State Transition

当采取某个行动，agent 会从一个状态转移到另一个状态，这称为 **state transition**，它定义了 agent 与环境的交互行为。例如

$$s_1 \stackrel{a_2}{\longrightarrow} s_2, s_1 \xrightarrow{a_1} s_1.$$

我们可以定义很多种 state transition，例如

- 可以进入障碍物区域，但 agent 会得到一些惩罚，那么 \(s_5 \xrightarrow{a_2} s_6\)。

- 无法进入障碍物区域，那么 \(s_5 \xrightarrow{a_2} s_5\)。

我们考虑第一种定义方式，这种方式的应用更加广泛和符合实际。

State transition 这个过程也可以用表格表示，但这种方式只能够表示确定性情况。在数学上可以用概率来表达，例如 \(p(s_2|s_1,a_2)=1,p(s_i|s_1,a_2)=0(\forall i \neq 2)\)。

![state transition](images/state_transition.png)

在大多数时候，agent 在某个 state 采取某个 action 时，有多种可能的结果，因此可以使用条件概率来描述这种随机情形。例如，\(p(s_2|s_1,a_2)=0.9,p(s_4|s_1,a_2)=0.1\)。

## Policy and Reward

**Policy：** 告诉 agent 在某个状态应该采取什么行动，不同的 policy 确定不同的路径。

![policy](images/policy.png)

在数学上，policy 可以用条件概率来表达。对于 state \(s_1\)，一个确定性 policy 可以写成

$$\pi(a_2|s_1)=1,\pi(a_i|s_1)=0(\forall i \neq 2).$$

对于一个随机policy，

$$\pi(a_2|s_1)=0.5,\pi(a_3|s_1)=0.5, 0 \ \text{for others}.$$

![stochastic policy](images/stochastic_policy.png)

**Reward：** 采取一个 action 后得到的一个实数值，一般而言，正数表示采取该行动获得的奖励，负数表示惩罚。

Reward 实际上是人与 agent 交互的手段，用于引导 agent 朝着预期的方向发展。我们可以设计如下 reward：

- 如果 agent 往边界走，\(r_{\text{bound}}=-1\)。
- 如果 agent 走进障碍物区域，\(r_{\text{forbid}}=-1\)。
- 如果 agent 到达目标，\(r_{\text{target}}=1\)。
- 其余情况，\(r=0\)。

通过这种 reward 设计，agent 就会尽量避免走出网格或者走进障碍物区域。

Reward 由 agent 当前的 state 和 action 决定，而不取决于下一步的 state，例如原地不动和撞墙的下一步 state 相同，但 reward 不同。reward 可以用条件概率来表达，对于确定性情况，\(p(r=-1|s_1,a_1)=1,p(r \neq -1|s_1,a_1)=0\)。

## Trajectory, Return and Episode

![trajectory](images/trajectory.png)

**Trajectory：** 是一个state-action-reward chain：

$$s_1 \xrightarrow[r=0]{a_2} s_2 \xrightarrow[r=0]{a_3} s_5 \xrightarrow[r=0]{a_3} s_8 \xrightarrow[r=1]{a_2} s_9, $$

$$s_1 \xrightarrow[r=0]{a_3} s_4 \xrightarrow[r=-1]{a_3} s_7 \xrightarrow[r=0]{a_2} s_8 \xrightarrow[r=+1]{a_2} s_9.$$

**Return：** 对于一个trajectory，return是其所有的reward之和：

$$\text{return}=0+0+0+1=1, $$

$$\text{return}=0-1+0+1=0.$$

不同的 policy 决定不同的 trajectory，根据 trajectory 对应的 return 来判断 policy 的好坏。显然上述第一个 trajectory 更好，因为它没有进入障碍区域，从数学上来解释，它有更大的 return。

一个 trajectory 有时是无穷的，例如当到达目的地后一直原地踏步，那么 \(\text{return} \rightarrow \infty\)，这是不合理的。因此可以引入一个 discount rate \(\gamma \in [0,1)\)，则 discounted return为

$$\begin{aligned}
\text{discounted return}&=0+\gamma 0+\gamma^2 0+\gamma^3 1+\gamma^4 1+\gamma^5 1+\cdots \\
&=\gamma^3(1+\gamma+\gamma^2+\cdots) \\
&=\gamma^3 \frac{1}{1-\gamma}.
\end{aligned}$$

如果 \(\gamma\) 接近于 0，模型更加关注前面的 action；如果 \(\gamma\) 接近于 1，模型则更加关注未来的 action。

**Episode：** Agent 按照某个 policy 与环境进行交互，最终停在某个 terminal state，这个
有限长度的 trajectory 称为 episode，这样的任务称为 episodic task。如果不存在 terminal state，则称为 continuing task。

我们可以通过一种统一的方式来处理这两种任务：

- 将 terminal state 看作一个 absorbing state，当 agent 到达时，使其原地踏步，并设置 reward 为 0。

- 将 terminal state 看作一个普通的 state，当 agent 到达时，获得 reward 为 1，并且仍有可能跳出 terminal state，以避免局部最优解。本课程使用这种方式。

## Markov Decision Process （MDP）

马尔可夫决策过程（Markov Decision Process，MDP）是一个描述随机动力系统的通用框架，agent 在与环境交互时也具有随机性，因此强化学习也可以使用 MDP 框架，它的主要成分有：

- **集合（Sets）：**
  - **State Space：** 所有 state 的集合，记为 \(\mathcal{S}\)。
  - **Action Space：** 某个 state 的 action 集合，记为 \(\mathcal{A}(s),s \in \mathcal{S}\)。
  - **Reward Set：** 某个 state 执行某个 action 的 reward，记为 \(\mathcal{R}(s,a),s \in \mathcal{S}, a \in \mathcal{A}(s)\)。
- **模型（Model）：**
  - **State Transition Probability：** 在 state \(s\) 采取 action \(a\) 到达 state \(s'\) 的概率为 \(p(s'|s,a)\)，需要满足 \(\sum_{s' \in \mathcal{S}} p(s'|s,a)=1, \forall (s,a)\)。
  - **Reward Probability：** 在state \(s\)采取 action \(a\)获得 reward \(r\) 的概率为 \(p(r|s,a)\)，需要满足 \(\sum_{r \in \mathcal{R}(s,a)} p(r|s,a)=1, \forall (s,a)\)。
- **策略（policy）：** 在 state \(s\)采取 action \(a\)的概率为 \(\pi(a|s)\)，需要满足 \(\sum_{a \in \mathcal{A}(s)} \pi(a|s)=1, \forall s \in \mathcal{S}\)。
- **马尔可夫性（Markov Property）：** 下一步的 state 和 reward 只与当前的 state 和 action 有关，即

  $$p(s_{t+1}|a_t,s_t,\cdots,a_0,s_0)=p(s_{t+1}|s_t,a_t),$$

  $$p(r_{t+1}|a_t,s_t,\cdots,a_0,s_0)=p(r_{t+1}|s_t,a_t).$$

当确定一个 policy 后，Markov decision process 就变为 Markov process。MP 中每一个 state 的 action 是确定的，而 MDP 中每一个 state 有多种 action 的可能。

## Summary

强化学习是 agent 与环境不断交互的过程。Agent 是一个可以感知 state、维护 policy 和执行 action 的决策者。Agent 执行 action 会改变 state，同时获得相应的 reward，reward 指导 agent 再执行 action，不断循环这个过程。
