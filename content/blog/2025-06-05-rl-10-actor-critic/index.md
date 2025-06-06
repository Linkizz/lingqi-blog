---
title: 强化学习（10）：Actor-Critic 方法
author: Lingqi Zeng
date: '2025-06-05'
slug: rl-10-actor-critic
categories:
  - Reinforcement Learning
tags: []
---

## Introduction

Actor-Critic 方法将值函数近似和策略梯度相结合的方法，它由两个部分组成：

- Actor：负责 policy update。

- Critic：负责 policy evaluation。

这两个部分相互合作，Actor 采取 action 与环境进行交互，Critic 评估 Actor 的表现，指导 Actor 的下一个 action。

## Q Actor-Critic

在策略梯度算法中，我们使用随机梯度上升最大化目标函数 \(J(\theta)\)：

$$\theta_{t+1} = \theta_t + \alpha \nabla_\theta \ln \pi(a_t | s_t, \theta_t) q_t(s_t, a_t).$$

该随机梯度上升公式就是 Actor，负责 policy update；估计 \(q_t(s_t, a_t)\) 的算法就是 Critic，负责 policy evaluation。

上一节课介绍的 REINFORCE 算法使用 MC learning 来估计 \(q_t(s_t, a_t)\)，