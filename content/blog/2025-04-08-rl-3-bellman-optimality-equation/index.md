---
title: '强化学习(3): 贝尔曼最优公式'
author: Lingqi Zeng
date: '2025-04-08'
slug: rl-3-bellman-optimality-equation
categories:
  - Reinforcement Learning
tags: []
---

## Optimal Policy

State value可以用于评价policy的好坏，如果有

`$$v_{\pi_1}(s) \geq v_{\pi_2}(s), \quad \forall s \in \mathcal{S}.$$`

则称policy `$\pi_1$`好于policy `$\pi_2$`。

如果有一个policy `$\pi^*$`满足

`$$v_{\pi^*}(s) \geq v_{\pi}(s), \quad \forall s \in \mathcal{S}, \forall \pi.$$`

则称policy `$\pi^*$`为optimal policy，`$\pi^*$`的state values称为optimal state values。

由上述两个定义可以引出一系列问题：

- Optimal policy是否存在？
- Optimal policy是否唯一？
- Optimal policy是随机性的还是确定性的？
- 如何得到optimal policy？

下面我们介绍Bellman optimality equation来回答上述问题。

## Bellman Optimality Equation (BOE)

我们先回忆Bellman equation：

`$$\begin{aligned}
\textcolor{blue}{v_{\pi}(s)}&=\mathbb{E}[R_{t+1}|S_t=s]+\gamma \mathbb{E}[G_{t+1}|S_t=s] \\
&=\underbrace{\sum_{a \in \mathcal{A}(s)} \pi(a|s) \sum_{r \in \mathcal{R}(s,a)} p(r|s, a) r}_{\text{mean of immediate rewards}}+\underbrace{\gamma \sum_{a \in \mathcal{A}(s)} \pi(a | s) \sum_{s' \in \mathcal{S}} p(s' | s, a) \textcolor{blue}{v_{\pi}(s')}}_{\text{mean of future rewards}} \\
&=\sum_{a \in \mathcal{A}(s)} \pi(a|s) \left[ \sum_{r \in \mathcal{R}(s,a)} p(r|s,a)r + \gamma \sum_{s' \in \mathcal{S}} p(s'|s,a)\textcolor{blue}{v_{\pi}(s')} \right], \forall s \in \mathcal{S}.
\end{aligned}$$`

Bellman optimality equation则是对Bellman euqation取optimal policy `$\pi^*$`：

`$$\begin{aligned}
v(s) &= \textcolor{blue}{\max_{\pi}} \sum_{a \in \mathcal{A}(s)} \textcolor{blue}{\pi(a|s)} \left( \sum_{r \in \mathcal{R}(s,a)} p(r|s,a)r + \gamma \sum_{s' \in \mathcal{S}} p(s'|s,a)v(s') \right) \\
&= \textcolor{blue}{\max_{\pi}} \sum_{a \in \mathcal{A}(s)} \textcolor{blue}{\pi(a|s)} q(s,a) , \quad \forall s \in \mathcal{S}.
\end{aligned}$$`

- `$p(r|s,a),p(s'|s,a)$`是已知的系统的模型。
- `$v(s),v(s')$`是要求解的未知量。
- Bellman equation是依赖于一个给定的policy `$\pi$`，而BOE则是要求最优的policy。

上式是elementwise form，将其写成简洁的matrix-vector form：

`$$\mathbf{v} = \max_{\pi} (\mathbf{r}_{\pi} + \gamma \mathbf{P}_{\pi} \mathbf{v}).$$`

由此会产生许多问题：

- 该方程是否有解？
- 如何求解这个方程？
- 方程的解是否唯一？
- 方程的解和optimal policy有什么关系？

## Maximization on the Right-Hand Side of BOE

回顾BOE的两种形式，elementwise form：

`$$\begin{aligned}
v(s) &= \textcolor{blue}{\max_{\pi}} \sum_{a \in \mathcal{A}(s)} \textcolor{blue}{\pi(a|s)} \left( \sum_{r \in \mathcal{R}(s,a)} p(r|s,a)r + \gamma \sum_{s' \in \mathcal{S}} p(s'|s,a)v(s') \right) \\
&= \textcolor{blue}{\max_{\pi}} \sum_{a \in \mathcal{A}(s)} \textcolor{blue}{\pi(a|s)} q(s,a) , \quad \forall s \in \mathcal{S}.
\end{aligned}$$`

Matrix-vector form：

`$$\mathbf{v} = \max_{\pi} (\mathbf{r}_{\pi} + \gamma \mathbf{P}_{\pi} \mathbf{v}).$$`

BOE一个式子有`$\mathbf{v}$`和`$\pi$`两个未知量，为了求解这个式子，可以先固定住`$\mathbf{v}$`，然后求解使得式子取最大值的`$\pi$`，记为`$\pi^*$`。进而得到`$\mathbf{v} = \mathbf{r}_{\pi^*} + \gamma \mathbf{P}_{\pi^*} \mathbf{v}$`，求解该方程就可以得到`$\mathbf{v}$`。

固定`$v(s')$`，那么`$q(s,a)$`就是已知的。由于`$\sum_{a \in \mathcal{A}(s)} \pi(a|s)=1$`，我们有

`$$\max_{\pi} \sum_{a \in \mathcal{A}(s)} \textcolor{blue}{\pi(a|s)} q(s,a)= \max_{a \in \mathcal{A}(s)} q(s,a),$$`

其中，

`$$\pi(a|s) = \left\{
\begin{array}{ll}
1 & a = a^* \\
0 & a \neq a^*
\end{array}
\right., \quad a^* = \arg\max_a q(s, a).$$`

上述求解的思想是，将最大的`$q(s,a)$`的权重`$\pi(a|s)$`设置为1，其余的设置为0。

## Solving the BOE

令`$f(\mathbf{v})=\max_{\pi} (\mathbf{r}_{\pi} + \gamma \mathbf{P}_{\pi} \mathbf{v})$`，那么

`$$\mathbf{v}=f(\mathbf{v}), \quad [f(\mathbf{v})]_s=\max_{\pi} \sum_{a \in \mathcal{A}(s)} \pi(a|s) q(s,a) , \quad \forall s \in \mathcal{S}.$$`

根据压缩映射原理，对于形如`$x=f(x)$`的方程，如果`$f$`是压缩映射，我们有

- 存在唯一的不动点`$x^*$`使得`$x^*=f(x^*)$`。
- 考虑一个序列`$\{x_k\}$`，其中`$x_{k+1}=f(x_k)$`，那么`$x_k \rightarrow x^*, k \rightarrow \infty$`，且以指数级收敛。

事实上，[BOE中的映射`$f$`就是一个压缩映射](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning/blob/main/3%20-%20Chapter%203%20Optimal%20State%20Values%20and%20Bellman%20Optimality%20Equation.pdf)，`$\forall v_1, v_2 \in \mathbb{R}^{|\mathcal{S}|}$`，有

`$$\|f(\mathbf{v_1}) - f(\mathbf{v_2})\|_{\infty} \leq \gamma \|\mathbf{v_1} - \mathbf{v_2}\|_{\infty},$$`

其中，`$\gamma \in (0,1)$`是discount rate。

对BOE`$f(\mathbf{v})=\max_{\pi} (\mathbf{r}_{\pi} + \gamma \mathbf{P}_{\pi} \mathbf{v})$`使用压缩映射原理，存在唯一解`$\mathbf{v}^*$`，其可以通过如下方式迭代求解：

`$$\mathbf{v}_{k+1}=f(\mathbf{v}_k)=\max_{\pi} (\mathbf{r}_{\pi} + \gamma \mathbf{P}_{\pi} \mathbf{v}_k).$$`

Elmentwise form为：

`$$\begin{aligned}
\textcolor{red}{v_{k+1}(s)} &= \max_{\pi} \sum_{a \in \mathcal{A}(s)} \textcolor{blue}{\pi(a|s)} \left( \sum_{r \in \mathcal{R}(s,a)} p(r|s,a)r + \gamma \sum_{s' \in \mathcal{S}} p(s'|s,a) \textcolor{red}{v_{k+1}(s')} \right) \\
&= \max_{\pi} \sum_{a \in \mathcal{A}(s)} \textcolor{blue}{\pi(a|s)} q_k(s,a) , \quad \forall s \in \mathcal{S} \\
&= \max_{a \in \mathcal{A}(s)} q_k(s,a).
\end{aligned}$$`

综上，求解BOE的流程可以分为下面几步：

- `$\forall s \in \mathcal{S}$`，有一个当前的估计值`$v_k(s)$`。

- `$\forall a\in \mathcal{A}(s)$`，计算

`$$q_k(s,a)=\sum_{r \in \mathcal{R}(s,a)} p(r|s,a)r + \gamma \sum_{s' \in \mathcal{S}} p(s'|s,a)v_k(s').$$`

- 使用贪心策略计算`$\pi_{k+1}$`：

`$$\pi_{k+1}(a|s) = \left\{
\begin{array}{ll}
1 & a = a_k^*(s) \\
0 & a \neq a_k^*(s)
\end{array}
\right., \quad a_k^*(s) = \arg\max_a q_k(s, a).$$`

- 计算`$v_{k+1}(s)=\max_a q_k(s,a)$`。

## Policy Optimality

假设`$\mathbf{v}^*$`是BOE的解，那么

`$$\mathbf{v}^* = \max_{\pi} (\mathbf{r}_{\pi} + \gamma \mathbf{P}_{\pi} \mathbf{v}^*), \quad \mathbf{\pi}^* = \arg\max_{\pi} (\mathbf{r}_{\pi} + \gamma \mathbf{P}_{\pi} \mathbf{v}^*).$$`

进而将BOE转化为特殊的Bellman equation：

`$$\mathbf{v}^* = \mathbf{r}_{\pi^*} + \gamma \mathbf{P}_{\pi^*} \mathbf{v}^*.$$`

我们有以下结论：

- `$\mathbf{v}^*$`是最优state value，`$\pi^*$`是最优policy。
- `$\forall s \in \mathcal{S}$`，最优policy `$\pi^*$`为

`$$\pi^*(a|s) = \left\{
\begin{array}{ll}
1 & a = a^*(s) \\
0 & a \neq a^*(s)
\end{array}
\right..$$`

其中，
`$$\begin{aligned}
a^*(s)&=\arg\max_a q^*(a,s), \\
q^*(s,a)&=\sum_{r \in \mathcal{R}(s,a)} p(r|s,a)r + \gamma \sum_{s' \in \mathcal{S}} p(s'|s,a)v^*(s').
\end{aligned}$$`
