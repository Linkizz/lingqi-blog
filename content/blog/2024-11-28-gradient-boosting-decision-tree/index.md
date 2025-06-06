---
title: '梯度提升树'
author: Lingqi Zeng
date: '2024-11-28'
slug: gradient-boosting-decision-tree
categories:
  - Machine Learning
tags: []
---

## 向前分步算法

[Boosting](/blog/ensemble-learning/) 可以看作是一个加法模型：

$$f(\mathbf{x};\boldsymbol{\theta})=\sum_{m=1}^{M}\beta_{m}F_{m}(\mathbf{x};\boldsymbol{\theta}),$$

其中，\(F_m\) 是第 \(m\) 个基模型，\(\beta_m\) 是第 \(m\) 个基模型的权重。为了训练一个好的集成学习模型 \(f(\mathbf{x};\boldsymbol{\theta})\)，我们可以通过最小化如下的经验损失函数来实现：

$$\mathcal{L}=\sum_{i=1}^{N}\ell(y_i,f(\mathbf{x}_i)).$$

由于 Boosting 是加法模型，且后一基模型都是为了优化前一基模型构建的，如果直接对损失函数进行优化比较困难，复杂度较高。因此，可以使用**向前分步算法（Forward stagewise additive modeling）**，从前向后，每次学习一个基模型，逐步优化目标函数。具体地，第 \(m\) 步的优化目标如下：

$$(\beta_{m},\boldsymbol{\theta}_{m})=\argmin_{\beta_,\boldsymbol{\theta}}\sum_{i=1}^{N}\ell(y_{i},f_{m-1}(\mathbf{x}_{i})+\beta F(\mathbf{x}_{i};\boldsymbol{\theta})).$$

进而

$$f_{m}(\mathbf{x})=f_{m-1}(\mathbf{x})+\beta_{m} F(\mathbf{x};\boldsymbol{\theta}_{m})=f_{m-1}(\mathbf{x})+\beta_{m} F_{m}(\mathbf{x}).$$

## 梯度提升树

使用不同的损失函数，优化求解的方法就不一样，也就诞生了许多 Boosting 算法，如 Least Squares Boosting，AdaBoost，LogitBoost 等。这些 Boosting 算法的损失函数都比较简单，优化比较容易。对于更加一般的损失函数，优化问题解决起来可能比较困难。并且对每一种损失函数都要给出一种优化算法，较为麻烦。因此，Friedman 提出一种适用于一般损失函数的 Boosting 方法，称为**梯度提升（Gradient Boosting）**。

梯度提升算法可以用梯度下降法来解释，将原来的参数空间扩展到函数空间。假设我们要训练出使得损失函数最小的模型，即 \(\hat{\boldsymbol{f}}=\argmin_{\boldsymbol{f}}\mathcal{L}(\boldsymbol{f})\)，这里 \(\boldsymbol{f}=[f(\boldsymbol{x}_{1}),\cdots,f(\boldsymbol{x}_{N})]\)。把 \(\boldsymbol{f}\) 看作自变量参数，根据梯度下降法，得到如下更新公式：

$$\boldsymbol{f}_{m}=\boldsymbol{f}_{m-1}-\beta_{m}\boldsymbol{g}_{m},$$

其中，\(\beta_{m}\) 是步长，\(\boldsymbol{g}_{m}\) 是 \(\mathcal{L}(\boldsymbol{f})\) 在 \(\boldsymbol{f}=\boldsymbol{f}_{m-1}\) 的梯度：

$$g_{im}=\bigg [\frac{\partial \ell(y_{i},f(\mathbf{x}_{i}))}{\partial f(\mathbf{x}_{i})}\bigg ]_{f=f_{m-1}}.$$

因此，我们可以用一个基模型来拟合负梯度：

$$F_{m}=\argmin_{F} \sum_{i=1}^{N}(-g_{im}-F(\mathbf{x}_{i}))^{2}.$$

为了防止过拟合，对每一次拟合残差的基模型乘以一个学习率 \(0<\nu\leq 1\)，以对梯度提升学习的步长进行调整：

$$f_{m}(\mathbf{x})=f_{m-1}(\mathbf{x})+\nu F_{m}(\mathbf{x}).$$

对于平方损失函数 \(\frac{1}{2}(y_{i}-f(\mathbf{x}_{i}))^{2}\)，则 \(-g_{im}=y_{i}-f_{m-1}(\mathbf{x}_{i})\)，负梯度就是残差；对于一般的损失函数，负梯度为残差的近似值。

### 梯度提升树

梯度提升算法中的基模型通常使用决策树，即**梯度提升树（GBDT）**。一个回归树可表示为

$$F_{m}(\mathbf{x})=\sum_{j=1}^{J_{m}}w_{jm} \mathbb{I}(\mathbf{x} \in R_{jm}),$$

其中，\(R_{im}\) 表示第 \(m\) 棵回归树的第 \(j\) 个叶结点所在的区域，\(w_{jm}\) 表示该结点的输出值。

为了使用梯度提升算法，首先初始化一棵只有根结点的树，其输出值为使得损失函数最小的值；然后用回归树拟合负梯度，找到划分区域 \(R_{jm}\)；对每一个结点，通过解如下优化问题来计算各区域的输出值 \(w_{jm}\)：

$$\hat{w}_{jm}=\argmin_{w}\sum_{\mathbf{x}_{i} \in R_{jm}}\ell(y_{i},f_{m-1}(\mathbf{x}_{i})+w).$$

进而更新模型

$$f_{m}(\mathbf{x})=f_{m-1}(\mathbf{x})+\sum_{j=1}^{J_{m}}w_{jm}\mathbb{I}(\mathbf{x} \in R_{jm}).$$

最后得到梯度提升回归树

$$f_{M}(x)=\sum_{m=1}^{M}\sum_{j=1}^{J}w_{mj}\mathbb{I}(\mathbf{x} \in R_{jm}).$$

对于平方损失函数，各区域的最优输出值 \(\hat{w}_{jm}\) 就是对应叶结点拟合的负梯度的均值。

## 参考文献

1.Kevin P. Murphy. Probabilistic Machine Learning: An introduction. MIT Press, 2022.

2.李航. 统计学习方法（第 2 版）. 北京: 清华大学出版社, 2019.