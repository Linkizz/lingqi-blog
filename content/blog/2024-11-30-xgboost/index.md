---
title: XGBoost
author: Lingqi Zeng
date: '2024-11-30'
slug: xgboost
categories:
  - Machine Learning
tags: []
---

## XGBoost数学原理

XGBoost与[GBDT](/blog/gradient-boosting/)一样，使用向前分步算法，但是XGBoost的目标是结构风险最小化，即加入了正则化项：

`$$\mathcal{L}(f)=\sum_{i=1}^{N}\ell(y_{i},f(\mathcal{x}_{i}))+\Omega(f)$$`

其中，`$\Omega(f)=\gamma J+\frac{1}{2}\lambda \sum_{j=1}^{J}w_{j}^{2}$`是正则化项，`$J$`是叶子结点数量，`$\gamma,\lambda \geq 0,$`是参数。

在构建第`$m$`棵树时，目标函数为：

`$$\mathcal{L}_{m}=\sum_{i=1}^{N}\ell(y_{i},f_{m-1}(\mathcal{x}_{i})+F_{m}(\mathcal{x}_{i}))+\Omega(F_{m})+\text{const}$$`

不同于GBDT只使用了一阶信息，XGBoost对损失函数`$\ell$`进行二阶泰勒展开：

`$$\mathcal{L}_{m}\approx \sum_{i=1}^{N}\bigg [\ell(y_{i},f_{m-1}(\mathcal{x}_{i}))+g_{im}F_{m}(\mathcal{x}_{i})+\frac{1}{2}h_{im}F_{m}^2(\mathbf{x}_{i})\bigg ]+\Omega(F_{m})+\text{const}$$`

其中，

`$$g_{im}=\bigg [ \frac{\partial\ell(y_{i},f(\mathbf{x}_{i}))}{\partial f(\mathbf{x}_{i})}\bigg ]_{f=f_{m-1}}, h_{im}=\bigg [ \frac{\partial^2\ell(y_{i},f(\mathbf{x}_{i}))}{\partial f(\mathbf{x}_{i})^{2}}\bigg ]_{f=f_{m-1}}$$`

在回归树中，每一棵树`$F(\mathbf{x})$`实际上是将样本`$\mathbf{x}$`划分到某个结点，然后输出该结点的值，因此`$F(\mathbf{x})=w_{q(\mathbf{x})}$`，其中`$q:\mathbb{R}^{D} \rightarrow \{1,2,\cdots,J\}$`将样本`$\mathbf{x}$`映射到对应的结点，`$\mathbf{w} \in  \mathbb{R}^{J}$`是各叶结点的输出值。进而目标函数可以表示成如下形式（忽略常数项）：

`$$\begin{aligned}\mathcal{L}_{m}(q,\mathbf{w}) &\approx \sum_{i=1}^{N}\bigg [g_{im}F_{m}(\mathbf{x}_{i})+\frac{1}{2}h_{im}F_{m}^2(\mathbf{x}_{i})\bigg ]+\gamma J+\frac{1}{2}\lambda \sum_{j=1}^{J}w_{j}^{2} \\
&=\sum_{j=1}^{J}\bigg [ (\sum_{i \in I_{j}}g_{im})w_{j}+\frac{1}{2}(\sum_{i \in I_{j}}h_{im}+\lambda)w_{j}^{2} \bigg ]+\gamma J \\
&=\sum_{j=1}^{J}\bigg [ G_{jm}w_{j}+\frac{1}{2}(H_{jm}+\lambda)w_{j}^{2} \bigg ]+\gamma J\end{aligned}$$`

其中，`$I_{j}=\{i:q(\mathbf{x}_{i})=j\}$`表示所有在第`$j$`个结点的样本索引集合，`$G_{jm}=\sum_{i \in I_{j}}g_{im},H_{jm}=\sum_{i \in I_{j}}h_{im}$`分别表示属于第`$j$`个结点的样本的一阶和二阶偏导数之和。

这样，目标函数就变成了以`$w_{j}$`为自变量的二次函数，那么各结点的最优输出值为

`$$w_{j}^{*}=-\frac{G_{jm}}{H_{jm}+\lambda}$$`

此时，目标函数值为

`$$\mathcal{L}_{m}(q,\mathbf{w}^{*})=-\frac{1}{2}\sum_{j=1}^{J}\frac{G_{jm}^{2}}{H_{jm}+\lambda}+\gamma J$$`

与构建[决策树](/blog/decision-tree/)一样，使用贪心算法逐个对结点进行划分。具体地，对于第`$j$`个结点，将其分裂为左右子树，`$I_{L}=\{i:q(\mathbf{x}_{i})=L\},I_{R}=\{i:q(\mathbf{x}_{i})=R\}$`，结点分裂增益为：

`$$\text{gain}=\frac{1}{2}\bigg [ \frac{G_{L}^{2}}{H_{L}+\lambda}+\frac{G_{R}^{2}}{H_{R}+\lambda}-\frac{(G_{L}+G_{R})^{2}}{(H_{L}+H_{R})+\lambda} \bigg ]-\gamma$$`

其中，`$G_{L}=\sum_{i \in I_{L}}g_{im},G_{R}=\sum_{i \in I_{R}}g_{im},H_{L}=\sum_{i \in I_{L}}h_{im},H_{R}=\sum_{i \in I_{R}}h_{im}$`，进而分裂增益`$\text{gain}$`表示不分裂时的目标函数值减去分裂后左右子树目标函数值和，若`$\text{gain}>0$`，说明分裂后损失减小，进行分裂，否则就不分裂。

## 分桶算法

通常来说，当特征为连续值时，用贪心法遍历寻求特征的最优划分点计算量非常大，因此可以使用近似算法。近似算法对特征进行分桶，对连续特征则按照百分位分桶，对离散特征则按照离散值分桶。在每次计算增益`$gain$`时，会先对每个桶中的`$G$`和`$H$`累计计算。

分桶有两种模式：

1. 全局模式：在算法开始时，对每个特征分桶一次，后续的分裂都依赖于该分桶并不再更新。

    - 优点：仅需计算一次。
    - 缺点：在经过多次分裂之后，叶结点的样本有可能在很多全局桶中是空的。

2. 局部模式：每次划分之后都会重新做一次分桶。

    - 优点：每次分桶都能保证各桶中的样本数量都是均匀的。
    - 缺点：计算量较大。
 
## 缺失值处理

XGBoost有专门处理缺失值的算法：

1. 在选择最佳分裂特征时，对于每一个特征，遍历各个划分点前，先将所有缺失值样本排序，将缺失值样本放在右边，这样可以保证缺失值样本全部分到右子树，进而计算分裂增益，求出最优划分点。

2. 再用同样的方法将所有缺失值样本全部分到左子树，也计算分裂增益，求出最优划分点。

3. 最后选取最大的分裂增益对应的特征和和划分点作为分裂依据。

## 参考文献

1.Kevin P. Murphy. Probabilistic Machine Learning: An introduction. MIT Press, 2022.

2.李航. 统计学习方法（第2版）. 北京: 清华大学出版社, 2019.

3.[https://www.bookstack.cn/read/huaxiaozhuan-ai/spilt.2.137d736745ba14b2.md#bkkgr](https://www.bookstack.cn/read/huaxiaozhuan-ai/spilt.2.137d736745ba14b2.md#bkkgr)