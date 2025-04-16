---
title: 'LLM Tokenization'
author: Lingqi Zeng
date: '2025-01-22'
slug: llm-tokenization
categories:
  - Machine Learning
tags:
  - NLP
---

## 分词的介绍

分词是将文本拆分成一系列词元的过程，使得模型能够进行处理，词元称为“token”。一个token可以是一个词，一个子词，或者一个字符。

分词为什么重要呢？LLM会产生一些奇怪的问题，例如：

- 不能拼写单词

- 不能反转一个字符串

- GPT-2处理代码的能力很差

前两个问题是由于大多数LLM以子词为粒度来进行分词，因而在字符级别的问题容易出错。最后一个问题是因为GPT-2将连续的空格字符单独处理成独立的token，使得token序列变长，LLM难以捕捉序列的信息。除此之外，还有许多问题都可以追溯的分词上，这也是分词在大模型中那么重要的原因。

## 分词算法

以英语为例，考虑字符级别的分词算法，则训练的词汇表包含26个英文字母和特殊符号，这使得在处理文本时会将其拆分成更多的token，增加计算成本。并且由于最大token数量的限制，使用字符级别的分词算法会导致模型能够考虑的上下文内容减少。

考虑单词级别的分词算法，通过空白符和标点符号分词，则训练的词汇表包含语料库中出现的所有词汇。如“I like playing basketball.”会被分割成["I", "like", "playing", "basketball", "."]。这种分词方法的token数量比字符级别的方法要小的多，但还是有一些问题。例如，当出现语料库中没有出现过的词汇(Out of Vocabulary, OOV)，那么模型就会将其标记为”unk”，这使得模型处理没见过的词汇的能力大大下降。此外，这种方法会把“strong”、“stronger”和“strongest”看成完全不同的单词，模型难以捕捉这三个词之间的关系。

如果采用子词级别的分词算法，例如将上述三个单词分别分割成["strong"]、["strong", "er"]、["strong", "est"]，这使得模型能够理解三者之间的关系，并且也不会像字符级别的方法那样分割成过多的token。

接下来介绍一些LLM中常用的子词级别的分词算法。

### Byte-pair Encoding (BPE)

[BPE](https://arxiv.org/pdf/1508.07909)通过逐步合并高频字符对来构建词汇表，过程如下：

- 将语料库拆分成字符级别以构建初始词汇表；
- 计算所有字符对的数量；
- 合并最高频的字符对作为新的token加入词汇表；
- 不断重复这个过程，直到达到预期的词汇表大小。

如果想了解BPE的具体合并操作，可以看这个[简单例子](https://en.wikipedia.org/wiki/Byte_pair_encoding)。

通过以上BPE算法，仍可能会出现OOV问题。为了解决这个问题，可以让初始词汇表包含所有可能的字符，进而所有的词汇都肯定可以分割成词汇表中的某些元素。若直接使用Unicode符号作为初始词汇表，则所有符号都包含其中，但Unicode符号超过100万，模型计算和存储开销过大。

限制词汇量的一种方法是使用字节级别的BPE分词算法(Byte-level BPE)，我们可以将Unicode按UTF-8方式进行编码，UTF-8是一种变长的编码方式，它可以用1-4个字节表示一个符号，由于每个字节是8位二进制数，因此一个字节有256种情况。因此，将词汇表初始化为256个token，那么所有的符号都可以用这个256个token来表示。如果想要了解更多字符编码的知识，可以看这篇[博客](https://www.ruanyifeng.com/blog/2007/10/ascii_unicode_and_utf-8.html)。

### WordPiece

[WordPiece](https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/37842.pdf)与BPE的主要区别是合并字符的规则，WordPiece合并使得语言模型似然函数值增加最多的字符对。假设句子`$S=(t_1, t_2, \cdots, t_n)$`是由`$n$`个子词组成，各子词相互独立，则对数似然函数为：

`$$\log P(S)=\sum_{i=1}^{n}\log P(t_i)$$`

若将相邻位置的`$t_{i}$`和`$t_{i+1}$`合并，产生新词`$t_{j}$`，则似然函数变化如下：

`$$\log P(t_{j})-(\log P(t_{i})+\log P(t_{i+1}))=\log (\frac{P(t_{j})}{P(t_{i})P(t_{I+1})})$$`

其中的子词概率可以用频率来近似，每次合并使得对数似然提升最大的字符对。

根据WordPiece的合并规则可以发现，WordPiece在分词时，每次都是匹配最长的子词，而BPE会记录合并的顺序，并按照顺序进行合并。举个例子，假设训练出的词汇表为：["b", "h", "p", "##g", "##n", "##s", "##u", "##gs", "hu", "hug"]，那么“hugs”会被WordPiece分割成["hug", "##s"]，但被BPE分割成["hu", "##gs"]。

### Unigram

与BPE和WordPiece不同，Unigram是从一个较大的词表中不断合并token以减少token数量，最终达到预期的token数。具体步骤如下：

- 建立一个较大的词表。一般可以用语料库中所有的字符加上一些常见的子词初始化，也可以用BPE算法初始化。

- 针对初始词表，使用EM算法求解各子词的概率。

- 对于每个子词，计算移除该子词时，总loss的增加量。

- 选取loss增加量最小的前80%的子词。为了避免OOV问题，单字符不能丢弃。

## 参考文献

1.Rico Sennrich et al. [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/pdf/1508.07909) [pdf]

2.Mike Schuster et al. [Japanese and Korean Voice Search](https://static.googleusercontent.com/media/research.google.com/ja//pubs/archive/37842.pdf) [pdf]

3.阮一峰. [字符编码笔记：ASCII，Unicode 和 UTF-8](https://www.ruanyifeng.com/blog/2007/10/ascii_unicode_and_utf-8.html)

4.[大语言模型语料库相关分词器的简单理解（以Unigram模型为例）](https://zhuanlan.zhihu.com/p/686186845)