---
layout: post
title: "TensorFlow ：GetStarted for ML Beginners"
author: "L-Y-N"
categories: tensorflow
tags: [tensorflow]
image: 2018-02-02-10MinutesToPandas2.png
---
# TensorFlow ：GetStarted for ML Beginners

TensorFlow的官网在get started中为没有接触过机器学习的人群，准备了一篇机器学习的基础介绍fgh

## 问题背景介绍

讲解了如何利用机器学习对数百张**Iris**花进行分类的例子。

>Imagine you are a botanist seeking an automated way to classify each Iris flower you find. Machine learning provides many ways to classify flowers. For instance, a sophisticated machine learning program could classify flowers based on photographs. Our ambitions are more modest--we're going to classify Iris flowers based solely on the length and width of their [sepals](https://en.wikipedia.org/wiki/Sepal) and [petals](https://en.wikipedia.org/wiki/Petal).

**我们将利用机器学习的办法，仅仅依靠判断sepals和petals的长度和宽度，从众多照片中自动判断iris花的种类**

iris花大约有300种，但是在这两个例子中我们的程序将只用于分辨以下三种花

我个人而言，从肉眼来看，其实三种花长的差不多，但是在计算机眼中，它们是否是同一种花呢？我们拭目以待！

![iris](https://www.tensorflow.org/images/iris_three_species.jpg?hl=zh-cn)**From left to right, Iris setosa , Iris versicolor , and Iris virginica**

![seapal and petals](https://upload.wikimedia.org/wikipedia/commons/thumb/7/78/Petal-sepal.jpg/220px-Petal-sepal.jpg)				sepal是萼片	petal是花瓣



## 原始数据来源及处理

幸运的是，已经有人对Iris花的相关原始数据进行了收集，向我们提供了150组数据



**分别对Sepal和Petal的width和length进行了测量**

**并给出了所对应的种类**

```python 
		Sepal length  Sepal width  Petal length  Petal width  Species
0             5.1          3.5           1.4          0.2     Iris-setosa
1             4.9          3.0           1.4          0.2     Iris-setosa
2             4.7          3.2           1.3          0.2     Iris-setosa
8             4.4          2.9           1.4          0.2     Iris-setosa
9             4.9          3.1           1.5          0.1     Iris-setosa

..            ...          ...           ...          ...             ...

..            ...          ...           ...          ...             ...
120           6.9          3.2           5.7          2.3  Iris-virginica
121           5.6          2.8           4.9          2.0  Iris-virginica
122           7.7          2.8           6.7          2.0  Iris-virginica
123           6.3          2.7           4.9          1.8  Iris-virginica
124           6.7          3.3           5.7          2.1  Iris-virginica
125           7.2          3.2           6.0          1.8  Iris-virginica
..            ...          ...           ...          ...             ...

..            ...          ...           ...          ...             ...
145           6.7          3.0           5.2          2.3  Iris-virginica
146           6.3          2.5           5.0          1.9  Iris-virginica
147           6.5          3.0           5.2          2.0  Iris-virginica
148           6.2          3.4           5.4          2.3  Iris-virginica
149           5.9          3.0           5.1          1.8  Iris-virginica
```

这里是部分原始数据。数据来自：[数据来源](http://archive.ics.uci.edu/ml/datasets/Iris)

**我们的任务就是通过这150个样本，让机器对其进行学习，然后用一个全新的数据输入，让程序根据生成的模型，判断全新数据所对应的iris花的种类。**

*其实这就是机器学习中的监督学习!*

介绍一下相关术语：

* 前四列的数据，我们称之为[feartures](https://developers.google.com/machine-learning/glossary/?refresh=1#f)（特征）
* 最后一列的数据，我们称之为[label](https://developers.google.com/machine-learning/glossary/?refresh=1#l)

features就是样本的characteristics（特点），而label就是我们将要预测的东西。

### 初步处理原始数据

我们人类能够认识Iris setosa , Iris versicolor , and Iris virginica，是自然的标签，是字符串。

但是对于机器来说，机器只能识别0和1，所以我们要做相应的映射处理

- 0 代表 setosa
- 1 代表 versicolor
- 2 代表 virginica

## 建模和对模型进行训练

> A **model** is the relationship between features and the label
>
> 模型就是在features和label中建立某种关系							————tensorflow 官网

to be continued………………