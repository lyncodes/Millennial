---
layout: post
title: "Tensorflow之Feature Column"
author: "L-Y-N"
categories: ml
tags: [tensorflow， machine learning]
image: 2018-02-08-tensorflowFeatureColumns.jpg
---

# Feature Column

feature columns可以看作是原始数据和Estimator之间的中介。Feature columns能够让我们将各种各样的原始数据转换成Estimator能够使用的数据形式。

在前面的Iris花分类问题中，我们的feature column是数字类型的数据（numerical feature），但是现实世界中的种种特征并不总是数字类型的。如下图所示：

![non-numeircal feature](https://www.tensorflow.org/images/feature_columns/feature_cloud.jpg)

## Deep Neural Network的输入

深度神经网络能够处理什么样的数据，当然了，是数字。总之，在一个神经网络中，每一个神经元都能根据权重，对输入数据进行乘法和加法运算。现实生活中，输入数据通常包含非数字的数据，比如说，在一个`prouct_class`中，能够包含如下三种非数字特征：

- kitchenware（厨房用具）
- electronics（电子用品）
- sports（体育用品）

**机器学习中通常用简单的向量来表示分类值。**

用数字1代表当前既有值，用数字0表示缺失值。当我们把`product_class`定义为`sports`的时候，将向量设定为`[0, 0, 1]`,即是表示：

- 0 ：kitchenware is absent
- 0  ：electroncis is absent
- 1：sports is present

虽然，原始数据既有数字类型，也有分类类型。但是在机器学习模型中，我们将所有的特征都进行数字化。（因为计算机只能处理数字:smile:)

