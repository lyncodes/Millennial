---
layout: post
title: "2018-02-08-Tensorflow之Feature Column"
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

虽然，原始数据既有数字类型，也有分类类型。但是在机器学习模型中，我们将所有的特征都进行数字化。（因为计算机只能处理数字)

## 特征列（Feature Columns）

如下图所示，我们通过对Estimator的`feature_columns`赋值，指定对模型的输入。Feature Columns是模型和原始数据的桥梁。

![feature columns](https://www.tensorflow.org/images/feature_columns/inputs_to_model_bridge.jpg)

从左往右，分别是raw data、feature columns、Estimator。

`tf.feature_column`模块下，一共有9个函数如下。其余8个返回categorical column或者dense column对象，只有`bucketized_column`,从class中继承而来。

![categorical](https://www.tensorflow.org/images/feature_columns/some_constructors.jpg)

接下来我们将详细介绍这些函数。

##  数字列（Numeric column）

在Iris花分类问题中，调用`tf.feature_column.numeric_column`来处理输入的数据：

- SepalLength
- SepalWidth
- PetalLength
- PetalWidth

调用`tf.numeric_column`函数，默认会采用`float32`格式：

```python
# Defaults to a tf.float32 scalar.
numeric_feature_column = tf.feature_column.numeric_column(key="SepalLength")
```

也可以用`dtype`指定数据类型:

```python
# Represent a tf.float64 scalar.
numeric_feature_column = tf.feature_column.numeric_column(key="SepalLength",dtype=tf.float64)
```

默认情况下，一个feature column以向量形式存储数据，但是也可以更改形式为矩阵，示例代码如下：

```python
# Represent a 10-element vector in which each cell contains a tf.float32.
vector_feature_column = tf.feature_column.numeric_column(key="Bowling",
                                                         shape=10)

# Represent a 10x5 matrix in which each cell contains a tf.float32.
matrix_feature_column = tf.feature_column.numeric_column(key="MyMatrix",
                                                         shape=[10,5])
```

`shape=[10, 5]`指定存储结构为一个矩阵。

## Bucketized column(分组列)

通常，我们不会将一串数字直接调入进一个模型，而是先将其进行分类。因此，我们创建一个`bucketized column`。比如说，将一个数据认为是房子修建的年份，而不是认为数据只是一个单纯的数字。我们能够将年份划分为4部分：

![buckets](https://www.tensorflow.org/images/feature_columns/bucketized_column.jpg)用4组向量分别表示4个年代。

| Date Range         | Represented as... |
| ------------------ | ----------------- |
| < 1960             | [1, 0, 0, 0]      |
| >= 1960 but < 1980 | [0, 1, 0, 0]      |
| >= 1980 but < 2000 | [0, 0, 1, 0]      |
| > 2000             | [0, 0, 0, 1]      |

将年份分割成4部份，是为了让程序能够学习4个(individual weights),而不是只有一个weight。4个权重（weights）能够让模型更加丰富。更重要的是，这可以让模型更清晰的分辨不同的年代划分。以下代码演示如何创建一个bucketized feature：

```python
# First, convert the raw input to a numeric column.
numeric_feature_column = tf.feature_column.numeric_column("Year")

# Then, bucketize the numeric column on the years 1960, 1980, and 2000.
bucketized_feature_column = tf.feature_column.bucketized_column(
    source_column = numeric_feature_column,
    boundaries = [1960, 1980, 2000])
```

值得注意的是，**三个时间点，1960，1980，2000将所有时间划分成了4部分。**

## 分类标识列

分类标识列（categorical identity columns）可以看作是bucketized columns的一种特例。在传统的bucketized columns'中，每个bucket代表一个范围的内的值（比如说1960-1979）。在categorical identity columns中，每一个bucket代表一个独一无二的整数。例如，表达0-4之间的整数，即表达0，1，2，3。这样，categorical identity columns将如下图所示：

![categorical identity columns](https://www.tensorflow.org/images/feature_columns/categorical_column_with_identity.jpg)

一个满秩的向量组可以用于表达R（A)=n的n阶分类问题。