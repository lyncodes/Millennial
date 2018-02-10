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

一个满秩的向量组可以用于表达R（A)=n的n阶分类问题。在bucketized column中，模型能够在分类表示列中，对每一个`class`的学习赋予的权重。比如 用独一无二的整数来表示一个分类，而不是用字符串来代表分类。

- `0="kitchenware"`
- `1="electronics"`
- `2="sport"`

**用数字代表分类之后**，调用`tf.feature_column.categorical_column_with_identity`函数，来实现分类标识列(categorical identity column)。代码如下：

```python
#建立feature column，叫做"my_feature_b",
# 设定buckets的数量为4，即分成4部分
identity_feature_column = tf.feature_column.categorical_column_with_identity(
    key='my_feature_b',
    num_buckets=4) # Values [0, 4)

#同时，input_fn()函数必须返回一个字典，用于盛放feature_b的数据，并且其中的数字必须在0-4之间
def input_fn():
    ...
    return ({ 'my_feature_a':[7, 9, 5, 2], 'my_feature_b':[3, 1, 2, 2] },
            [Label_values])
```

## 分类词汇列

有时候，并不把字符串转化成为整数。而是直接用矩阵来代表他们。

![categorical vocabulary column](https://www.tensorflow.org/images/feature_columns/categorical_column_with_vocabulary.jpg)

tensorflow提供两种方法来生成categorical vocabulary columns：

- `tf.feature_column.categorical_column_with_vocabulary_list`
- `tf.feature_column.categorical_column_with_vocabulary_file`

其中一个是list函数，一个是file函数。

**list函数，基于显式词汇表将每个字符串映射成整数。**代码如下：

```python
vocabulary_feature_column =
    tf.feature_column.categorical_column_with_vocabulary_list(
        key="a feature returned by input_fn()",
        vocabulary_list=["kitchenware", "electronics", "sports"])
#通过在输入和vocabulary_lsit中建立映射，建立一个categorical feature
```

用list函数，调用过程非常直观，但是却有一个非常明显的缺点。当这个list非常长的时候，我们不可能来将其全部手动输入，所以我们就调用file函数，`tf.feature_column.categorical_column_with_vocabulary_file`。代码如下：

```python
vocabulary_feature_column =
    tf.feature_column.categorical_column_with_vocabulary_file(
        key="a feature returned by input_fn()",
        vocabulary_file="product_class.txt",
        vocabulary_size=3)
```

**参数中指定文件为product_class.txt**

**vocabulary_saize=3,表示这个txt文件中的元素个数，参考上面的list函数，一共3个元素，所以为3**。当然还有很多其他参数设置，详情查看tensorflow的API文档。

而其中的`produc_class.txt`文件则包含如下三个元素。

```python
kitchenware
electronics
sports
```

## Hashed Column(哈希列)

到目前位置，我们只用简单的数字来进行分类。然而，在有大量分类的情况下，我们将不可能为每一个vocabulary word和integer划分一个独立的分类，因为那样太消耗内存了。针对这种情况，我们不禁问道：“针对我的输入信息，我需要多少个分类？”事实上，`tf.feature_column.categorical_column_with_hash_bucket`函数，这个`hash_buvket`函数就能够让我们明确的确定分类的数量。

对于这种类型的特征列，模型会计算输入的哈希值，然后使用模运算符将其放入其中一个`hash_bucket_size`类别中，如下面的伪代码所示：

```python
# pseudocode
feature_id = hash(raw_feature) % hash_buckets_size
```

创建`feature_column`de 代码如下：

```python
hashed_feature_column =
    tf.feature_column.categorical_column_with_hash_bucket(
        key = "some_feature",
        hash_buckets_size = 100) # The number of categories
```

**简单来说，就是由我们指定一共生成多少个种类，而不是直接生成很多很多种类，以至于我们看不过来，这里是指定生成了100个种类。**

但是，这可能会出错，这很有可能会把一些本该分开的数据合并到同一个种类中了，如下图所示：

![hash bucket](https://www.tensorflow.org/images/feature_columns/hashed_column.jpg)

图中，`kitchenware`和`sports`都被划分到了12组中，而它们本应该被分开。

**虽然在机器学习中有很多违反直觉的现象，但是在实践中，hasd的结果往往却很有效！**这是因为hash categories为研究模型提供了**额外的特征**将`kitchenware`和`sports`进一步划分开。

## Crossed column

