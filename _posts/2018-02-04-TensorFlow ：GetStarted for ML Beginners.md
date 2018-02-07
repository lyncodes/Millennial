---
layout: post
title: "TensorFlow ：GetStarted for ML Beginners"
author: "L-Y-N"
categories: tensorflow
tags: [tensorflow]
image: 2018-02-04-GetStarted_for_ML_Beginners.png
---
# TensorFlow ：GetStarted for ML Beginners

TensorFlow的官网在get started中为没有接触过机器学习的人群，准备了一篇机器学习的基础介绍

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

在这个例子中，是花瓣、花萼大小和花种类之间的关系。

### 建模

一些简单的模型可以用几行简单的线性代数进行描述，但是更多复杂的模型，涵盖了大量错综复杂的数学方程和参数，这让我们很难用数学来对其进行解释。

在这个例子中，我们有可能用传统编程的方法，规定一大堆条件语句。也许我们可能用很长一段时间来搞清楚花瓣、花萼和种类之间的关系。然而，一个好的机器学习方法是自动为我们生成模型。

如果我们给恰当的机器学习模型提供足量的具有代表性的数据样本，那么这个程序就能辨别在花瓣、花萼、花种之间的某种关系。

### 训练

训练就是在建立模型的过程中，对模型进行不断的优化的过程。

Iris花的辨别过程，是一个监督学习的例子。

> The Iris problem is an example of [**supervised machine learning**](https://developers.google.com/machine-learning/glossary/?hl=zh-cn#supervised_machine_learning) in which a model is trained from examples that contain labels. (In [**unsupervised machine learning**](https://developers.google.com/machine-learning/glossary/?hl=zh-cn#unsupervised_machine_learning), the examples don't contain labels. Instead, the model typically finds patterns among the features.

* supervised ML(dataset with label)
* unsupervised ML(dataset without label)

## 程序的执行

源数据由tensorflow提供：

https://github.com/tensorflow/models

其中model文件夹内包含`premade_estimator.py`实例代码，将其运行之后，输出结果的最后几行如下所示：

```python
Prediction is "Sentosa" (99.5%), expected "Setosa"

Prediction is "Versicolor" (99.7%), expected "Versicolor"

Prediction is "Virginica" (96.6%), expected "Virginica"
```

## TensorFlow的程序架构

tensorflow的编程环境。

![tensorflow stack](https://www.tensorflow.org/images/tensorflow_programming_environment.png?hl=zh-cn)

官方则强烈推荐新手关注于：

* estimator
* dataset

## 对源码的解析

接下来我们将对`premade_estimator.py`进行深入一点的了解

首先程序的流程如下：

- Import and parse the data sets.（导入并解析数据集）
- Create feature columns to describe the data.（建立特征列来描述数据）
- Select the type of model（选择模型种类）
- Train the model.（训练模型）
- Evaluate the model's effectiveness.（评估模型的有效性）
- Let the trained model make predictions.（让训练好的模型做预测）

### Import and parse the data sets.（导入并解析数据集）

- train dataset：`http://download.tensorflow.org/data/iris_training.csv`
- test dataset: `http://download.tensorflow.org/data/iris_test.csv`

train set用于建立模型和对模型进行训练，test set用于对已经生成的模型进行测试。

这中间有一个此起彼伏的关系在里面：

* 更多的数据划分到train set中能够建立一个更准确的模型
* 更多的数据划分到test set中能够更好的检验模型的有效性，否则我们不能准确的评估模型的准确性

`premade_estimator.py`依赖于`load_data`函数，而`load_data`函数在`iris_data.py`中，用于从网络上下载数据，并在本地将数据载入内存中。

```python
TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']

...

def load_data(label_name='Species'):
    """Parses the csv file in TRAIN_URL and TEST_URL."""

    # Create a local copy of the training set.
    train_path = tf.keras.utils.get_file(fname=TRAIN_URL.split('/')[-1],
                                         origin=TRAIN_URL)
    # train_path now holds the pathname: ~/.keras/datasets/iris_training.csv

    # Parse the local CSV file.
    train = pd.read_csv(filepath_or_buffer=train_path,
                        names=CSV_COLUMN_NAMES,  # list of column names
                        header=0  # ignore the first row of the CSV file.
                       )
    # train now holds a pandas DataFrame, which is data structure
    # analogous to a table.

    # 1. Assign the DataFrame's labels (the right-most column) to train_label.
    # 2. Delete (pop) the labels from the DataFrame.
    # 3. Assign the remainder of the DataFrame to train_features
    train_features, train_label = train, train.pop(label_name)

    # Apply the preceding logic to the test set.
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)
    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_features, test_label = test, test.pop(label_name)

    # Return four DataFrames.
    return (train_features, train_label), (test_features, test_label)
```

`tf.keras`是tensorflow对Keras的封装，`tf.keras.utils.get_file`用于下载远程文件并在本地加载。

`load_data`函数可以返回两对(feature,label)数据对。分别是训练数据和测试数据。

```python
    # Call load_data() to parse the CSV file.
    (train_feature, train_label), (test_feature, test_label) = load_data()
```

### 对数据的描述

特征列(**feature column**)是一种数据结构，用于在模型中对每一个特征进行数据解释。在Iris花分辨问题中，特征数据是浮点数。

从代码的角度来说，我们建立一个`feature_column`对象。

```python
# Create feature columns for all features.
my_feature_columns = []
for key in train_x.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
```

**其中，train_x在load_data函数中定义**

### 选择模型的种类

我们需要在现有的多种的数据模型中选择合适的模型，来对其进行训练。在这里，我们选用神经网络来解决Iris花分类问题。

**神经网络**能够在众多的features和labels中寻找出复杂的关系。一个神经网络是一个高度结构化的图型数据结构，由一层或多层隐藏层(hidden layers)组成。每一层都由一个或者多个神经元(neurons)组成。

并且神经网络也有多种分类，在这里我们采用全连接神经网络(fully connected neural network)。即是每一层的神经元从上一层的所有神经元接受输入信息。