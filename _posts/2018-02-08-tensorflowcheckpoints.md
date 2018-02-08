---
layout: post
title: "Tensorflow之Checkpoints"
author: "L-Y-N"
categories: tensorflow
tags: [tensorflow， machine learning]
image: 2018-02-08-tensorflowcheckpoints.png
---

# CheckPoints

tensorflow提供两种方式对模型进行保存和重载：

- checkpoints方法，和代码相关
- SaveModel方法，和代码无关

## 代码样本

代码在：`https://www.tensorflow.org/get_started/Iris%20classification%20example`

## 保存未训练完的模型

Estimator自动将如下内容写入到硬盘中：

- checkpoints，即是在训练过程中，模型的各个版本
- event files，包含用TensorBoard进行可视化的信息

同时，我们可以指定信息保存的目录

```python
#设定model_dir
model_dir='models/iris'
```

```python
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units=[10, 10],
    n_classes=3,
    model_dir='models/iris')
```

现在假定我们调用了`train`函数，进行训练模型

```python
classifier.train(
        input_fn=lambda:train_input_fn(train_x, train_y, batch_size=100),
                steps=200)
```

这时将会把checkpoints和其他文件写入到指定的文件夹中，如下图所示：

![first call to train](https://www.tensorflow.org/images/first_train_calls.png)**the first call to train()**

## 保存点的频率

默认情况下，Estimator根据以下策略来保存checkpoints:

* 每十分钟保存一次
* 在train（）开始和结束时分别保存一次
* 保存最近的5次checkpoints

同样，我们可以对保存测率进行自行配置：

1. 建立一个`RunConfig`对象，用来定义自己的策略
2. 当建立一个Estimator时，将`RunConfig`对象传入Estimator的`config`参数中

例如，如下代码将保存策略修改为：**每20分钟保存一次，保存最近10次checkpoints**

```python
my_checkpointing_config = tf.estimator.RunConfig(
    save_checkpoints_secs = 20*60,  # Save checkpoints every 20 minutes.
    keep_checkpoint_max = 10,       # Retain the 10 most recent checkpoints.
)

classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units=[10, 10],
    n_classes=3,
    model_dir='models/iris',
    config=my_checkpointing_config)
```

## 重载模型

在Estimator中调用`train` `eval` `predict`方法的时候，将会发生如下：

1. Estimator通过调用`model_fn()`函数来建立一个模型的graph
2. Estimator读取最近的checkpoints的数据，并对新模型的权重(weights)进行初始化

换句话来说，如下图所示，一旦checkpoints存在，tensorflow将在我们每次调用`train\evaluate\predct`的时候，重建模型。

![restore model](https://www.tensorflow.org/images/subsequent_calls.png)

## 避免错误的重载

只有在模型和checkpoints相互兼容的时候，重载模型才是可行的。比如，我们训练了一个DNNClassifier，包含两层hidden layer，每一层有10个节点：

```python
classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[10, 10],
    n_classes=3,
    model_dir='models/iris')

classifier.train(
    input_fn=lambda:train_input_fn(train_x, train_y, batch_size=100),
        steps=200)
```

并且，指定了checkpoints保存的目录。

**之后，我们可能想要对神经网络的结构进行修改，将每一层的神经元个数修改为20个。**然后尝试**重新训练**这个新的神经网络2号。

```python
classifier2 = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units=[20, 20],  # Change the number of neurons in the model.
    n_classes=3,
    model_dir='models/iris')

classifier.train(
    input_fn=lambda:train_input_fn(train_x, train_y, batch_size=100),
        steps=200)
```

由于神经网络的结构已经发生变化，这里就会报错了，因为并不兼容。错误如下：

```python
...
InvalidArgumentError (see above for traceback): tensor_name =
dnn/hiddenlayer_1/bias/t_0/Adagrad; shape in shape_and_slice spec [10]
does not match the shape stored in checkpoint: [20]
```

如果想要在不同的模型之间做比较的话，为每一个模型都要建立一个独立保存checkpoints的路径。