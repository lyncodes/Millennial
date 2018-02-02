---
layout: post
title: "10 minutes to pandas（上）"
author: "L-Y-N"
categories: pandas
tags: [pandas]
image: 2018-02-02-10 minutes to pandas(1).png
---

# 10 minutes to pandas（上） 

这一次我来尝试将其翻译过来，一来是方便透彻理解，二来方便以后回忆。

---

首先载入相应的包

```python
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
```

## object creation 对象的创立

**前言：**

pandas主要有两种数据类型，Series和DataFrame，分别可以看作是一维的数组和二位的数组，每个单元容器内可以存放int、float、str等各种类型的数据。

dataframe有些类似于excel的意思。:smile:

**正文：**

### Series创建

```python
s = pd.Series([1,3,5,np.nan,6,8])
s
```

```python
0    1.0
1    3.0
2    5.0
3    NaN
4    6.0
5    8.0
dtype: float64
```

可见，调用Series函数，将方括号内由逗号分割的数字，转变为一个Series，是一个线性结构。

###DataFrame的创建

#### 以给定array来创建DataFrame

由于pandas最开始是用于做金融分析的，所以pandas中对于时间处理的函数很多。

*date_range()函数用于生成时间序列*

输入

```python
dates = pd.date_range('20130101', periods=6)
dates
```

输出

```python
DatetimeIndex(['2013-01-01', '2013-01-02', '2013-01-03', '2013-01-04',
               '2013-01-05', '2013-01-06'],
              dtype='datetime64[ns]', freq='D')
```

*创建一个DataFrame*

输入

```python
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))
#以时间为index，以ABCD的自定义标签为columns
```

输出

![dataframe-creation](https://github.com/lyncodes/image_repo/blob/master/pandas/2018-02-02-dataframe-creation.PNG?raw=true)



可以发现dataframe的结构和excel非常相似。

在pandas术语中，index是左侧的index坐标，column是上侧的index坐标	

#### 以给定的Series来创建DataFrame

输入：

```python
df2 = pd.DataFrame({ 'A' : 1.,
                     'B' : pd.Timestamp('20130102'),
                     'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                     'D' : np.array([3] * 4,dtype='int32'),
                    'E' : pd.Categorical(["test","train","test","train"]),
                    'F' : 'foo' })
df2
```

输出：

![dataframe-creation2](https://github.com/lyncodes/image_repo/blob/master/pandas/2018-02-02-dataframe-creation2.PNG?raw=true)

可见，左侧没有指定index，所以自动生成，0、1、2、3、4…………，同时指定columns为“ABCDEF”

并且不同的地方，可以存在不同的数据类型（int、float、str）等，这对于繁杂的数据内容很方便。

## 对dataframe的预览

头部预览：

```python
df.head()#用来预览df的前五行内容
```



![dataframe-head](https://github.com/lyncodes/image_repo/blob/master/pandas/2018-02-02-dataframe-head.PNG?raw=true)

尾部预览：

```python
df.tail()#用来预览最后五行内容
```

![dataframe-tail](https://github.com/lyncodes/image_repo/blob/master/pandas/2018-02-02-dataframe-tail.PNG?raw=true)

### 查看一个DataFrame的index和columns

输入：

```python
df.index#查看index
```

输出：

```python
DatetimeIndex(['2013-01-01', '2013-01-02', '2013-01-03', '2013-01-04',
               '2013-01-05', '2013-01-06'],
              dtype='datetime64[ns]', freq='D')
```

输入：

```python
df.columns#查看columns
```

输出：

```python
Index(['A', 'B', 'C', 'D'], dtype='object')
```

### 对DataFrame的快速基本统计

用describe函数对其进行统计

输入：

```python
df.describe（）
```

输出：

```python
              A         B         C         D
count  6.000000  6.000000  6.000000  6.000000
mean   0.073711 -0.431125 -0.687758 -0.233103
std    0.843157  0.922818  0.779887  0.973118
min   -0.861849 -2.104569 -1.509059 -1.135632
25%   -0.611510 -0.600794 -1.368714 -1.076610
50%    0.022070 -0.228039 -0.767252 -0.386188
75%    0.658444  0.041933 -0.034326  0.461706
max    1.212112  0.567020  0.276232  1.071804
```

可以观察出：

* count用于统计各个columns的数据个数，看是否缺失数据

* mean用于统计各个columns的数据平均值

* std表示各组数据的标准差

* min、max表示最大值

* 25、50、75%表示统计数字特征 
### dataframe的转置

输入：

```python
df.T
```

输出：

```python
   2013-01-01  2013-01-02  2013-01-03  2013-01-04  2013-01-05  2013-01-06
A    0.469112    1.212112   -0.861849    0.721555   -0.424972   -0.673690
B   -0.282863   -0.173215   -2.104569   -0.706771    0.567020    0.113648
C   -1.509059    0.119209   -0.494929   -1.039575    0.276232   -1.478427
D   -1.135632   -1.044236    1.071804    0.271860   -1.087401    0.524988
```

### 指定轴对dataframe进行索引

输入：

```python
df.sort_index(axis=1, ascending=False)
#axis=1表示按columns的值从大到小进行排列
#false表示数据按降序排列
```

输出：

```python
                   D         C         B         A
2013-01-01 -1.135632 -1.509059 -0.282863  0.469112
2013-01-02 -1.044236  0.119209 -0.173215  1.212112
2013-01-03  1.071804 -0.494929 -2.104569 -0.861849
2013-01-04  0.271860 -1.039575 -0.706771  0.721555
2013-01-05 -1.087401  0.276232  0.567020 -0.424972
2013-01-06  0.524988 -1.478427  0.113648 -0.673690
```

### 指定columns进行排序

输入：

```python
df.sort_values(by='B')#按照B columns对数据进行排序
```

输出：

```python
                   A         B         C         D
2013-01-03 -0.861849 -2.104569 -0.494929  1.071804
2013-01-04  0.721555 -0.706771 -1.039575  0.271860
2013-01-01  0.469112 -0.282863 -1.509059 -1.135632
2013-01-02  1.212112 -0.173215  0.119209 -1.044236
2013-01-06 -0.673690  0.113648 -1.478427  0.524988
2013-01-05 -0.424972  0.567020  0.276232 -1.087401
```

## 数据的选择

### 按行、列对dataframe进行提取

输入：

```python
df['A']
#这样可以将A columns的数据提取出来
```

输出：

```python
2013-01-01    0.469112
2013-01-02    1.212112
2013-01-03   -0.861849
2013-01-04    0.721555
2013-01-05   -0.424972
2013-01-06   -0.673690
Freq: D, Name: A, dtype: float64
```

输入：

```python
df[0:3]
#对columns进行切片操作也是可以的
```

输出：

```python
                   A         B         C         D
2013-01-01  0.469112 -0.282863 -1.509059 -1.135632
2013-01-02  1.212112 -0.173215  0.119209 -1.044236
2013-01-03 -0.861849 -2.104569 -0.494929  1.071804
```

输入：

```python
df['20130102':'20130104']
#如果按照index的名字进行提取，那么就可以按照index进行提取
#同样支持    切片操作
```

输出：

```python
                   A         B         C         D
2013-01-02  1.212112 -0.173215  0.119209 -1.044236
2013-01-03 -0.861849 -2.104569 -0.494929  1.071804
2013-01-04  0.721555 -0.706771 -1.039575  0.271860
```

### 按区域对dataframe进行提取

输入：

```python
df.loc[dates[0]]
#按照dates[0],提取出来一行的所有内容
```

输出：

```python
A    0.469112
B   -0.282863
C   -1.509059
D   -1.135632
Name: 2013-01-01 00:00:00, dtype: float64
```

输入：

```python
df.loc[:,['A','B']]
#逗号前，切片操作，取所有的行
#逗号后，传入['A','B']  提取AB两列
```

输出：

```python
                   A         B
2013-01-01  0.469112 -0.282863
2013-01-02  1.212112 -0.173215
2013-01-03 -0.861849 -2.104569
2013-01-04  0.721555 -0.706771
2013-01-05 -0.424972  0.567020
2013-01-06 -0.673690  0.113648
```

输入：

```python
df.loc['20130102':'20130104',['A','B']]
#逗号前，切片操作，取三行
#逗号后，传入['A','B']  提取AB两列
```

输出：

```python
                   A         B
2013-01-02  1.212112 -0.173215
2013-01-03 -0.861849 -2.104569
2013-01-04  0.721555 -0.706771
```

![box-selection](https://github.com/lyncodes/image_repo/blob/master/pandas/2018-02-02-%E6%A1%86%E9%80%89.png?raw=true)类似于此图，分别指定横纵范围，从dataframe中提取一定范围内的数据

**更准确的来说**我们可以精确到对某一个位置的数据进行精确的定位，并且进行提取。

### 布尔索引

输入：

```python
df[df.A > 0]
#对df的A columns内容提取，并以是否>0进行索引
#然后将提取出来的数据，再生成一个心得df
```

输出：

```python
                   A         B         C         D
2013-01-01  0.469112 -0.282863 -1.509059 -1.135632
2013-01-02  1.212112 -0.173215  0.119209 -1.044236
2013-01-04  0.721555 -0.706771 -1.039575  0.271860

#可以看到，A columns下的数据全部都是大于零的
```

输入：

```python
df[df > 0]
#判断df中大于0的数据
#满足条件的数据会显示出来
#不满足条件的数据会以NaN显示
```

输出：

```python
                   A         B         C         D
2013-01-01  0.469112       NaN       NaN       NaN
2013-01-02  1.212112       NaN  0.119209       NaN
2013-01-03       NaN       NaN       NaN  1.071804
2013-01-04  0.721555       NaN       NaN  0.271860
2013-01-05       NaN  0.567020  0.276232       NaN
2013-01-06       NaN  0.113648       NaN  0.524988
```

## 对缺失数据的处理

先建立一个有缺失数据的dataframe

输入：

```python
df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])
df1.loc[dates[0]:dates[1],'E'] = 1	
df1
```

输出：

```python
 				  A         B         C  D    F    E
2013-01-01  0.000000  0.000000 -1.509059  5  NaN  1.0
2013-01-02  1.212112 -0.173215  0.119209  5  1.0  1.0
2013-01-03 -0.861849 -2.104569 -0.494929  5  2.0  NaN
2013-01-04  0.721555 -0.706771 -1.039575  5  3.0  NaN
```

F E 列分别有缺失值。

### 剔除具有缺失值的rows

输入：

```python
df1.dropna(how='any')
```

输出：

```python
				   A         B         C  D    F    E
2013-01-02  1.212112 -0.173215  0.119209  5  1.0  1.0
#可见仅剩第二行具有全部的值，而被表留下来了
```

### 对缺失的值进行填充

输入：

```python
df1.fillna(value=5)
#以5对当前dataframe进行填充
```

输出：

```python
				   A         B         C  D    F    E
2013-01-01  0.000000  0.000000 -1.509059  5  5.0  1.0
2013-01-02  1.212112 -0.173215  0.119209  5  1.0  1.0
2013-01-03 -0.861849 -2.104569 -0.494929  5  2.0  5.0
2013-01-04  0.721555 -0.706771 -1.039575  5  3.0  5.0
#可见上面的NaN都被5所取代
```

## 对dataframe元素的运算

### 函数运算

输入：

```python
df.apply(np.cumsum)
#对df进行卷积运算
```

输出：

```python
 				   A         B         C   D     F
2013-01-01  0.000000  0.000000 -1.509059   5   NaN
2013-01-02  1.212112 -0.173215 -1.389850  10   1.0
2013-01-03  0.350263 -2.277784 -1.884779  15   3.0
2013-01-04  1.071818 -2.984555 -2.924354  20   6.0
2013-01-05  0.646846 -2.417535 -2.648122  25  10.0
2013-01-06 -0.026844 -2.303886 -4.126549  30  15.0
```

输入：

```python
df.apply(lambda x: x.max() - x.min())
#应用lambda函数，求最大值减去最小值
```

输出：

```python
A    2.073961
B    2.671590
C    1.785291
D    0.000000
F    4.000000
dtype: float64
```

### 简单条形统计

输入：

```python
s = pd.Series(np.random.randint(0, 7, size=10))
#利用numpy，在0-7内，生成10个随机数，并且转化成Series
s
```

输出：

```python
0    4
1    2
2    1
3    2
4    6
5    4
6    4
7    6
8    4
9    4
dtype: int64
```

输入：**value_counts()函数用于简单统计不同值的出现次数**

```python
s.value_counts()
#对s的值的个数进行统计
#这在实际应用中非常常见，非常重要
```

输出：

```python
4    5
6    2
2    2
1    1
dtype: int64
#与以上吻合，4出现了5次，6出现了2次
```

### 对字符串的操作

输入：

```python
s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
#建立一个包含字符串的Series
s.str.lower()
#将其子母转化为小写
```

输出：

```python
0       a
1       b
2       c
3    aaba
4    baca
5     NaN
6    caba
7     dog
8     cat
dtype: object
```

**明显可见将所有单词都转化成了小写。**

