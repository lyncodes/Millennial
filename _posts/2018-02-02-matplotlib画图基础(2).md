---
layout: post

title: "Matplotlib画图基础(2)"

author: "L-Y-N"

categories: matplotlib

tags: [matplotlib]

image: 2018-02-02-matplotlib_syntax2.png 
---

#  Matplotlib画图基础(2)

## 对数坐标设置logarithmic()

### 输入

```python
import numpy as np
import matplotlib.pyplot as plt

# Data for plotting
t = np.arange(0.01, 20.0, 0.01)

# Create figure
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
#布置2x2=4个subplots

# log y axis
ax1.semilogy(t, np.exp(-t / 5.0), color='r')#设置y轴为log类型
ax1.set(title='semilogy')
ax1.grid()

# log x axis
ax2.semilogx(t, np.sin(2 * np.pi * t), color='g')#设置x轴log类型
ax2.set(title='semilogx')
ax2.grid()

# log x and y axis
ax3.loglog(t, 20 * np.exp(-t / 10.0), basex=2, color='k')#设置x、y两轴均为log类型
ax3.set(title='loglog base 2 on x')
ax3.grid()

# With errorbars: clip non-positive values
# Use new data for plotting
x = 10.0**np.linspace(0.0, 2.0, 20)
y = x**2.0

ax4.set_xscale("log", nonposx='clip')
ax4.set_yscale("log", nonposy='clip')
ax4.set(title='Errorbars go negative')
ax4.errorbar(x, y, xerr=0.1 * x, yerr=5.0 + 0.75 * y)
# ylim must be set after errorbar to allow errorbar to autoscale limits
ax4.set_ylim(ymin=0.1)

fig.tight_layout()

plt.savefig("logarithmic_figure_and_sublots_setting.jpg")
plt.show()
```

### 输出

![logarithmic](https://github.com/lyncodes/matplotlib-learning-experience/blob/master/logarithmic_figure_and_sublots_setting.jpg?raw=true)

## 饼状图 piechart

### 输入

```python
import matplotlib.pyplot as plt

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
#默认是逆时针方向
labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
sizes = [15, 30, 45, 10]#总共加起来100
explode = (0, 2, 0, 1)  #非零数字越大，扇形离圆心就越远
# only "explode" the 2nd slice (i.e. 'Hogs')，作为特别标记

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90, radius=15)
# autopct用于给各个扇形加上数字标签，显示具体份额
#label the wedges with their numeric value

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.savefig("piechart.jpg")
plt.show()
```

### 输出

![piechart](https://github.com/lyncodes/matplotlib-learning-experience/blob/master/piechart.jpg?raw=true)

## 散点图（frequently used）

###  输入

```python
import numpy as np
import matplotlib.pyplot as plt



N = 500
x = np.random.rand(N)#在0-1之间生成随机数
y = np.random.rand(N)
colors = np.random.rand(N)
area = np.pi * (15 * y)**2  # 0 to 15 point radii
#s:   size of points
plt.scatter(x, y, s=area, c=colors, alpha=0.5)#以x坐标的函数为点的大小，x越大，直径越大
plt.savefig("scatter.jpg")
plt.show()
```

### 输出

![scatter](https://github.com/lyncodes/matplotlib-learning-experience/blob/master/scatter.jpg?raw=true)如图所示，可以通过参数对scatter plot的points 大小进行设定。

## 画多张图subplot

###  输入

```python
import numpy as np
import matplotlib.pyplot as plt
#准备数学上的离散数据点
x1 = np.linspace(0,5)
x2 = np.linspace(0,2)

y1 = np.cos(2*np.pi*x1)*np.exp(-x1)
y2 = np.cos(2*np.pi*x2)

plt.subplot(2, 1, 1)
#####分成2x1的结构。2 rows，1 column，占用第一个位置
plt.plot(x1, y1, 'o-')
#以o-进行线的特性规划，o，表示圆圈，-表示以实线穿起来
plt.title('2 subplots')
plt.ylabel('damped oscillation')

plt.subplot(2, 1, 2)
#####分成2x1的结构。2 rows，1 column，占用第一个位置
plt.plot(x2, y2, '.-')
#以o-进行线的特性规划，.表示圆圈，-表示以实线穿起来
plt.xlabel('time (s)')
plt.ylabel('Undamped')


plt.savefig("subplots.jpg")
plt.show()
```

### 输出

![subplots](https://github.com/lyncodes/matplotlib-learning-experience/blob/master/subplots.jpg?raw=true)

## 时间序列分析

### 输入

```python
import datetime
'''
The datetime module supplies classes for 
manipulating dates and times in both simple
 and complex ways
'''
import numpy as np
import matplotlib.pyplot as plt
'''
matplotlib.pyplot is a state-based interface 
to matplotlib. It provides a MATLAB-like way 
of plotting.
'''
import matplotlib.dates as mdates
'''
Matplotlib provides sophisticated date plotting
 capabilities, standing on the shoulders of
  python datetime
'''
import matplotlib.cbook as cbook
'''
A collection of utility functions and classes
'''
years = mdates.YearLocator()
#在每一年的一月一号做tick
months = mdates.MonthLocator()
#给每一个月都做上tick
yearsFmt = mdates.DateFormatter('%Y')#按年份设置时间格式，此处精确到年份，最多可以精确到秒


# Load a numpy record array from yahoo csv data with fields date, open, close,
# volume, adj_close from the mpl-data/example directory. The record array
# stores the date as an np.datetime64 with a day unit ('D') in the date column.
with cbook.get_sample_data('goog.npz') as datafile:
	#NPZ file is a NumPy Zipped Data
    r = np.load(datafile)['price_data'].view(np.recarray)
# Matplotlib works better with datetime.datetime than np.datetime64, but the
# latter is more portable.
date = r.date.astype('O')

fig, ax = plt.subplots()
ax.plot(date, r.adj_close)####此时股价的图片已经画好了

# plt.show()
# exit()


# format the ticks
ax.xaxis.set_major_locator(years)#主要刻度精度
ax.xaxis.set_major_formatter(yearsFmt)#主要刻度的单位
ax.xaxis.set_minor_locator(months)#次要刻度精度

datemin = datetime.date(date.min().year, 1, 1)
#date函数，三个参数分别是年月日，返回一个date类型的object，代表时间
datemax = datetime.date(date.max().year + 1, 1, 1)
ax.set_xlim(datemin, datemax)#设置图画的x limitation


# format the coords message box
def price(x):
    return '$%1.2f' % x
ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
ax.format_ydata = price
ax.grid(True)

# rotates and right aligns the x labels, and moves the bottom of the
# axes up to make room for them
fig.autofmt_xdate()
#autofmt_xdate（）函数，用来自动分配ticklabels，免得各个ticlabels重叠

plt.title("the price")
plt.savefig("timeseries.jpg")
plt.show()
```

### 输出

![timeseries](https://github.com/lyncodes/matplotlib-learning-experience/blob/master/timeseries.jpg?raw=true)

**时间序列分析**要注意的是，x，y坐标的ticks的设置，注意间隔。



