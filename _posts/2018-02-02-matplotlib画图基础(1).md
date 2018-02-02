---
layout: post
title: "Matplotlib画图基础(1)"
author: "L-Y-N"
categories: matplotlib
tags: [matplotlib]
image: 2018-02-02-matplotlib_syntax1.png 
---

# Matplotlib画图基础(1)

## 条形图bar（）函数

### 输入1

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
#This module contains classes to support 
#completely configurable tick locating and formatting. 
#MaxNLocator:::Finds up to a max number of intervals with ticks at nice locations.
from collections import namedtuple
#collections是Python内建的一个集合模块
#namedtuple是一个函数，它用来创建一个自定义的tuple对象，
#并且规定了tuple元素的个数

n_groups = 5#总共会生成5组对比

means_men = (20, 35, 30, 35, 27)#平均值
std_men = (2, 3, 4, 1, 2)#标准差

means_women = (25, 32, 34, 20, 25)
std_women = (3, 5, 2, 3, 3)

fig, ax = plt.subplots()#未设置参数，画出来为一张单张

index = np.arange(n_groups)
bar_width = 0.3#设置bar的宽度

opacity = 0.8#透明度
error_config = {'ecolor': '0.3'}

rects1 = ax.bar(index, means_men, bar_width,
                alpha=opacity, color='g',
                yerr=std_men, error_kw=error_config,
                label='Men')
#index,x coordinates of the bars,长方条的位置，可以是数字，也可使是sequence
#由于index是一个sequences，所以后面的means_men也是sequence
#bar_width，bar宽度，默认是0.8
#yerr是y方向上，的errorbar，代表不确定度，uncertainty，即标准差
#error_kw是，errorbar的绘制参数，以字典形式传入
#label用于添加图例




#画了rects1后，再执行rects2，将直接在上面继续画图
rects2 = ax.bar(index + bar_width, means_women, bar_width,
                alpha=opacity, color='r',
                yerr=std_women, error_kw=error_config,
                label='Women')
##index+bar_width,加上barwidth，正好开始紧接着画



ax.set_xlabel('Group')
ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
ax.set_xticks(index + bar_width / 2)
#index+bar_width/2,将ticks们，设置在两个bar的中间，good idea啊
ax.set_xticklabels(('A', 'B', 'C', 'D', 'E'))
#给每一个tick这只一个名字
ax.legend()
#前面bar（）函数，label参数生成的图例，由legend函数把他们加上去

fig.tight_layout()#subplots时，将各个子图自动fit，在此处无用处
plt.savefig("barchart1.jpg")
plt.show()
```

### 输出1

![barchat1](https://github.com/lyncodes/matplotlib-learning-experience/blob/master/barchart1.jpg?raw=true)

### 输入2

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
#ticker模块用于设定刻度的位置和格式
#maxnlocator，找出ticks的最大个数
#
from collections import namedtuple
#namedtple创建一个和tuple类似的对象，而且对象可访问

#namedtple创建一个和tuple类似的对象，而且对象可访问
Student = namedtuple('Student', ['name', 'grade', 'gender'])


Score = namedtuple('Score', ['score', 'percentile'])

# GLOBAL CONSTANTS
testNames = ['Pacer Test', 'Flexed Arm\nHang', 'Mile Run', 'Agility',
             'Push Ups']
testMeta = dict(zip(testNames, ['laps', 'sec', 'min:sec', 'sec', '']))
#zip()函数，将两组sequence进行一一对应的封装
#见图中左右对应的测试项目和测试成绩对应的单位

def attach_ordinal(num):#给名次加上英文后缀
    """helper function to add ordinal string to integers

    1 -> 1st
    56 -> 56th
    """
    #ordinal序数的，排序的

    suffixes = dict((str(i), v) for i, v in#这里是for的多变量循环
                    enumerate(['th', 'st', 'nd', 'rd', 'th',
                               'th', 'th', 'th', 'th', 'th']))
    #print(suffixes),将会输出
    #{'0': 'th', '1': 'st', '2': 'nd', '3': 'rd', 
    #'4': 'th', '5': 'th', '6': 'th', '7': 'th',
    # '8': 'th', '9': 'th'}，这是一个末尾数字和相应的字母对应的字典
    #0 1 2 3 4 5 6 7 8 9，分别是enumerate（）函数自己产生的index数字
    #str（i）函数，把数字1-0，也转换成为string。再封装成为dictionary

    

    #给排名添加上后缀

    v = str(num)#将正儿八经的排名数字转换为str
    # special case early teens
    if v in {'11', '12', '13'}:
        return v + 'th'#英语语法，teens的年龄，11-13，加th
    return v + suffixes[v[-1]]#此时的v是一个str，从后面引用v的最后一个字符，得到末尾数字


def format_score(scr, test):#添加成绩标签，laps or seconds，或者没有标签

    #scr为返回的成绩数字，test为成绩的类型

    """
    Build up the score labels for the right Y-axis by first
    appending a carriage return to each string and then tacking on
    the appropriate meta information (i.e., 'laps' vs 'seconds'). We
    want the labels centered on the ticks, so if there is no meta
    info (like for pushups) then don't add the carriage return to
    the string
    """
    md = testMeta[test]
    #print(testMeta)
    #testMeta长这个样子，{'Pacer Test': 'laps', 'Flexed Arm Hang': 'sec',
    # 'Mile Run': 'min:sec', 'Agility': 'sec', 'Push Ups': ''}，一个字典

    if md:
        return '{0}\n{1}'.format(scr, md)#md就是分数score的单位
    else:
        return scr#score没有单位的时候，就直接返回其值

def format_ycursor(y):#提供y参数，y是测试项目的数量，返回相应的测试项目名称
    y = int(y)
    if y < 0 or y >= len(testNames):
        return ''
    else:
        return testNames[y]


def plot_student_results(student, scores, cohort_size):
    #对最终结果进行绘图
    #  create the figure
    fig, ax1 = plt.subplots(figsize=(8, 5))#设置图片大小,ax1只有一张子图

    pos = np.arange(len(testNames))#由testName的容量大小，设定pos的位置
    #pos用于精确放置标签位置

    rects = ax1.barh(pos, [scores[k].percentile for k in testNames],
                     align='center',
                     height=0.5, color='b',
                     tick_label=testNames)
    #[scores[k].percentile for k in testNames]是数据来源

    ax1.set_title(student.name)#学生的名字，再后面会传入参数

    ax1.set_xlim([0, 100])#设定x轴的最大值
    ax1.xaxis.set_major_locator(MaxNLocator(11))#MaxNLocator设定ticks的最大个数
    
    ax1.xaxis.grid(True, linestyle='--', which='major',
                   color='r', alpha=.5)#图片网格的相关设置，gridline
    # Plot a solid vertical gridline to highlight the median position

    ax1.axvline(50, color='grey', alpha=0.25)
    # set X-axis tick marks at the deciles
    cohort_label = ax1.text(.5, -.07, 'Cohort Size: {0}'.format(cohort_size),
                            horizontalalignment='center', size='small',
                            transform=ax1.transAxes)



    # Set the right-hand Y-axis ticks and labels
    ax2 = ax1.twinx()#Create a twin Axes sharing the xaxis，建立第二个坐标轴

    scoreLabels = [format_score(scores[k].score, k) for k in testNames]
    #调用前面的format_score()函数，生成成绩标签，k是成绩项目的名字，

    # set the tick locations
    ax2.set_yticks(pos)#pos是位置
    # make sure that the limits are set equally on both yaxis so the
    # ticks line up
    ax2.set_ylim(ax1.get_ylim())
    #get_ylim()自动返回y轴的界限所在，as the tuple (bottom, top).



    # set the tick labels
    ax2.set_yticklabels(scoreLabels)#116行处，scorelabels是已经生成的成绩标签，现在将其放上图片

    ax2.set_ylabel('Test Scores')

    ax2.set_xlabel(('Percentile Ranking Across '
                    '{grade} Grade {gender}s').format(
                        grade=attach_ordinal(student.grade),
                        gender=student.gender.title()))

    rect_labels = []
    # Lastly, write in the ranking inside each bar to aid in interpretation
    for rect in rects:
        # Rectangle widths are already integer-valued but are floating
        # type, so it helps to remove the trailing decimal point and 0 by
        # converting width to int type
        width = int(rect.get_width())

        rankStr = attach_ordinal(width)
        # The bars aren't wide enough to print the ranking inside
        if (width < 5):
            # Shift the text to the right side of the right edge
            xloc = width + 1
            # Black against white background
            clr = 'black'
            align = 'left'
        else:
            # Shift the text to the left side of the right edge
            xloc = 0.98*width
            # White on magenta
            clr = 'white'
            align = 'right'

        # Center the text vertically in the bar
        yloc = rect.get_y() + rect.get_height()/2.0
        label = ax1.text(xloc, yloc, rankStr, horizontalalignment=align,
                         verticalalignment='center', color=clr, weight='bold',
                         clip_on=True)
        rect_labels.append(label)

    # make the interactive mouse over give the bar title
    ax2.fmt_ydata = format_ycursor
    # return all of the artists created
    return {'fig': fig,
            'ax': ax1,
            'ax_right': ax2,
            'bars': rects,
            'perc_labels': rect_labels,
            'cohort_label': cohort_label}

student = Student('Johnny Doe', 2, 'boy')#对前面的namedtuple进行初始化
scores = dict(zip(testNames,(Score(v, p) 
    for v, p in zip(['7', '48', '12:52', '17', '14'],#这个是成绩，绝对成绩
        np.round(np.random.uniform(0, 1,len(testNames))*100, 0)))))
cohort_size = 62  # The number of other 2nd grade boys
#总参加测试人数

arts = plot_student_results(student, scores, cohort_size)#调用绘图函数
#student是一个用tuple初始化好的tuple


plt.savefig("barchart2.jpg")
plt.show()
```

### 输出2

![barchart2](https://github.com/lyncodes/matplotlib-learning-experience/blob/master/barchart2.jpg?raw=true)

## 填充图

### 输入

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 500)
y = np.sin(4 * np.pi * x) * np.exp(-5 * x)

fig, ax = plt.subplots()

ax.fill(x, y, zorder=5.1)

####这个zorder啊，是表示一个状态，fill的zorder比较大，所以后渲染

#####fill()函数，画出filled polygons


ax.grid(True, zorder=5)#grid的zorder表较小，所以先渲染
#这样，我们看到的图就是填充好的色块在网格的上方
plt.title("color block is above the grid")



x = np.linspace(0, 2 * np.pi, 500)
y1 = np.sin(x)
y2 = np.sin(3 * x)

fig, ax = plt.subplots()
ax.fill(x, y1, 'b', x, y2, 'r', alpha=0.5, zorder=1)
ax.grid(True, zorder=2)
plt.title("color block is under the grid")


plt.savefig("filled plot.jpg")
plt.show()
```

### 输出

![filled plot](https://github.com/lyncodes/matplotlib-learning-experience/blob/master/filled%20plot.jpg?raw=true)

## 互动性图像 (官方说法叫widget)

### 输入

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
#slider是滑块选项，button用于设置reset


# RadioButton 控件为用户提供由两个或多个互斥选项组成的选项集
#即是左侧的red、blue、green三个选项

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
t = np.arange(0.0, 1.0, 0.001)
a0 = 5
f0 = 3
s = a0*np.sin(2*np.pi*f0*t)#三角函数标准表达式Asin(wx+fai)


l, = plt.plot(t, s, lw=1, color='red')#lw is linewodth
####为什么l后面要跟一个空格呢？？


plt.axis([0, 1, -10, 10])

axcolor = 'g'#slider的颜色
axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
#facecolor is background color
#rect = [left, bottom, width, height] 
axamp = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

sfreq = Slider(axfreq, 'Freq', 0.1, 30.0, valinit=f0)#f0,a0是初始位置
samp = Slider(axamp, 'Amp', 0.1, 10.0, valinit=a0)
#两个数字数slider的左右界限
#valinit是silde的初始位置

def update(val):#这个update用于，silder移动的时候，向绘图函数实时update参数
    amp = samp.val
    freq = sfreq.val
    l.set_ydata(amp*np.sin(2*np.pi*freq*t))
    fig.canvas.draw_idle()

sfreq.on_changed(update)
samp.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])#设置reset buttom的位置
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.95')


def reset(event):
    sfreq.reset()
    samp.reset()
button.on_clicked(reset)

rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)#设置radiobuttom
radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)


def colorfunc(label):
    l.set_color(label)
    fig.canvas.draw_idle()


radio.on_clicked(colorfunc)


plt.savefig("gui widgets.jpg")
plt.show()
```

### 输出

![widgets](https://github.com/lyncodes/matplotlib-learning-experience/blob/master/gui%20widgets.jpg?raw=true)

这里不能够展现冬天的图片，要使用，需要自行运行以上程序。

## 条形统计图histogram

### 输入

```python
import numpy as np

import matplotlib.pyplot as plt

np.random.seed()
######seed()函数，用一套特殊的算法，生成一系列随机数
##不给参数的话，seed（）函数以系统时间为参数生成随机数，这样每次生成的随机数则不同
#如果给定seed（）一个相同的参数，则每次生成的随机数是一样的


#生成数据
mean = 100
sigma = 15
rand = np.random.randn(500)#每次运行的值都不一样####
x = mean + sigma*rand#randn()函数，参数为几个，则生成几个数
#randn（3，4）则会生成array



#设定直方分布图参数
bins_num = 50#五十个条条
fig, ax = plt.subplots()#画板在此
n, bins, pathes = ax.hist(x, bins_num)#normed越大，柱子越窄
plt.savefig("hist")
plt.show()#画出来的图每次也不一样


```

### 输出

![histogram](https://github.com/lyncodes/matplotlib-learning-experience/blob/master/hist.png?raw=true)

## 添加标签

### 输入

```python
import numpy as np
import matplotlib.pyplot as plt

# Make some fake data.
a = b = np.arange(0, 3, .02)
c = np.exp(a)
d = c[::-1]#c的索引步长为-1，即从尾开始索引，即倒序

# Create plots with pre-defined labels.
fig, ax = plt.subplots()
ax.plot(a, c, 'k--', label='Model length')#--线形,------------并且添加上了label
ax.plot(a, d, 'g:', label='Data length')#：线形
ax.plot(a, c + d, 'r', label='Total message length')#不带符号，为实线

legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')

# Put a nicer background color on the legend.
legend.get_frame().set_facecolor('#00FFCC')#hex color
#get_frame()函数返回一个rectangle，用于放置三个label，即是中间哪个绿色的方块
plt.savefig("legends.jpg")
plt.show()
```

### 输出

![legends](https://github.com/lyncodes/matplotlib-learning-experience/blob/master/legends.jpg?raw=true)

由此可见利用legend函数，对不同的线型、颜色、粗细进行调整，并添加上标签。



