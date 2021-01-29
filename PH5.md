---
title: Persistent-Homology-5
tag: TDA-PH
categories: TDA-PH
---

[![yPnSHO.jpg](https://s3.ax1x.com/2021/01/29/yPnSHO.jpg)](https://imgchr.com/i/yPnSHO)

------

> https://zhuanlan.zhihu.com/p/41278774

------

在这一部分中，我们终于要利用我们所学的所有知识来计算持续同调群，并绘制持续性图来图像化地总结信息。



让我们来总结一下我们目前所知道的。

1. 如何使用任意距离参数 ![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon) 的点云数据生成一个单纯复形。
2. 如何计算单纯复形的同调群
3. 如何计算单纯复形的连通数

从我们知道的到单纯同调的跨度从概念上说很小。我们只需要计算连续改变的 ![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon%3A+0+%5Crightarrow+%5Cinfty) 生成的一系列单纯复形集合的连通数。然后，我们可以看到哪些拓扑特性持续时间明显比其他的长，说明它们是信号而不是噪声。

> 注意：我忽略了“持续时间”的客观定义，因为这是一个统计问题，超出了这个阐述的范围。对于我们在这里所考虑的所有例子，很明显，只需要目测就知道它们持续时间的长短。

不幸的是，概念上的跨度虽然很小，但技术上的跨度却很大。特别是如果我们还想问最初的数据集中是那些数据点发出了这些持续性的拓扑信号，从而构成这个数据集空间特定的拓扑特性。

让我们重新回顾一下我们在构建圆圈而(带有一些随机性)采样的数据点以及构建简单复形的代码。

```text
import numpy as np
import matplotlib.pyplot as plt

n = 30 #number of points to generate

#generate space of parameter
theta = np.linspace(0, 2.0*np.pi, n) 

a, b, r = 0.0, 0.0, 5.0

x = a + r*np.cos(theta)
y = b + r*np.sin(theta)

#code to plot the circle for visualization
plt.plot(x, y)
plt.show()
```

![img](https://pic2.zhimg.com/80/v2-7bbad736dc77b2d3657363d3da46115d_1440w.jpg)

```python3
x2 = np.random.uniform(-0.75,0.75,n) + x #add some "jitteriness" to the points
y2 = np.random.uniform(-0.75,0.75,n) + y
fig, ax = plt.subplots()
ax.scatter(x2,y2)
plt.show()
```



![img](https://pic3.zhimg.com/80/v2-aa25f6d9387e99d13333eff02cf71a66_1440w.jpg)

```text
newData = np.array(list(zip(x2,y2)))
import SimplicialComplex
```



```text
graph = SimplicialComplex.buildGraph(raw_data=newData, epsilon=3.0) #Notice the epsilon parameter is 3.0
ripsComplex = SimplicialComplex.rips(graph=graph, k=3)
SimplicialComplex.drawComplex(origData=newData, ripsComplex=ripsComplex)
```



![img](https://pic1.zhimg.com/80/v2-eb3c753510b474ce7d90b0d77106cf7c_1440w.jpg)

如你所见，设置 ![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon+%3D+3.0) 生成一个还不错的单纯复形，它捕获了原始数据的1维“孔”。

然而，让我们改变 ![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon) ，看看它会怎么改变我们的复形。

```python3
graph = SimplicialComplex.buildGraph(raw_data=newData, epsilon=2.0)
ripsComplex = SimplicialComplex.rips(graph=graph, k=3)
SimplicialComplex.drawComplex(origData=newData, ripsComplex=ripsComplex)
```



![img](https://pic1.zhimg.com/80/v2-a443d78de424fca607f21fda83d1a004_1440w.jpg)

我们把 ![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon) 减到 ![[公式]](https://www.zhihu.com/equation?tex=2.0) ，现在我们的圆有了一个“破口”。如果我们计算这个复形的同调和连通数，我们将不再有一个 ![[公式]](https://www.zhihu.com/equation?tex=1) 维圈圈。我们将只看到一个连接的组件。



我们再把它减到 ![[公式]](https://www.zhihu.com/equation?tex=1.9) 。

```text
newData = np.array(list(zip(x2,y2)))
graph = SimplicialComplex.buildGraph(raw_data=newData, epsilon=1.9)
ripsComplex = SimplicialComplex.rips(graph=graph, k=3)
SimplicialComplex.drawComplex(origData=newData, ripsComplex=ripsComplex)
```



![img](https://pic1.zhimg.com/80/v2-a593fc74e1f406e31d41f3b8ca2e1774_1440w.jpg)

现在我们有三个连接的组件，复形中没有圈圈/孔洞。OK，让我们试试把![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon)加到![[公式]](https://www.zhihu.com/equation?tex=4.0)。

```text
newData = np.array(list(zip(x2,y2)))
graph = SimplicialComplex.buildGraph(raw_data=newData, epsilon=4.0)
ripsComplex = SimplicialComplex.rips(graph=graph, k=3)
SimplicialComplex.drawComplex(origData=newData, ripsComplex=ripsComplex)
```



![img](https://pic4.zhimg.com/80/v2-05eee9fb830b1a87ca101cec17fa5cb7_1440w.jpg)

和把半径参数 ![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon) 减降 ![[公式]](https://www.zhihu.com/equation?tex=1) 个单位不同，把 ![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon) 加到 ![[公式]](https://www.zhihu.com/equation?tex=4.0) ，我们没有改变我们的同调群。我们仍然有一个连接的组件和一个 ![[公式]](https://www.zhihu.com/equation?tex=1) 维的圆环。

让我们直接把 ![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon) 加到 ![[公式]](https://www.zhihu.com/equation?tex=7.0) 。

```text
graph = SimplicialComplex.buildGraph(raw_data=newData, epsilon=7.0)
ripsComplex = SimplicialComplex.rips(graph=graph, k=3)
SimplicialComplex.drawComplex(origData=newData, ripsComplex=ripsComplex)
```



![img](https://pic1.zhimg.com/80/v2-7e0bb86590937b02025af096a9fb8468_1440w.jpg)

好吧，尽管我们已经给原来的 ![[公式]](https://www.zhihu.com/equation?tex=3.0) 加了 ![[公式]](https://www.zhihu.com/equation?tex=4) 个单位量，但我仍然得到一个同样拓扑特性的复形：一个连接的组件和一个 ![[公式]](https://www.zhihu.com/equation?tex=1) 维的圆环。

这是在持续同调中持续的拓扑量化分析所能获得的洞见。由于这些观测到的特征在大范围的 ![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon) 中是持续存在的，因此它们很可能是底层数据的真实特征，而不是噪音。

我们可以用两种主要的可视化来展示我们的发现：条形码图和持续性图。下面是上面例子的条形码图：

![img](https://pic4.zhimg.com/80/v2-6a6558ea201374c2f0692605437ddc8b_1440w.jpg)

> **注意**：我通过”分段式“的方式完成了条码图，也就是说，它不是精确计算的条码图。我在噪音中标亮了“真正”的拓扑特征。 ![[公式]](https://www.zhihu.com/equation?tex=H_0%2C+H_1%2C+H_2) 指的是各维度的同调群和连通数的情况。



重要的是，可能有两种不同的真实拓扑特征在不同的尺度上存在，因此只用一个持续同调来捕捉，它可能会被单一尺度下的单纯复形忽略。例如，如果数据样子是大圆旁边有小圆，可能 ![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon) 较小时，只有小圆会被连接，产生一个 ![[公式]](https://www.zhihu.com/equation?tex=1) 维孔洞，然后在 ![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon) 较大时，大圈会被连接，而小圈会被“填满”。所以没有适合的 ![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon) 值能够显示出两个圈。



## 滤流

原来有一种相对直接的方法来扩展我们之前计算边界矩阵的连通数的工作，以及处理不断膨胀的持续同调的集合。

我们定义 ***复形滤流\*** 是由不断增加的比例参数 ![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon) 生成的单纯复形的序列。

用滤流的方式来处理持续性计算的精要在于，与其构建多个不同的 ![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon) 参数的单纯复形，然后把它们组合成一个序列，不如在我们的数据上用一个大（最大）的 ![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon) 值构建一个简单复形。然后我们会记录所有点两两之间的距离(我们已经用我们写的算法做过)，所以我们知道什么 ![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon) 值会使每对点形成边。因此在各个 ![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon) 值中隐藏着的所有单纯复形构成了一个连贯的滤流（或者说，一系列嵌套的复形）。

下面有一个非常简单的例子：



![img](https://pic2.zhimg.com/80/v2-ba89c8050aa5db98a0714f4c948f98a5_1440w.jpg)



所以如果我们取最大值， ![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon%3D4) ，我们的单纯复形是：

![[公式]](https://www.zhihu.com/equation?tex=%5C%5C+S+%3D+%5Ctext%7B+%7B+%7B0%7D%2C+%7B1%7D%2C+%7B2%7D%2C+%7B0%2C1%7D%2C+%7B2%2C0%7D%2C+%7B1%2C2%7D%2C+%7B0%2C1%2C2%7D+%7D+%7D)

但如果我们追踪点与点之间的距离（即所有边的长度/比重），那么就会得到足够的信息来刻画需要的滤流。

下面是该单纯复形（竖线表示比重/长度）每条边（ ![[公式]](https://www.zhihu.com/equation?tex=1) 维）的比重（长度）：

![[公式]](https://www.zhihu.com/equation?tex=%5C%5C+%7C%7B0%2C1%7D%7C+%3D+1.4+%5C%5C+%7C%7B2%2C0%7D%7C+%3D+2.2+%5C%5C+%7C%7B1%2C2%7D%7C+%3D+3)

而这是我们用这些信息构建滤流的方法：



![[公式]](https://www.zhihu.com/equation?tex=%5C%5CS_0+%5Csubseteq+S_1+%5Csubseteq+S_2+%5C%5C+S_0+%3D+%5Ctext%7B+%7B+%7B0%7D%2C+%7B1%7D%2C+%7B2%7D+%7D+%7D+%5C%5C+S_1+%3D+%5Ctext%7B+%7B+%7B0%7D%2C+%7B1%7D%2C+%7B2%7D%2C+%7B0%2C1%7D+%7D+%7D+%5C%5C+S_2+%3D+%5Ctext%7B+%7B+%7B0%7D%2C+%7B1%7D%2C+%7B2%7D%2C+%7B0%2C1%7D%2C+%7B2%2C0%7D%2C+%7B1%2C2%7D%2C+%7B0%2C1%2C2%7D+%7D+%7D+%5C%5C)

基本上，当它最长的边出现时，子复形中每一个单形的域流都将出现。所以 ![[公式]](https://www.zhihu.com/equation?tex=2) 维复形 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7B0%2C1%2C2%5C%7D) 只会在边 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7B1%2C2%5C%7D) 出现的时候出现，因为那条边是最长的。

为了让它变成滤流，从而用在我们（将来）的算法中，它需要有 **全序**。**全序** 是在滤流中单形根据“小于”关系（即任意两个单形的“值”都不等）的排序。最常见的关于集合全序的例子来自自然数集 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7B0%2C1%2C2%2C3%2C4%2C...%5C%7D) ，因为没有两个数是相等的，我们总是可以说一个数比另一个数大或者小。

我们怎么确定单形在滤流中的“值”（滤值）（从而确定域流的顺序）呢？我之前就说过了。单形的滤值部分取决于最长边的长度。但有时两种不同的单形的最长边是一样长的，所此我们必须定义一种规则来确定单形的值(顺序)。



对于任意两个单形 ![[公式]](https://www.zhihu.com/equation?tex=%5Csigma_1%2C+%5Csigma_2) ：

1. ![[公式]](https://www.zhihu.com/equation?tex=0) 维单形必须比 ![[公式]](https://www.zhihu.com/equation?tex=1) 维单形少， ![[公式]](https://www.zhihu.com/equation?tex=1) 维单形必须比 ![[公式]](https://www.zhihu.com/equation?tex=2) 维单形少，以此类推。这意味着单形的任意面（即 ![[公式]](https://www.zhihu.com/equation?tex=f+%5Csubset+%5Csigma) ）自动小于（按顺序之前）单形。即如果 ![[公式]](https://www.zhihu.com/equation?tex=dim%28%5Csigma_1%29+%3C+dim%28%5Csigma_2%29+%5Cimplies+%5Csigma_1+%3C+%5Csigma_2) （ ![[公式]](https://www.zhihu.com/equation?tex=dim+%3D+) 维度，符号 ![[公式]](https://www.zhihu.com/equation?tex=%5Cimplies) 意思是“意味着”）
2. 如果 ![[公式]](https://www.zhihu.com/equation?tex=%5Csigma_1%2C+%5Csigma_2) 的维度相等（因此彼此都不是另一个的面），那么每个单形的值由它最长（比重最大）的 ![[公式]](https://www.zhihu.com/equation?tex=1) 维单形（边）决定。在我们上面的例子中， ![[公式]](https://www.zhihu.com/equation?tex=%5C%7B0%2C1%5C%7D+%5Clt+%5C%7B2%2C0%5C%7D+%5Clt+%5C%7B1%2C2%5C%7D) 取决于它们各自的比重。为了比较高维单形，你仍旧只需要通过对比它们最大边的值比较它们。即如果 ![[公式]](https://www.zhihu.com/equation?tex=dim%28%5Csigma_1%29+%3D+dim%28%5Csigma_2%29) ，那么 ![[公式]](https://www.zhihu.com/equation?tex=max%5C_edge%28%5Csigma_1%29+%3C+max%5C_edge%28%5Csigma_2%29+%5Cimplies+%5Csigma_1+%3C+%5Csigma_2) 。

\3. 如果 ![[公式]](https://www.zhihu.com/equation?tex=%5Csigma_1%2C+%5Csigma_2) 具有相同维度，而它们的最长边也相等（即它们最长边在相同 ![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon) 值进入域流），那么 ![[公式]](https://www.zhihu.com/equation?tex=max%5C_vertex%28%5Csigma_1%29+%3C+max%5C_vertex%28%5Csigma_2%29+%5Cimplies+%5Csigma_1+%3C+%5Csigma_2) 。什么是最大结点？我们只需要任意给结点排序，即使它们同时出现。

> 顺便说一下，我们刚才讨论了全序。这个想法的推论是偏序，我们在一些但并非所有元素之间定义了“少于”关系，而另一些元素可能和其他相等。

记得第三章时我们讲过如何通过设置列来表示 ![[公式]](https://www.zhihu.com/equation?tex=n) 维链群中n维单形，以及用行表示 ![[公式]](https://www.zhihu.com/equation?tex=%EF%BC%88n-1%EF%BC%89) 链群中的 ![[公式]](https://www.zhihu.com/equation?tex=%EF%BC%88n-1%EF%BC%89) 维单形，来设置边界矩阵？我们可以用下面的方法来扩展这个过程以此通过整个域流复形计算连通数。

让我们用上面的域流：



![[公式]](https://www.zhihu.com/equation?tex=%5C%5CS_0+%5Csubseteq+S_1+%5Csubseteq+S_2+%5C%5C+S_0+%3D+%5Ctext%7B+%5B+%7B0%7D%2C+%7B1%7D%2C+%7B2%7D+%7D+%5D+%5C%5C+S_1+%3D+%5Ctext%7B+%5B+%7B0%7D%2C+%7B1%7D%2C+%7B2%7D%2C+%7B0%2C1%7D+%5D+%7D+%5C%5C+S_2+%3D+S+%3D+%5Ctext%7B+%5B+%7B0%7D%2C+%7B1%7D%2C+%7B2%7D%2C+%7B0%2C1%7D%2C+%7B2%2C0%7D%2C+%7B1%2C2%7D%2C+%7B0%2C1%2C2%7D+%5D+%7D+%5C%5C)

注意我已经用方括号表示在每个子复形按顺序（我在单形的集合中加入了全序）的域流中的单形。

因此，我们将以与我们之前维每个同源群建立各自的边界矩阵相同的方式为整个域流建立一个边界矩阵。

然后，像之前一样，我们让每个单元 ![[公式]](https://www.zhihu.com/equation?tex=%5Bi%2Cj%5D%3D1) 如果 ![[公式]](https://www.zhihu.com/equation?tex=%5Csigma_i) 是 ![[公式]](https://www.zhihu.com/equation?tex=%5Csigma_j) 的面。其他单元为 ![[公式]](https://www.zhihu.com/equation?tex=0) 。



这是上面的域流中它的样子:

![[公式]](https://www.zhihu.com/equation?tex=%5C%5C%5Cpartial_%7Bfiltration%7D+%3D+%5Cbegin%7Barray%7D%7Bc%7Clcr%7D+%5Cpartial+%26+%5C%7B0%5C%7D+%26+%5C%7B1%5C%7D+%26+%5C%7B2%5C%7D+%26+%5C%7B0%2C1%5C%7D+%26+%5C%7B2%2C0%5C%7D+%26+%5C%7B1%2C2%5C%7D+%26+%5C%7B0%2C1%2C2%5C%7D+%5C%5C+%5Chline+%5C%7B0%5C%7D+%26+0+%26+0+%26+0+%26+1+%26+1+%26+0+%26+0+%5C%5C+%5C%7B1%5C%7D+%26+0+%26+0+%26+0+%26+1+%26+0+%26+1+%26+0+%5C%5C+%5C%7B2%5C%7D+%26+0+%26+0+%26+0+%26+0+%26+1+%26+1+%26+0+%5C%5C+%5C%7B0%2C1%5C%7D+%26+0+%26+0+%26+0+%26+0+%26+0+%26+0+%26+1+%5C%5C+%5C%7B2%2C0%5C%7D+%26+0+%26+0+%26+0+%26+0+%26+0+%26+0+%26+1+%5C%5C+%5C%7B1%2C2%5C%7D+%26+0+%26+0+%26+0+%26+0+%26+0+%26+0+%26+1+%5C%5C+%5C%7B0%2C1%2C2%5C%7D+%26+0+%26+0+%26+0+%26+0+%26+0+%26+0+%26+0+%5C%5C+%5Cend%7Barray%7D)

正如上面，我们将用一个算法来改变这个矩阵的形式。然而，不像从前，我们将这个边界矩阵转换为 **Smith** 标准型，我们现在要把它变成另一种叫列阶梯型。这个转换过程叫做**矩阵化减（Matrix Reduction）**，这意味着我们把它简化成一个简单的形式。



## **矩阵化减**

现在，我要为我在上一篇文章中犯下的错误道歉，因为我从来没有解释过为什么要把边界矩阵转换为 **Smith** 标准型，我只说了怎么做。

我们之前的的边界矩阵给了我们一个从 ![[公式]](https://www.zhihu.com/equation?tex=n) 维链群到 ![[公式]](https://www.zhihu.com/equation?tex=%EF%BC%88n-1%EF%BC%89) 维链群的线性映射。我们可以用 ![[公式]](https://www.zhihu.com/equation?tex=n) 链中任意元素乘以边界矩阵，结果得到 ![[公式]](https://www.zhihu.com/equation?tex=%EF%BC%88n-1%EF%BC%89) 链中对应（映射）的元素。当我们将矩阵简化为 **Smith** 标准型时，我们改变了边界矩阵，这样我们就不能再乘以它来映射元素了。我们所做的实际上是在边界矩阵上应用另一个线性映射，结果是 **Smith** 标准型。

更正式地讲，矩阵 ![[公式]](https://www.zhihu.com/equation?tex=A) 的 **Smith** 标准型 ![[公式]](https://www.zhihu.com/equation?tex=R) 是矩阵乘积： ![[公式]](https://www.zhihu.com/equation?tex=R%3DSAT) ，其中 ![[公式]](https://www.zhihu.com/equation?tex=S%2CT) 是其它矩阵。因此我们有构成 ![[公式]](https://www.zhihu.com/equation?tex=R) 的线性映射，以及我们原则上可以分解 ![[公式]](https://www.zhihu.com/equation?tex=R) 进入组成它单独的线性映射中。

因此，减少到 **Smith** 标准型的算法本质上是找到另外两个矩阵 ![[公式]](https://www.zhihu.com/equation?tex=S%2CT) ，使得 ![[公式]](https://www.zhihu.com/equation?tex=SAT) 生成一个沿着对角线以 ![[公式]](https://www.zhihu.com/equation?tex=1) 作为元组成的矩阵。

但为什么我们要这么做？记住，矩阵作为线性映射意味着它将一个向量空间映射到另一个向量空间。如果我们有矩阵 ![[公式]](https://www.zhihu.com/equation?tex=M%3A+V_1+%5Crightarrow+V_2) ，那么它映射基向量 ![[公式]](https://www.zhihu.com/equation?tex=V_1) 到基向量 ![[公式]](https://www.zhihu.com/equation?tex=S_2) 。所以当我们减化矩阵，本质上，我们重新定义了每个向量空间中的基向量。正是这样，**Smith** 标准型找到了形成环和边界的基础。有许多不同类型的减化矩阵的形式，它们拥有有用的解释和性质。我不打算再讲数学了，我只是想稍微解释一下我们正在做的矩阵化减。

当我们通过算法把滤流边界矩阵变成列阶梯型，它告诉我们，在每个维度上有特定的拓扑特征是在域流的不同阶段形成还是“消亡”（通过被归入更显著的特征）（即增加 ![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon) 的值，通过全序在域流中表示）。因此，一旦我们减化了边界矩阵，我们所需要做的就是在特性形成或消亡的时候将信息作为间隔读取，然后我们可以将这些间隔绘制成条形码图。

列阶梯型 ![[公式]](https://www.zhihu.com/equation?tex=C) 同样是线性映射的组成，正如 ![[公式]](https://www.zhihu.com/equation?tex=C%3DVB) ，其中 ![[公式]](https://www.zhihu.com/equation?tex=V) 是一些使组成有效的矩阵， ![[公式]](https://www.zhihu.com/equation?tex=B) 是滤流边界矩阵。事实上一旦我们减化了 ![[公式]](https://www.zhihu.com/equation?tex=B) ，我们仍将保持 ![[公式]](https://www.zhihu.com/equation?tex=V) 的副本，因为 ![[公式]](https://www.zhihu.com/equation?tex=V) 记录了需要的信息，从而让我们知道那些数据点构成了该数据点云空间里让我们感兴趣的拓扑特征。

将矩阵转化成列阶梯型的普适性方法是高斯消去法的一种：

```text
for j = 1 to n
    while there exists i < j with low(i) = low(j) 
        add column i to column j
    end while
end for
```



当中的函数 **low** 输入 ![[公式]](https://www.zhihu.com/equation?tex=j) 列返回最低的 ![[公式]](https://www.zhihu.com/equation?tex=1) 的行索引。举个例子，如果我们有矩阵的一行：

![[公式]](https://www.zhihu.com/equation?tex=j+%3D+%5Cbegin%7Bpmatrix%7D+1+%5C%5C+0+%5C%5C1+%5C%5C1+%5C%5C+0+%5C%5C0+%5C%5C0+%5Cend%7Bpmatrix%7D)

那么 **low**( ![[公式]](https://www.zhihu.com/equation?tex=j) ) ![[公式]](https://www.zhihu.com/equation?tex=%3D3) （索引从 ![[公式]](https://www.zhihu.com/equation?tex=0) 开始）因为列中最低的 ![[公式]](https://www.zhihu.com/equation?tex=1) 在第 ![[公式]](https://www.zhihu.com/equation?tex=4) 行（索引为 ![[公式]](https://www.zhihu.com/equation?tex=3) ）。



基本上，算法从左到右扫描矩阵中的每一列，所以如果我们现在在 ![[公式]](https://www.zhihu.com/equation?tex=j) 列，算法会找所有在 ![[公式]](https://www.zhihu.com/equation?tex=j) 列前的 ![[公式]](https://www.zhihu.com/equation?tex=i) 列使得 **low**( ![[公式]](https://www.zhihu.com/equation?tex=i) ) ![[公式]](https://www.zhihu.com/equation?tex=%3D%3D) **low**( ![[公式]](https://www.zhihu.com/equation?tex=j) )，而如果它找到了这样一列 ![[公式]](https://www.zhihu.com/equation?tex=i) ，它会把那一列加到 ![[公式]](https://www.zhihu.com/equation?tex=j) 中。我们还会在另一个矩阵中记录每次我们把一列加到另一列的过程。如果某一列全是 ![[公式]](https://www.zhihu.com/equation?tex=0) ，那么**low**( ![[公式]](https://www.zhihu.com/equation?tex=j)) ![[公式]](https://www.zhihu.com/equation?tex=%3D-1) （代表未定义）。



让我们用上面的边界矩阵手动计算，尝试这个算法。我已经删除了列/行标签，以便更简洁：



![[公式]](https://www.zhihu.com/equation?tex=%5C%5C%5Cpartial_%7Bfiltration%7D+%3D+%5Cbegin%7BBmatrix%7D+0+%26+0+%26+0+%26+1+%26+1+%26+0+%26+0+%5C%5C+0+%26+0+%26+0+%26+1+%26+0+%26+1+%26+0+%5C%5C+0+%26+0+%26+0+%26+0+%26+1+%26+1+%26+0+%5C%5C+0+%26+0+%26+0+%26+0+%26+0+%26+0+%26+1+%5C%5C+0+%26+0+%26+0+%26+0+%26+0+%26+0+%26+1+%5C%5C+0+%26+0+%26+0+%26+0+%26+0+%26+0+%26+1+%5C%5C+0+%26+0+%26+0+%26+0+%26+0+%26+0+%26+0+%5C%5C+%5Cend%7BBmatrix%7D)



所以记住，列是 ![[公式]](https://www.zhihu.com/equation?tex=j) ，行是 ![[公式]](https://www.zhihu.com/equation?tex=i) 。我们从左到右扫描。前三列全是 ![[公式]](https://www.zhihu.com/equation?tex=0) ，所以 **low**( ![[公式]](https://www.zhihu.com/equation?tex=j) ) 没有被定义，我们什么都不用做。而当我们到了第四列（![[公式]](https://www.zhihu.com/equation?tex=j%3D3) ），因为之前所有列都是 ![[公式]](https://www.zhihu.com/equation?tex=0) ，所以我们还是什么都不做。当我们到第五列（ ![[公式]](https://www.zhihu.com/equation?tex=j%3D4) ），那么 **low**( ![[公式]](https://www.zhihu.com/equation?tex=4) ) ![[公式]](https://www.zhihu.com/equation?tex=%3D2) 而 **low**( ![[公式]](https://www.zhihu.com/equation?tex=3) ) ![[公式]](https://www.zhihu.com/equation?tex=%3D1) ，因为**low**( ![[公式]](https://www.zhihu.com/equation?tex=4))!=**low**( ![[公式]](https://www.zhihu.com/equation?tex=3) )，我们不用做什么。而到了第六列（ ![[公式]](https://www.zhihu.com/equation?tex=j%3D5) ），存在一列 ![[公式]](https://www.zhihu.com/equation?tex=i+%3C+j) （在这个情况下，列 ![[公式]](https://www.zhihu.com/equation?tex=4%3C5) ），而且 **low**( ![[公式]](https://www.zhihu.com/equation?tex=4) )=**low**( ![[公式]](https://www.zhihu.com/equation?tex=5) )。所以我们把 ![[公式]](https://www.zhihu.com/equation?tex=5) 列加到 ![[公式]](https://www.zhihu.com/equation?tex=6) 列。因为这是二进制列， ![[公式]](https://www.zhihu.com/equation?tex=1%2B1%3D0) 。把 ![[公式]](https://www.zhihu.com/equation?tex=5) 列加到 ![[公式]](https://www.zhihu.com/equation?tex=6) 列的结果如下：



![[公式]](https://www.zhihu.com/equation?tex=%5C%5C+%5Cpartial_%7Bfiltration%7D+%3D+%5Cbegin%7BBmatrix%7D+0+%26+0+%26+0+%26+1+%26+1+%26+1+%26+0+%5C%5C+0+%26+0+%26+0+%26+1+%26+0+%26+1+%26+0+%5C%5C+0+%26+0+%26+0+%26+0+%26+1+%26+0+%26+0+%5C%5C+0+%26+0+%26+0+%26+0+%26+0+%26+0+%26+1+%5C%5C+0+%26+0+%26+0+%26+0+%26+0+%26+0+%26+1+%5C%5C+0+%26+0+%26+0+%26+0+%26+0+%26+0+%26+1+%5C%5C+0+%26+0+%26+0+%26+0+%26+0+%26+0+%26+0+%5C%5C+%5Cend%7BBmatrix%7D)



现在我们继续到最后，最后一列的最低的 ![[公式]](https://www.zhihu.com/equation?tex=1) ，只有那一列，所以我们什么都不做。现在我们再从左边开始。我们到了第 ![[公式]](https://www.zhihu.com/equation?tex=6) 列，发现第 ![[公式]](https://www.zhihu.com/equation?tex=4) 列最低的 ![[公式]](https://www.zhihu.com/equation?tex=1) 和它一样，**low**( ![[公式]](https://www.zhihu.com/equation?tex=3) )=**low**( ![[公式]](https://www.zhihu.com/equation?tex=5) )，所以我们把第 ![[公式]](https://www.zhihu.com/equation?tex=4) 列加到第 ![[公式]](https://www.zhihu.com/equation?tex=6) 列，结果如下：



![[公式]](https://www.zhihu.com/equation?tex=%5C%5C%5Cpartial_%7Bfiltration%7D+%3D+%5Cbegin%7BBmatrix%7D+0+%26+0+%26+0+%26+1+%26+1+%26+0+%26+0+%5C%5C+0+%26+0+%26+0+%26+1+%26+0+%26+0+%26+0+%5C%5C+0+%26+0+%26+0+%26+0+%26+1+%26+0+%26+0+%5C%5C+0+%26+0+%26+0+%26+0+%26+0+%26+0+%26+1+%5C%5C+0+%26+0+%26+0+%26+0+%26+0+%26+0+%26+1+%5C%5C+0+%26+0+%26+0+%26+0+%26+0+%26+0+%26+1+%5C%5C+0+%26+0+%26+0+%26+0+%26+0+%26+0+%26+0+%5C%5C+%5Cend%7BBmatrix%7D)



看，我们现在有了一个全为 ![[公式]](https://www.zhihu.com/equation?tex=0) 的新列！而这是什么意思？它意味着列是新的拓扑特征。它要么表示连通的分量，要么表示某些 ![[公式]](https://www.zhihu.com/equation?tex=n) 维圈。在这中情况下，他表示一个 ![[公式]](https://www.zhihu.com/equation?tex=1) 维圈，这个圈由三个 ![[公式]](https://www.zhihu.com/equation?tex=1) 维单形组成。

注意到现在矩阵被完全还原到列阶梯型，因为左右最低的 ![[公式]](https://www.zhihu.com/equation?tex=1) 都不在同一行，所以我们的算法能满足需求。现在，边界矩阵被减化了，不再是每一列和一行表示滤流中唯一一个单形的情况。因为我们已经将各列加在一起，所以每一列都可以表示滤流中多个单形。在本例中，我们只把列加起来两次，而且两次加的都是第 ![[公式]](https://www.zhihu.com/equation?tex=6) 列（ ![[公式]](https://www.zhihu.com/equation?tex=j%3D5) ），所以第 ![[公式]](https://www.zhihu.com/equation?tex=6) 列表示第 ![[公式]](https://www.zhihu.com/equation?tex=5) 列和第 ![[公式]](https://www.zhihu.com/equation?tex=4) 列的单形（恰好是 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7B0%2C1%5C%7D) 和 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7B2%2C0%5C%7D) ）。所以第 ![[公式]](https://www.zhihu.com/equation?tex=6) 列是单形的群： ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7B+%7B0%2C1%7D%2C+%7B2%2C0%7D%2C+%7B1%2C2%7D+%7D) ，如果你回头看这个单形的图形表示，那些 ![[公式]](https://www.zhihu.com/equation?tex=1) 维单形组成了一个1维循环（尽管立即被 ![[公式]](https://www.zhihu.com/equation?tex=2) 维单形 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7B0%2C1%2C2%5C%7D) 消除掉了）。

保持对算法的跟踪记录是很重要的，这样我们就能在算法完成时找出每一列代表什么。我们通过建立另一个我称之为 *记忆矩阵* 的矩阵来做这个。它开始只是与边界矩阵维数相同的单位矩阵。



![[公式]](https://www.zhihu.com/equation?tex=%5C%5CM_%7Bmemory%7D+%3D+%5Cbegin%7BBmatrix%7D+1+%26+0+%26+0+%26+0+%26+0+%26+0+%26+0+%5C%5C+0+%26+1+%26+0+%26+0+%26+0+%26+0+%26+0+%5C%5C+0+%26+0+%26+1+%26+0+%26+0+%26+0+%26+0+%5C%5C+0+%26+0+%26+0+%26+1+%26+0+%26+0+%26+0+%5C%5C+0+%26+0+%26+0+%26+0+%26+1+%26+0+%26+0+%5C%5C+0+%26+0+%26+0+%26+0+%26+0+%26+1+%26+0+%5C%5C+0+%26+0+%26+0+%26+0+%26+0+%26+0+%26+1+%5C%5C+%5Cend%7BBmatrix%7D)



但每次我在我们的还原算法中把 ![[公式]](https://www.zhihu.com/equation?tex=i) 列加到 ![[公式]](https://www.zhihu.com/equation?tex=j) 列，我们就要将 ![[公式]](https://www.zhihu.com/equation?tex=1) 放入单元 ![[公式]](https://www.zhihu.com/equation?tex=%5Bi%2Cj%5D) ，在记忆矩阵中记录这一变化。所以在我们的例子中，我们记录了添加列 ![[公式]](https://www.zhihu.com/equation?tex=4) 和 ![[公式]](https://www.zhihu.com/equation?tex=5) 到第 ![[公式]](https://www.zhihu.com/equation?tex=6) 列的事件。因为在我们的记忆矩阵中，我们让单元 ![[公式]](https://www.zhihu.com/equation?tex=%5B3%2C5%5D) 和 ![[公式]](https://www.zhihu.com/equation?tex=%5B4%2C5%5D) 等于 ![[公式]](https://www.zhihu.com/equation?tex=1) 。具体如下：



![[公式]](https://www.zhihu.com/equation?tex=%5C%5CM_%7Bmemory%7D+%3D+%5Cbegin%7BBmatrix%7D+1+%26+0+%26+0+%26+0+%26+0+%26+0+%26+0+%5C%5C+0+%26+1+%26+0+%26+0+%26+0+%26+0+%26+0+%5C%5C+0+%26+0+%26+1+%26+0+%26+0+%26+0+%26+0+%5C%5C+0+%26+0+%26+0+%26+1+%26+0+%26+1+%26+0+%5C%5C+0+%26+0+%26+0+%26+0+%26+1+%26+1+%26+0+%5C%5C+0+%26+0+%26+0+%26+0+%26+0+%26+1+%26+0+%5C%5C+0+%26+0+%26+0+%26+0+%26+0+%26+0+%26+1+%5C%5C+%5Cend%7BBmatrix%7D)



一旦算法运行完毕，我们总可以参考这个记忆矩阵来回忆算法实际做了什么，并计算出减少边界矩阵的列。

让我们回顾一下我们降低(列阶梯型)边界矩阵的滤流：



![[公式]](https://www.zhihu.com/equation?tex=%5C%5C%5Cpartial_%7Breduced%7D+%3D+%5Cbegin%7BBmatrix%7D+0+%26+0+%26+0+%26+1+%26+1+%26+0+%26+0+%5C%5C+0+%26+0+%26+0+%26+1+%26+0+%26+0+%26+0+%5C%5C+0+%26+0+%26+0+%26+0+%26+1+%26+0+%26+0+%5C%5C+0+%26+0+%26+0+%26+0+%26+0+%26+0+%26+1+%5C%5C+0+%26+0+%26+0+%26+0+%26+0+%26+0+%26+1+%5C%5C+0+%26+0+%26+0+%26+0+%26+0+%26+0+%26+1+%5C%5C+0+%26+0+%26+0+%26+0+%26+0+%26+0+%26+0+%5C%5C+%5Cend%7BBmatrix%7D)



为了记录拓扑特征的形成和消亡间隔，我们只需扫描从左到右的每一列。如果 ![[公式]](https://www.zhihu.com/equation?tex=j) 列全是 ![[公式]](https://www.zhihu.com/equation?tex=0) （即**low**( ![[公式]](https://www.zhihu.com/equation?tex=j) ) ![[公式]](https://www.zhihu.com/equation?tex=%3D-1) ）那么我们将这记为新特征的形成（不管 ![[公式]](https://www.zhihu.com/equation?tex=j) 代表什么，可能是一个单形，可能是一组单形）。

否则，如果一列不是所有的 ![[公式]](https://www.zhihu.com/equation?tex=0) ，而是有一些 ![[公式]](https://www.zhihu.com/equation?tex=1) ，那么我们就会说，在 ![[公式]](https://www.zhihu.com/equation?tex=j) 中等于 **low**( ![[公式]](https://www.zhihu.com/equation?tex=j) ) 的列，因此是该特性的区间的端点。

在我们的例子中，所有三个顶点(前三列)都是生成的新特性(它们的列都是 ![[公式]](https://www.zhihu.com/equation?tex=0) ，**low**( ![[公式]](https://www.zhihu.com/equation?tex=j) ) ![[公式]](https://www.zhihu.com/equation?tex=%3D-1) )，因此我们记录 ![[公式]](https://www.zhihu.com/equation?tex=3) 个新的间隔，起始点是它们的列索引。因为我们从左到右进行顺序扫描，我们还不知道这些特征何时会消失，所以我们只是暂时将端点设置为 ![[公式]](https://www.zhihu.com/equation?tex=-1) ，以表示结束或无限。这是前三段：

```text
#Remember the start and end points are column indices
[0,-1], [1,-1], [2,-1]
```

然后我们继续从左到右扫描，到了第 ![[公式]](https://www.zhihu.com/equation?tex=4) 列( ![[公式]](https://www.zhihu.com/equation?tex=j%3D3) )，我们计算 **low**( ![[公式]](https://www.zhihu.com/equation?tex=3) )= ![[公式]](https://www.zhihu.com/equation?tex=1) 。所以这意味着在 ![[公式]](https://www.zhihu.com/equation?tex=j%3D1) (第 ![[公式]](https://www.zhihu.com/equation?tex=2) 列)中生成的特性在 ![[公式]](https://www.zhihu.com/equation?tex=j%3D3) 的时候消失了。现在我们可以返回并更新这个间隔的暂定终点，我们的更新间隔为：

```text
#updating intervals...
[0,-1], [1,3], [2,-1]
```

我们继续这个过程直到最后一列，我们得到所有的间隔：

```text
#The final set of intervals
[0,-1], [1,3], [2,4], [5,6]
```

前三个特性是 ![[公式]](https://www.zhihu.com/equation?tex=0) 维单形，因为它们是 ![[公式]](https://www.zhihu.com/equation?tex=0) 维度，它们代表了域流的连接分量。第 ![[公式]](https://www.zhihu.com/equation?tex=4) 个特征是 ![[公式]](https://www.zhihu.com/equation?tex=1) 维循环，因为它的间隔指数指的是 ![[公式]](https://www.zhihu.com/equation?tex=1) 维单形群。

信不信由你，我们刚刚完成了持续同调。这就是它的全部。一旦我们有时间间隔，我们需要做的就是用条形码图画出它们。我们应该通过回到我们在边缘权重的集合，在这些 ![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon) 值的区间内转换开始/结束点，并分配 ![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon) 值给每个单形。下面是条形码图：

![img](https://pic4.zhimg.com/80/v2-b6ba72bbe837f29c7bc1efd86032acdb_1440w.jpg)

> 我在 ![[公式]](https://www.zhihu.com/equation?tex=H_1) 群画了一个点来表示 ![[公式]](https://www.zhihu.com/equation?tex=1) 维循环形成并马上在同一点消亡（因为它一形成就有 ![[公式]](https://www.zhihu.com/equation?tex=2) 维循环归并它了）。大多数真正的条码图都不会产生这些点。我们并不关心这些短暂的特性。

注意到现在在 ![[公式]](https://www.zhihu.com/equation?tex=H_0) 中有一条“横杠”明显比其他两条长。这表明我们的数据只有 ![[公式]](https://www.zhihu.com/equation?tex=1) 个连接的组件。群 ![[公式]](https://www.zhihu.com/equation?tex=H_1%2CH_2) 没有任何一根“横杠”，所以数据没有孔洞/循环。当然，有了更真实的数据集，我们也会发现一些环。



## **代码**

好了，我们在第五章上半部分基本上已经涵盖了计算持续同调的概念框架。让我们编写一些代码来计算实际数据的持续同调。我不会花太多精力解释所有的代码以及它的原理，因为我更关心实际和直观的解释，所以你可以自己写算法。我尝试添加行内的注释，这应该有帮助。也请记住，因为这一系列文章都是有科普目的的，这些算法和数据结构可能不会很有效，但会很简单。我希望可以针对一些点写一篇后续文章，演示如何高效使用这些算法和数据结构的版本。

让我们首先使用第四章中所写的代码构造一个简单的单纯复形结构。

```text
data = np.array([[1,4],[1,1],[6,1],[6,4]])
```



```text
#for example... this is with a small epsilon, to illustrate the presence of a 1-dimensional cycle
graph = SimplicialComplex.buildGraph(raw_data=data, epsilon=5.1)
ripsComplex = SimplicialComplex.rips(nodes=graph[0], edges=graph[1], k=3)
SimplicialComplex.drawComplex(origData=data, ripsComplex=ripsComplex, axes=[0,7,0,5])
```

![img](https://pic3.zhimg.com/80/v2-bff0a9a728d6bb6fb8dc3b8135ea33a2_1440w.jpg)

所以我们的单纯复形是一个盒子。显然它有 ![[公式]](https://www.zhihu.com/equation?tex=1) 个连接分量和 ![[公式]](https://www.zhihu.com/equation?tex=1) 维的环。如果你持续增大 ![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon) ，那么盒子会被“填满”，而我们会得到一个最大单形——所有四个点相连形成一个 ![[公式]](https://www.zhihu.com/equation?tex=3) 维单形（四面体）

接下来，我们将对第四章的代码做一些修改，让它能够处理滤流复形的计算过程，而不只是基本的单纯复形而已。

所以我准备撸一堆代码在下面，并直观的描述一下每个函数都在做什么的。`buildGraph`函数和之前一样。但是我们也有一些新的函数：`ripsFiltration`，`getFilterValue`，`compare`和`sortComplex`。

`ripsFiltration`函数接收从`buildGraph`输出的图对象和最大维度`k`（即我们要计算的单形的维度），并返回通过过滤值储存的单纯复形对象。过滤值按照上面描述的那样定义。我们有`sortComplex`方法，它能接收复形和过滤值，返回复形的排序。

所以我们之前的单纯复形函数和`ripsFiltration`函数唯一的不同是后者还要维每个单形生成过滤值，并根据域流对单形进行全序的排序。



```text
import itertools
import functools
```



```text
def euclidianDist(a,b): #this is the default metric we use but you can use whatever distance function you want
    return np.linalg.norm(a - b) #euclidian distance metric


#Build neighorbood graph
def buildGraph(raw_data, epsilon = 3.1, metric=euclidianDist): #raw_data is a numpy array
    nodes = [x for x in range(raw_data.shape[0])] #initialize node set, reference indices from original data array
    edges = [] #initialize empty edge array
    weights = [] #initialize weight array, stores the weight (which in this case is the distance) for each edge
    for i in range(raw_data.shape[0]): #iterate through each data point
        for j in range(raw_data.shape[0]-i): #inner loop to calculate pairwise point distances
            a = raw_data[i]
            b = raw_data[j+i] #each simplex is a set (no order), hence [0,1] = [1,0]; so only store one
            if (i != j+i):
                dist = metric(a,b)
                if dist <= epsilon:
                    edges.append({i,j+i}) #add edge if distance between points is < epsilon
                    weights.append(dist)
    return nodes,edges,weights

def lower_nbrs(nodeSet, edgeSet, node): #lowest neighbors based on arbitrary ordering of simplices
    return {x for x in nodeSet if {x,node} in edgeSet and node > x}

def ripsFiltration(graph, k): #k is the maximal dimension we want to compute (minimum is 1, edges)
    nodes, edges, weights = graph
    VRcomplex = [{n} for n in nodes]
    filter_values = [0 for j in VRcomplex] #vertices have filter value of 0
    for i in range(len(edges)): #add 1-simplices (edges) and associated filter values
        VRcomplex.append(edges[i])
        filter_values.append(weights[i])
    if k > 1:
        for i in range(k):
            for simplex in [x for x in VRcomplex if len(x)==i+2]: #skip 0-simplices and 1-simplices
                #for each u in simplex
                nbrs = set.intersection(*[lower_nbrs(nodes, edges, z) for z in simplex])
                for nbr in nbrs:
                    newSimplex = set.union(simplex,{nbr})
                    VRcomplex.append(newSimplex)
                    filter_values.append(getFilterValue(newSimplex, VRcomplex, filter_values))

    return sortComplex(VRcomplex, filter_values) #sort simplices according to filter values

def getFilterValue(simplex, edges, weights): #filter value is the maximum weight of an edge in the simplex
    oneSimplices = list(itertools.combinations(simplex, 2)) #get set of 1-simplices in the simplex
    max_weight = 0
    for oneSimplex in oneSimplices:
        filter_value = weights[edges.index(set(oneSimplex))]
        if filter_value > max_weight: max_weight = filter_value
    return max_weight


def compare(item1, item2): 
    #comparison function that will provide the basis for our total order on the simpices
    #each item represents a simplex, bundled as a list [simplex, filter value] e.g. [{0,1}, 4]
    if len(item1[0]) == len(item2[0]):
        if item1[1] == item2[1]: #if both items have same filter value
            if sum(item1[0]) > sum(item2[0]):
                return 1
            else:
                return -1
        else:
            if item1[1] > item2[1]:
                return 1
            else:
                return -1
    else:
        if len(item1[0]) > len(item2[0]):
            return 1
        else:
            return -1

def sortComplex(filterComplex, filterValues): #need simplices in filtration have a total order
    #sort simplices in filtration by filter values
    pairedList = zip(filterComplex, filterValues)
    #since I'm using Python 3.5+, no longer supports custom compare, need conversion helper function..its ok
    sortedComplex = sorted(pairedList, key=functools.cmp_to_key(compare)) 
    sortedComplex = [list(t) for t in zip(*sortedComplex)]
    #then sort >= 1 simplices in each chain group by the arbitrary total order on the vertices
    orderValues = [x for x in range(len(filterComplex))]
    return sortedComplex
```



```text
graph2 = buildGraph(raw_data=data, epsilon=7) #epsilon = 9 will build a "maximal complex"
ripsComplex2 = ripsFiltration(graph2, k=3)
SimplicialComplex.drawComplex(origData=data, ripsComplex=ripsComplex2[0], axes=[0,7,0,5])
```

![img](https://pic4.zhimg.com/80/v2-af441ffeaf5d5a45db297e92fdfdae37_1440w.jpg)

```text
ripsComplex2
```



```text
[[{0},
  {1},
  {2},
  {3},
  {0, 1},
  {2, 3},
  {1, 2},
  {0, 3},
  {0, 2},
  {1, 3},
  {0, 1, 2},
  {0, 1, 3},
  {0, 2, 3},
  {1, 2, 3},
  {0, 1, 2, 3}],
 [0,
  0,
  0,
  0,
  3.0,
  3.0,
  5.0,
  5.0,
  5.8309518948453007,
  5.8309518948453007,
  5.8309518948453007,
  5.8309518948453007,
  5.8309518948453007,
  5.8309518948453007,
  5.8309518948453007]]
```



```text
  #return the n-simplices and weights in a complex
def nSimplices(n, filterComplex):
    nchain = []
    nfilters = []
    for i in range(len(filterComplex[0])):
        simplex = filterComplex[0][i]
        if len(simplex) == (n+1):
            nchain.append(simplex)
            nfilters.append(filterComplex[1][i])
    if (nchain == []): nchain = [0]
    return nchain, nfilters

#check if simplex is a face of another simplex
def checkFace(face, simplex):
    if simplex == 0:
        return 1
    elif (set(face) < set(simplex) and ( len(face) == (len(simplex)-1) )): #if face is a (n-1) subset of simplex
        return 1
    else:
        return 0
#build boundary matrix for dimension n ---> (n-1) = p
def filterBoundaryMatrix(filterComplex):
    bmatrix = np.zeros((len(filterComplex[0]),len(filterComplex[0])), dtype='>i8')
    #bmatrix[0,:] = 0 #add "zero-th" dimension as first row/column, makes algorithm easier later on
    #bmatrix[:,0] = 0
    i = 0
    for colSimplex in filterComplex[0]:
        j = 0
        for rowSimplex in filterComplex[0]:
            bmatrix[j,i] = checkFace(rowSimplex, colSimplex)
            j += 1
        i += 1
    return bmatrix
```



```text
bm = filterBoundaryMatrix(ripsComplex2)
bm #Here is the (non-reduced) boundary matrix
```



```text
array([[0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
```



下面的函数是用来减化前面描述的边界矩阵(当我们手工计算的时候)。



```text
#returns row index of lowest "1" in a column i in the boundary matrix
def low(i, matrix):
    col = matrix[:,i]
    col_len = len(col)
    for i in range( (col_len-1) , -1, -1): #loop through column from bottom until you find the first 1
        if col[i] == 1: return i
    return -1 #if no lowest 1 (e.g. column of all zeros), return -1 to be 'undefined'

#checks if the boundary matrix is fully reduced
def isReduced(matrix):
    for j in range(matrix.shape[1]): #iterate through columns
        for i in range(j): #iterate through columns before column j
            low_j = low(j, matrix)
            low_i = low(i, matrix)
            if (low_j == low_i and low_j != -1):
                return i,j #return column i to add to column j
    return [0,0]

#the main function to iteratively reduce the boundary matrix
def reduceBoundaryMatrix(matrix): 
    #this refers to column index in the boundary matrix
    reduced_matrix = matrix.copy()
    matrix_shape = reduced_matrix.shape
    memory = np.identity(matrix_shape[1], dtype='>i8') #this matrix will store the column additions we make
    r = isReduced(reduced_matrix)
    while (r != [0,0]):
        i = r[0]
        j = r[1]
        col_j = reduced_matrix[:,j]
        col_i = reduced_matrix[:,i]
        #print("Mod: add col %s to %s \n" % (i+1,j+1)) #Uncomment to see what mods are made
        reduced_matrix[:,j] = np.bitwise_xor(col_i,col_j) #add column i to j
        memory[i,j] = 1
        r = isReduced(reduced_matrix)
    return reduced_matrix, memory
```



```text
z = reduceBoundaryMatrix(bm)
z
```



```text
Mod: add col 6 to 8 

Mod: add col 7 to 8 

Mod: add col 5 to 8 

Mod: add col 7 to 9 

Mod: add col 5 to 9 

Mod: add col 6 to 10 

Mod: add col 7 to 10 

Mod: add col 11 to 13 

Mod: add col 12 to 14 

Mod: add col 13 to 14 
```



```text
(array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
 array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]))
```



所以`reduceBoundaryMatrix`方法返回两个矩阵，减化边界矩阵和记录简化算法所有动作的记忆矩阵。这是必要的，所以我们可以查一下边界矩阵的每一列实际上指的是什么。它减少了边界矩阵中的每一列并不一定是一个单形，可能是一组单形，如某个 ![[公式]](https://www.zhihu.com/equation?tex=n) 维的环。



下面的函数使用减化的矩阵来读取所有在滤流中特征生成和消亡的间隔。

```text
def readIntervals(reduced_matrix, filterValues): #reduced_matrix includes the reduced boundary matrix AND the memory matrix
    #store intervals as a list of 2-element lists, e.g. [2,4] = start at "time" point 2, end at "time" point 4
    #note the "time" points are actually just the simplex index number for now. we will convert to epsilon value later
    intervals = []
    #loop through each column j
    #if low(j) = -1 (undefined, all zeros) then j signifies the birth of a new feature j
    #if low(j) = i (defined), then j signifies the death of feature i
    for j in range(reduced_matrix[0].shape[1]): #for each column (its a square matrix so doesn't matter...)
        low_j = low(j, reduced_matrix[0])
        if low_j == -1:
            interval_start = [j, -1]
            intervals.append(interval_start) # -1 is a temporary placeholder until we update with death time
            #if no death time, then -1 signifies feature has no end (start -> infinity)
            #-1 turns out to be very useful because in python if we access the list x[-1] then that will return the
            #last element in that list. in effect if we leave the end point of an interval to be -1
            # then we're saying the feature lasts until the very end
        else: #death of feature
            feature = intervals.index([low_j, -1]) #find the feature [start,end] so we can update the end point
            intervals[feature][1] = j #j is the death point
            #if the interval start point and end point are the same, then this feature begins and dies instantly
            #so it is a useless interval and we dont want to waste memory keeping it
            epsilon_start = filterValues[intervals[feature][0]]
            epsilon_end = filterValues[j]
            if epsilon_start == epsilon_end: intervals.remove(intervals[feature])
    return intervals

def readPersistence(intervals, filterComplex): 
    #this converts intervals into epsilon format and figures out which homology group each interval belongs to
    persistence = []
    for interval in intervals:
        start = interval[0]
        end = interval[1]
        homology_group = (len(filterComplex[0][start]) - 1) #filterComplex is a list of lists [complex, filter values]
        epsilon_start = filterComplex[1][start]
        epsilon_end = filterComplex[1][end]
        persistence.append([homology_group, [epsilon_start, epsilon_end]])
    return persistence
```



```text
intervals = readIntervals(z, ripsComplex2[1])
intervals
```



```text
[[0, -1], [1, 4], [2, 6], [3, 5], [7, 12]]
```



这些都是特征出现和消亡的间隔。`readPersistence`函数只会将边界矩阵目录的起始/结束点转换成相应的 ![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon) 值。它也会计算出每个区间所属的同调群(即是哪个连通数维度)。

```text
persist1 = readPersistence(intervals, ripsComplex2)
persist1
```



```text
[[0, [0, 5.8309518948453007]],
 [0, [0, 3.0]],
 [0, [0, 5.0]],
 [0, [0, 3.0]],
 [1, [5.0, 5.8309518948453007]]]
```



这个函数只会为单个维度绘制持续条码图。



```text
import matplotlib.pyplot as plt
def graph_barcode(persistence, homology_group = 0): 
    #this function just produces the barcode graph for each homology group
    xstart = [s[1][0] for s in persistence if s[0] == homology_group]
    xstop = [s[1][1] for s in persistence if s[0] == homology_group]
    y = [0.1 * x + 0.1 for x in range(len(xstart))]
    plt.hlines(y, xstart, xstop, color='b', lw=4)
    #Setup the plot
    ax = plt.gca()
    plt.ylim(0,max(y)+0.1)
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    plt.xlabel('epsilon')
    plt.ylabel("Betti dim %s" % (homology_group,))
plt.show()
```



```text
graph_barcode(persist1, 0)
graph_barcode(persist1, 1)
```



![img](https://pic3.zhimg.com/80/v2-5c1543cf63755c2487785b45d5f08b9a_1440w.jpg)

![img](https://pic1.zhimg.com/80/v2-bbe908a3f0ccfa9d931280ab0cebaa30_1440w.jpg)

太爽了，终于是持续同调的完整实现。

我们已经画出了前两个连通数的条码图。第一个条形码图比较稀松平常，因为我们想看到的是一些比其他的长得多的条，那才能说明真正的特征。在这种情况下，连通维度 ![[公式]](https://www.zhihu.com/equation?tex=0) 的条码图有最长的条，它代表着形成盒子的单连通组件，但它并没有比第二长的条长很多。这主要是示例非常简单的特征。如果我再加几个点，我们就会发现一个更长的最长条。

连通维度 ![[公式]](https://www.zhihu.com/equation?tex=2) 的条码图的形状就要好得多。我们明显只有一个长条，这表明 ![[公式]](https://www.zhihu.com/equation?tex=1) 维的环会一直存在到 ![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon+%3D5.8) 时盒子被“填充”的时候。

持续同调非常重要的特征是能够找到一些有有趣拓扑特征的数据点。如果所有持续同调能够给我们条码图，并告诉我们有多少连接组件和环，那它将非常有用，但也有所不足。

我们真正想做的是说，“看，条形码图展示了这有一个有统计学意义的 ![[公式]](https://www.zhihu.com/equation?tex=1) 维环，我想知道那些数据点形成了这个环？”

为了来测试这个过程，我们来稍微修改一下这个简单的“盒子”的单纯复形，并加入另一条边（给我们另一个连接组件）。



```text
data_b = np.array([[1,4],[1,1],[6,1],[6,4],[12,3.5],[12,1.5]])
```



```text
graph2b = buildGraph(raw_data=data_b, epsilon=8) #epsilon is set to a high value to create a maximal complex
rips2b = ripsFiltration(graph2b, k=3)
SimplicialComplex.drawComplex(origData=data_b, ripsComplex=rips2b[0], axes=[0,14,0,6])
```

![img](https://pic3.zhimg.com/80/v2-8b46172d969eee7968a5b6338df63006_1440w.jpg)

随着我们将 ![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon) 设成一个较高值，这形成了最大的复形。但我试着设计数据使得“真实”的特征是一个盒子（ ![[公式]](https://www.zhihu.com/equation?tex=1) 维的环），以及右边的一条边，总共两个“真正的”连接组件。



那让我们在数据上进行持续同调。



```text
bm2b = filterBoundaryMatrix(rips2b)
rbm2b = reduceBoundaryMatrix(bm2b)
intervals2b = readIntervals(rbm2b, rips2b[1])
persist2b = readPersistence(intervals2b, rips2b)
```



```text
graph_barcode(persist2b, 0)
graph_barcode(persist2b, 1)
```

![img](https://pic3.zhimg.com/80/v2-a53705cc810f5de15d9e74f68bda1e9e_1440w.jpg)

![img](https://pic4.zhimg.com/80/v2-d6da94cadd317384e035535074afa567_1440w.jpg)

我们可以看到在 ![[公式]](https://www.zhihu.com/equation?tex=0) 维连通数中两个连接组件（最长的两条横杠），还有 ![[公式]](https://www.zhihu.com/equation?tex=1) 维连通数中的两个条，但一个很明显是另一个的两倍长。较短的条来自右侧边线和左侧盒子最左的两条线形成的环。

因此，在这一点上，我们认为我们有一个重要的 ![[公式]](https://www.zhihu.com/equation?tex=1) 维环，但我们不知道哪些点形成这个环，因此，如果我们愿意，我们可以进一步分析这些数据的子集。

为了把它表示出来，我们只需要用到我们的归约算法返回给我们的记忆矩阵。首先，我们从`intervals2b`列表找到我们需要的区间，在例子中，它是第一个元素，然后我们得到起始点（因为这表明了特性的诞生）。起始点是边界数组中的一个索引值，所以我们将在记忆数组中找到那一列并在那一列中找到带1的单元。在该列中带1的行是群中的其他单形（包括列本身）。



```text
persist2b
```



```text
[[0, [0, 6.5]],
 [0, [0, 3.0]],
 [0, [0, 5.0]],
 [0, [0, 3.0]],
 [0, [0, 6.0207972893961479]],
 [0, [0, 2.0]],
 [1, [5.0, 5.8309518948453007]],
 [1, [6.0207972893961479, 6.5]]]
```



首先，看看同调群1的区间，然后我们想要 ![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon) 范围是从 ![[公式]](https://www.zhihu.com/equation?tex=5.0) 到 ![[公式]](https://www.zhihu.com/equation?tex=5.83) 的区间。那是在持续性列表的索引 ![[公式]](https://www.zhihu.com/equation?tex=6) ，也是间隔列表的索引 ![[公式]](https://www.zhihu.com/equation?tex=6) 。间隔列表，不是 ![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon) 的起始和终点，有索引值，因此我们可以在记忆矩阵中检索到单形。



```text
cycle1 = intervals2b[6]
cycle1
#So birth index is 10
```



```text
[10, 19]
```



```text
column10 = rbm2b[1][:,10]
column10
```



```text
array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
```



所以这是记忆矩阵索引值= ![[公式]](https://www.zhihu.com/equation?tex=10) 的列。所以我们自然而然地知道无论索引 ![[公式]](https://www.zhihu.com/equation?tex=10) 中的单形是什么，都是环的一部分，也是这一列中有 ![[公式]](https://www.zhihu.com/equation?tex=1) 的行。



```text
ptsOnCycle = [i for i in range(len(column10)) if column10[i] == 1]
ptsOnCycle
```



```text
[7, 8, 9, 10]
```



```text
#so the simplices with indices 7,8,9,10 lie on our 1-dimensional cycle, let's find what those simplices are
rips2b[0][7:11] #range [start:stop], but stop is non-inclusive, so put 11 instead of 10
```



```text
[{0, 1}, {2, 3}, {1, 2}, {0, 3}]
```



完全正确！这就是我们在条码中看到的1维循环的 ![[公式]](https://www.zhihu.com/equation?tex=1) 维单形列表。从这个列表到原始数据点应该很简单，所以我不会在这里向你介绍这些细节。

那让我们用一些更实际的数据来试试。我们将像之前做的那样，从圆中取一些数据。在这个例子中，我在`ripsFiltration`函数中设置参数`k=2`，那么它最多只会生成 ![[公式]](https://www.zhihu.com/equation?tex=2) 维单形。这只是为了减少内存的需要。如果你有一台有很多内存的电脑，你当然可以把`k`设置成 ![[公式]](https://www.zhihu.com/equation?tex=3) ，但我不会让它更大。通常我们对连接的分量和1或者 ![[公式]](https://www.zhihu.com/equation?tex=2) 维环感兴趣。拓扑特征在维度上的效用貌似是逐渐递减的，并且内存和算法运行时间的代价一般不太值得。

> 注意：接下来可能需要一段时间运行，大概要几分钟。这是因为在这些教程中编写的代码是为了清晰和方便而进行的优化，而不是为了效率和速度。如果我们想要在任何地方接近一个现成的TDA库，那么有很多性能可以优化并且应该被制造。我计划到时候写一篇后续文章，讨论最合理的算法和数据结构优化，因为我希望在未来开发一个合理高效的 **python** 开源TDA库，希望能得到一些帮助。



```text
n = 30 #number of points to generate
#generate space of parameter
theta = np.linspace(0, 2.0*np.pi, n) 
a, b, r = 0.0, 0.0, 5.0
x = a + r*np.cos(theta)
y = b + r*np.sin(theta)
#code to plot the circle for visualization
plt.plot(x, y)
plt.show()
xc = np.random.uniform(-0.25,0.25,n) + x #add some "jitteriness" to the points (but less than before, reduces memory)
yc = np.random.uniform(-0.25,0.25,n) + y
fig, ax = plt.subplots()
ax.scatter(xc,yc)
plt.show()
circleData = np.array(list(zip(xc,yc)))
```



![img](https://pic2.zhimg.com/80/v2-7bbad736dc77b2d3657363d3da46115d_1440w.jpg)

![img](https://pic4.zhimg.com/80/v2-bf31d36ecb07f690e2bdc49fc46e78db_1440w.jpg)

```text
graph4 = buildGraph(raw_data=circleData, epsilon=3.0)
rips4 = ripsFiltration(graph4, k=2)
SimplicialComplex.drawComplex(origData=circleData, ripsComplex=rips4[0], axes=[-6,6,-6,6])
```

![img](https://pic3.zhimg.com/80/v2-1451be355d17a1fbe742a1f08fb760de_1440w.jpg)

显然，持续同调告诉我们，我们有 ![[公式]](https://www.zhihu.com/equation?tex=1) 个连接组件和 ![[公式]](https://www.zhihu.com/equation?tex=1) 维环。

```text
len(rips4[0])
#On my laptop, a rips filtration with more than about 250 simplices will take >10 mins to compute persistent homology
#anything < ~220 only takes a few minutes or less
```



```text
148
```



```text
%%time
bm4 = filterBoundaryMatrix(rips4)
rbm4 = reduceBoundaryMatrix(bm4)
intervals4 = readIntervals(rbm4, rips4[1])
persist4 = readPersistence(intervals4, rips4)
```



```text
CPU times: user 43.4 s, sys: 199 ms, total: 43.6 s
Wall time: 44.1 s
```



```text
graph_barcode(persist4, 0)
graph_barcode(persist4, 1)
```

![img](https://pic1.zhimg.com/80/v2-371dd7467f022b680ffbd9e80ded0d64_1440w.jpg)

![img](https://pic4.zhimg.com/80/v2-325ca876fe6d4938fc68998579bb944b_1440w.jpg)

我们可以清楚地看到，在`Betti dim 0`条码图中，有一个明显较长的条，这表明我们只有一个明显的连接组件。这很符合我们绘制的圆形数据。

`Betti dim 1`条码图更简单，因为它只有一个条，所以我们当然有明显的特征，即 ![[公式]](https://www.zhihu.com/equation?tex=1) 维循环。

那么，像之前一样，我们开始测试我们的算法。

我准备从一个叫双扭线（也可以叫 ![[公式]](https://www.zhihu.com/equation?tex=8) 字形）的图形中采样。正如你所见，它应该有一个连接组件和两个 ![[公式]](https://www.zhihu.com/equation?tex=1) 维环。



```text
n = 50
t = np.linspace(0, 2*np.pi, num=n)
#equations for lemniscate
x = np.cos(t) / (np.sin(t)**2 + 1)
y = np.cos(t) * np.sin(t) / (np.sin(t)**2 + 1)

plt.plot(x, y)
plt.show()
```

![img](https://pic2.zhimg.com/80/v2-3cb20f4d6c0277495c4f218ac12b4f55_1440w.jpg)



```text
x2 = np.random.uniform(-0.03, 0.03, n) + x #add some "jitteriness" to the points
y2 = np.random.uniform(-0.03, 0.03, n) + y
fig, ax = plt.subplots()
ax.scatter(x2,y2)
plt.show()
```

![img](https://pic4.zhimg.com/80/v2-9e426b4bd13ea02841c6ba7a672bdbc3_1440w.jpg)

```text
figure8Data = np.array(list(zip(x2,y2)))
```



```text
graph5 = buildGraph(raw_data=figure8Data, epsilon=0.2)
rips5 = ripsFiltration(graph5, k=2)
SimplicialComplex.drawComplex(origData=figure8Data, ripsComplex=rips5[0], axes=[-1.5,1.5,-1, 1])
```

![img](https://pic3.zhimg.com/80/v2-b258ce5ef6fd131eab4f82b2c980bf46_1440w.jpg)

```text
%%time
bm5 = filterBoundaryMatrix(rips5)
rbm5 = reduceBoundaryMatrix(bm5)
intervals5 = readIntervals(rbm5, rips5[1])
persist5 = readPersistence(intervals5, rips5)
```



```text
CPU times: user 17min 8s, sys: 3.93 s, total: 17min 12s
Wall time: 17min 24s
```



```text
graph_barcode(persist5, 0)
graph_barcode(persist5, 1)
```

![img](https://pic1.zhimg.com/80/v2-d43c0455261eb0a268f7ded9fc2a159c_1440w.jpg)

![img](https://pic3.zhimg.com/80/v2-156f9ddb2066f85d5b14541c0fba55ae_1440w.jpg)

正如我们所想，`Betti dim 0`显示有一个条明显长于其他，`Betti dim 1`显示有两个条，即有两个 ![[公式]](https://www.zhihu.com/equation?tex=1) 维环。



## **结束**

第5章是关于持续同调的子系列的最后一章。现在，你应该具备理解和使用现有的持续同调工具所需的所有知识，甚至可以自己构建自己的工具。