---
title: Persistent-Homology-2
tag: TDA-PH
categories: TDA-PH
---

[![yPnSHO.jpg](https://s3.ax1x.com/2021/01/29/yPnSHO.jpg)](https://imgchr.com/i/yPnSHO)

------

> https://zhuanlan.zhihu.com/p/31946203

------

## 单纯形，单纯复形和 VR复形

接下来我们开始写代码。一般来说，我的文章都会涉及到代码，而且非常实用，但是拓扑数据分析并不简单，一个人必须理解基础的数学才能取得进展。

我们将学习如何从 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+R%5E2) 中一个圆形的模拟数据中构建一个VR复形。

我们从这个形状中随机取样，假设它是我们的原始点云数据。许多真正的数据是由循环过程产生的，所以这只是一次不现实的练习。我们将使用我们的点云数据依照上面的描述构建一个 VR 复形。然后我们要用更多的数学方法来确定复形的同调群。

回想一下，生成圆的点集的参数形式如下:

![[公式]](https://www.zhihu.com/equation?tex=x%3Da%2Brcos%28%CE%B8%29%2C+y%3Db%2Brsin%28%CE%B8%29) ，其中 ![[公式]](https://www.zhihu.com/equation?tex=%28a%2C+b%29) 是圆心， ![[公式]](https://www.zhihu.com/equation?tex=%CE%B8) 是 0 到 ![[公式]](https://www.zhihu.com/equation?tex=2%CF%80) 的参数，而 ![[公式]](https://www.zhihu.com/equation?tex=r) 是半径。

下面的代码将生成并绘制采样圆的离散点

```python
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



![img](https://pic3.zhimg.com/80/v2-77b967812e334ef0144812130c3cf44a_1440w.jpg)



好吧，让我们从这个(有点)完美的圆圈中随机取样，顺便加点误差。

```python
x2 = np.random.uniform(-0.75,0.75,n) + x #add some "jitteriness" to the points
y2 = np.random.uniform(-0.75,0.75,n) + y
fig, ax = plt.subplots()
ax.scatter(x2,y2)
plt.show()
```


![img](https://pic1.zhimg.com/80/v2-b78ad357e3b9b9ede1012233d286f3a4_1440w.jpg)



很明显，生成的点看起来是“圆”，因为在这里它是有一个洞的明显的循环的圈，所以我们想要我们的单纯复形捕捉到这个属性。

让我们用以下步骤慢慢分解VR复形的结构：

1. 定义距离函数 ![[公式]](https://www.zhihu.com/equation?tex=d%28a%2Cb%29+%3D+%5Csqrt%7B%28a_1-b_1%29%5E2%2B%28a_2-b_2%29%5E2%7D) （欧几里得距离）
2. 为构建VR复形建立参数 ![[公式]](https://www.zhihu.com/equation?tex=%CF%B5)
3. 建立一个点云数据的集合（用python的list，最接近集合的结构），这将是 0 维单纯形的复形。
4. 扫描每一对点，计算点之间的距离。如果两点距离小于 ![[公式]](https://www.zhihu.com/equation?tex=%CF%B5) ，我们就给这两个点连一条线，这时我们会获得一个1维复形。
5. 如果我们计算了所有距离并画好一个图，我们可以遍历每一个顶点，识别它们的邻点并尝试递增地建立高维复形（在 1 维复形中加入所有 2 维复形，再加入所有维复形，依此类推）。



有很多算法可以从数据中构建一个单纯复形（除了VR复形，还有很多其他类型的单纯复形）。不幸的是，以我了解，这世上没有从点数据中创建一个完整的（不是向下采样的）单纯复形的多项式时间复杂度的算法。因此，无论如何，一旦我们开始处理真正的大数据集，构建复形将变成计算代价相当高的行为(甚至令人望而却步)。在这方面，我们需要做更多的工作。

我们将使用 Afra Zomorodian 的“快速构建VR复形”的算法。这个算法有两个主要步骤。

\1. 构造点集数据的邻域图。领域图是无向加权图 ![[公式]](https://www.zhihu.com/equation?tex=%28G%2C+w%29) ，其中 ![[公式]](https://www.zhihu.com/equation?tex=G%3D%28V%2C+E%29) ， ![[公式]](https://www.zhihu.com/equation?tex=V) 是顶点集， ![[公式]](https://www.zhihu.com/equation?tex=E)是边集，而 ![[公式]](https://www.zhihu.com/equation?tex=w%3AE%E2%86%92%5Cmathbb+R) （ ![[公式]](https://www.zhihu.com/equation?tex=w) 是每个边到实数的映射，这是权重）。我们的边是通过连接一些我们曾互相定义过距离（用参数 ![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon) ）的点创造的。特别的，

![[公式]](https://www.zhihu.com/equation?tex=E_%7B%5Cepsilon%7D+%3D+%5C%7B%5C%7Bu%2Cv%5C%7D+%5Cmid+d%28u%2Cv%29+%5Cleq+%5Cepsilon%2C+u+%5Cneq+v+%5Cin+V%5C%7D)

其中 ![[公式]](https://www.zhihu.com/equation?tex=d%28u%2C+v%29) 是两个点 ![[公式]](https://www.zhihu.com/equation?tex=u%2C+v%E2%88%88V) 的度量/距离函数。而权重函数简单地将每条边的权值设为等于边的两个点之间的距离。也就是， ![[公式]](https://www.zhihu.com/equation?tex=w%28%5C%7Bu%2Cv%5C%7D%29+%3D+d%28u%2Cv%29%2C+%5Cforall+%5C%7Bu%2Cv%5C%7D+%5Cin+E_%7B%5Cepsilon%7D%28V%29) 。

\2. 在第一步生成的领域图形成VR扩张。给定一个领域图 ![[公式]](https://www.zhihu.com/equation?tex=%28G%2C+w%29) ，VR复形 ![[公式]](https://www.zhihu.com/equation?tex=%28R%28G%29%2C+w%29) （其中 ![[公式]](https://www.zhihu.com/equation?tex=R) 是 VR复形）权重过滤通过下列方式给定：

![[公式]](https://www.zhihu.com/equation?tex=R%28G%29+%3D+V+%5Ccup+E+%5Ccup+%5C%7B+%5Csigma+%5Cmid+%5Cleft+%28%7B%5Csigma%7D%5Cabove+0pt+%7B2%7D+%5Cright+%29+%5Csubseteq+E+%5C%7D+%2C)

对于 ![[公式]](https://www.zhihu.com/equation?tex=%CF%83%E2%88%88R%28G%29) ，

![[公式]](https://www.zhihu.com/equation?tex=w%28%5Csigma%29+%3D+%5Cleft%5C%7B+%5Cbegin%7Barray%7D%7Bll%7D+0%2C+%26+%5Csigma+%3D+%5C%7Bv%5C%7D%2Cv+%5Cin+V%2C+%5C%5C+w%28%5C%7Bu%2Cv%5C%7D%29%2C+%26+%5Csigma+%3D+%5C%7Bu%2Cv%5C%7D+%5Cin+E+%5C%5C+%5Cdisplaystyle+%5Coperatorname%2A%7Bmax%7D_%7B%5Crm+%5Ctau+%5C+%5Csubset+%5C+%5Csigma%7D+w%28%5Ctau%29%2C+%26+otherwise.+%5Cend%7Barray%7D+%5Cright%5C%7D)



这是什么意思？在这个简单的例子中，我们准备从我们的领域图（左）中获得 VR 复形（右）：



![img](https://pic2.zhimg.com/80/v2-77d585acb32fa40e1b17c58b9d6c6049_1440w.jpg)



所以上面的算法是说维VR复形是领域图中所有顶点和边线并的集合，也是所有在E的单纯形 ![[公式]](https://www.zhihu.com/equation?tex=%5Csigma) 中每种两个顶点可能的组合的并。

下面的部分给所有VR复形的每个单纯形，从单个0维单纯形到高维单纯形，定义了权重函数。如果单纯形是 0 维单纯形（只有顶点），那么单纯形的权重是 0。如果单纯形是1维单纯形（有边），那么权重是两点之间的距离。如果单纯形是高维单纯形，如2维单纯形（三角形），那么权重是单纯形中最长边的权重。

在我们计算我们“圈形”数据的VR复形之前，让我们给上面出现过的单纯形进行一个全面检查。我们先在 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+R%5E2) 中嵌入顶点，然后尝试先去构建领域图。

```python
raw_data = np.array([[0,2],[2,2],[1,0],[1.5,-3.0]]) #embedded 3 vertices in R^2
plt.axis([-1,3,-4,3])
plt.scatter(raw_data[:,0],raw_data[:,1]) #plotting just for clarity
for i, txt in enumerate(raw_data):
    plt.annotate(i, (raw_data[i][0]+0.05, raw_data[i][1])) #add labels
```

![img](https://pic2.zhimg.com/80/v2-3114bbd7eaa024da2e61a65f387e42e9_1440w.jpg)



我们将用原始数据数组中的索引号来表示我们单纯复形中的每个顶点。例如点 ![[公式]](https://www.zhihu.com/equation?tex=%5B0%2C+2%5D) 出现在我们数据数组的开头，因此我们把它当作单纯复形中的参考点。

```python
#Build neighorbood graph
nodes = [x for x in range(raw_data.shape[0])] #initialize node set, reference indices from original data array
edges = [] #initialize empty edge array
weights = [] #initialize weight array, stores the weight (which in this case is the distance) for each edge
eps = 3.1 #epsilon distance parameter
for i in range(raw_data.shape[0]): #iterate through each data point
    for j in range(raw_data.shape[0]-i): #inner loop to calculate pairwise point distances
        a = raw_data[i]
        b = raw_data[j+i] #each simplex is a set (no order), hence [0,1] = [1,0]; so only store one
        if (i != j+i):
            dist = np.linalg.norm(a - b) #euclidian distance metric
            if dist <= eps:
                edges.append({i,j+i}) #add edge
                weights.append([len(edges)-1,dist]) #store index and weight
print("Nodes: " , nodes)
print("Edges: " , edges)
print("Weights: ", weights)
Nodes:  [0, 1, 2, 3]
Edges:  [{0, 1}, {0, 2}, {1, 2}, {2, 3}]
Weights:  [[0, 2.0], [1, 2.2360679774997898], [2, 2.2360679774997898], [3, 3.0413812651491097]]
```

完美。现在我们有一个结点集，边集和一个权重集，这所有构成了我们的领域图 ![[公式]](https://www.zhihu.com/equation?tex=%28G%2C+w%29) 。我们的下一个任务是用邻域图构建高维单纯形。在这种情况下，我们只有一个额外的2维单纯形（三角形）。我们需要设置一些基础的方法。

```python
def lower_nbrs(nodeSet, edgeSet, node):
    return {x for x in nodeSet if {x,node} in edgeSet and node > x}

def rips(nodes, edges, k):
    VRcomplex = [{n} for n in nodes]
    for e in edges: #add 1-simplices (edges)
        VRcomplex.append(e)
    for i in range(k):
        for simplex in [x for x in VRcomplex if len(x)==i+2]: #skip 0-simplices
            #for each u in simplex
            nbrs = set.intersection(*[lower_nbrs(nodes, edges, z) for z in simplex])
            for nbr in nbrs:
                VRcomplex.append(set.union(simplex,{nbr}))
    return VRcomplex
```



好的，让我们来试试它能不能用。假设我们要将所有的单纯形都提升到 ![[公式]](https://www.zhihu.com/equation?tex=3) 维。



```python
theComplex = rips(nodes, edges, 3)
theComplex
```



```python
[{0}, {1}, {2}, {3}, {0, 1}, {0, 2}, {1, 2}, {2, 3}, {0, 1, 2}]
```

嗯，看起来很完美。

现在我们想看看它是什么样的。我写了一些代码，它能根据我们VR算法的输出绘制单纯复形。但这对于理解 TDA 并不重要(大多数情况下，我们不会尝试可视化单纯复形，因为它们太过高维)，因此我不会尝试解释绘制图形的代码。

```python
plt.clf()
plt.axis([-1,3,-4,3])
plt.scatter(raw_data[:,0],raw_data[:,1]) #plotting just for clarity
for i, txt in enumerate(raw_data):
    plt.annotate(i, (raw_data[i][0]+0.05, raw_data[i][1])) #add labels

#add lines for edges
for edge in [e for e in theComplex if len(e)==2]:
    pt1,pt2 = [raw_data[pt] for pt in [n for n in edge]]
    print(pt1,pt2)
    line = plt.Polygon([pt1,pt2], closed=None, fill=None, edgecolor='r')
    plt.gca().add_line(line)
    
#add triangles
for triangle in [t for t in theComplex if len(t)==3]:
    pt1,pt2,pt3 = [raw_data[pt] for pt in [n for n in triangle]]
    line = plt.Polygon([pt1,pt2,pt3], closed=False, color="blue",alpha=0.3, fill=True, edgecolor=None)
    plt.gca().add_line(line)
plt.show()
```



```python
[ 0.  2.] [ 2.  2.]
[ 0.  2.] [ 1.  0.]
[ 2.  2.] [ 1.  0.]
[ 1.  0.] [ 1.5 -3. ]
```



![img](https://pic2.zhimg.com/80/v2-a62a882d69d38702a8233cb5a5b0d861_1440w.jpg)

现在我们对我们非常简单的VR复形有一个很好的描述。现在我们知道该怎么做。我们需要了解单纯同调，一种关于单纯形和复形拓扑不变量的研究。特别是，我们希望用数学方法辨认 ![[公式]](https://www.zhihu.com/equation?tex=n) 维连通的组件，圈和孔。为了帮助这些工作，我重新打包了我们在上面使用的代码作为一个单独的文件，这样我们就可以直接导入它，并方便地使用我们的数据上的函数。你可以在这里下载最新的代码：<[https://github.com/outlace/OpenTDA/blob/master/SimplicialComplex.py](https://link.zhihu.com/?target=https%3A//github.com/outlace/OpenTDA/blob/master/SimplicialComplex.py)>

在这里，我将从我们从一个圆中取样的(抖动的)点上压缩我们的 ![[公式]](https://www.zhihu.com/equation?tex=x) 和 ![[公式]](https://www.zhihu.com/equation?tex=y) 坐标，这样我们就可以用它来构建一个更复杂的单纯复形。

```python
newData = np.array(list(zip(x2,y2)))
import SimplicialComplex
graph = SimplicialComplex.buildGraph(raw_data=newData, epsilon=3.0)
ripsComplex = SimplicialComplex.rips(nodes=graph[0], edges=graph[1], k=3)
SimplicialComplex.drawComplex(origData=newData, ripsComplex=ripsComplex)
```



![img](https://pic3.zhimg.com/80/v2-6fbe479a85d916429d2541e9f29eb6c2_1440w.jpg)



整洁如约！很明显，我们已经复制了采样点的圆形空间。注意，这里有1维单纯形和更高维的单纯形，但它形成了用单个洞连接的组件。

如果我们将半径参数 ![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon) 减短过多，就会变成下图的样子：

![img](https://pic4.zhimg.com/80/v2-8736fe8c60ad135524deb4a115602a53_1440w.jpg)



## 同调群

既然我们已经知道什么是单纯复形，以及如何从原始数据中生成它们，我们就需要进一步计算这些单纯复形有趣的拓扑特征。

以计算同调为形式的拓扑数据分析给了我们一种在我们基于数据集建立的拓扑空间（通常是单纯复形）中识别组件和 ![[公式]](https://www.zhihu.com/equation?tex=n)维“孔”（如圆中间的孔）的数量的方法。

在我们继续之前，我想去描述一种可以加在我们已经用过的单纯复形上的额外属性。我们可以给它一个定向的性质。一个定向的单纯形 ![[公式]](https://www.zhihu.com/equation?tex=%CF%83%3Du_1%2C+u_2%2C+u_3%2C+%E2%80%A6u_n)是通过它的顶点顺序定义的。因此，定向单纯形 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7Ba%2C+b%2C+c%5C%7D) 和定向单纯形 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7Bc%2C+b%2C+a%5C%7D) 是不同的。在绘制低维图像时，我们可以通过将边缘变成箭头来描述它。



![img](https://pic4.zhimg.com/80/v2-260e590723d479449666bed29b20d7a3_1440w.jpg)

现在，严格地说一个数学集合(用曲线括号表示)是一个无序的对象集合，所以为了在单纯形上加入方向，我们需要增加一些额外的数学结构，如通过在元素间加入二元的≤关系使顶点的集合成为有序集。但这并不值得深入研究，我们只需要假设顶点集是有序的，不需要大张旗鼓地声明额外的结构必须精确定义其顺序。

回头看看上面的两个单纯形，我们可以发现这两个单纯形箭头的方向是相反的。假设左边那个单纯形是 ![[公式]](https://www.zhihu.com/equation?tex=%CF%83_1) ，右边那个单纯形是 ![[公式]](https://www.zhihu.com/equation?tex=%CF%83_2) ，我们可以说 ![[公式]](https://www.zhihu.com/equation?tex=%CF%83_1%3D-%CF%83_2) 。

引入方向的原因将会在后面清楚地说明。



## ![[公式]](https://www.zhihu.com/equation?tex=n) 维链

记住，一个单纯复形包含了这个复形中每一个最高维度的单纯形的所有面。也就是说，如果我们有一个 ![[公式]](https://www.zhihu.com/equation?tex=2) 维复形（一个拥有最高维度单纯形的单纯复形是 ![[公式]](https://www.zhihu.com/equation?tex=2) 维单纯形（三角形）），那这个复形也包含了所有比它低维的面（如边和顶点）。

![[公式]](https://www.zhihu.com/equation?tex=C%3D%5C%7B%5C%7B0%5C%7D%2C+%5C%7B1%5C%7D%2C+%5C%7B2%5C%7D%2C+%5C%7B3%5C%7D%2C+%5C%7B0%2C+1%5C%7D%2C+%5C%7B0%2C+2%5C%7D%2C+%5C%7B1%2C+2%5C%7D%2C+%5C%7B2%2C+3%5C%7D%2C+%5C%7B0%2C+1%2C+2%5C%7D%5C%7D) 是由一个点云(例如数据集)构成的单纯复形结构， ![[公式]](https://www.zhihu.com/equation?tex=X%3D%5C%7B0%2C+1%2C+2%2C+3%5C%7D) 。

![[公式]](https://www.zhihu.com/equation?tex=C) 是一个 ![[公式]](https://www.zhihu.com/equation?tex=2) 维复形，因为它最高维的单纯形是 ![[公式]](https://www.zhihu.com/equation?tex=2) 维单纯形（三角形）。我们可以把这个复形分解成一系列的子集，其中每组都由所有 ![[公式]](https://www.zhihu.com/equation?tex=k) 维单纯形组成。在单纯同调理论中，这些组叫做链群，并且任意特定组是k维链群 ![[公式]](https://www.zhihu.com/equation?tex=C_k%28X%29) 。例如 ![[公式]](https://www.zhihu.com/equation?tex=C) 的 ![[公式]](https://www.zhihu.com/equation?tex=1) 维群组是 ![[公式]](https://www.zhihu.com/equation?tex=C_1%28X%29%3D%5C%7B%5C%7B0%2C+1%5C%7D%2C+%5C%7B0%2C+2%5C%7D%2C+%5C%7B1%2C+2%5C%7D%2C+%5C%7B2%2C+3%5C%7D%5C%7D) 。



## 抽象代数基础

“链群”中的“群”事实上有一个特定的数学意义。“群”的概念来自抽象代数，一个概括了你高中代数课上学过的课题的数学领域。不用说，它很抽象，但我会尽力从一些容易概念化的具体例子开始，然后才慢慢抽象出来，得到最基础的概念。我将覆盖到群，环，域，模，向量空间和各种其他小主题，当他们出现的时候。一旦我们解决了这些问题，我们将会到链群的讨论中。



## 群

被称为群的数学结构可以被认为是对对称概念的概括。有一个完善的研究群的数学体系叫群论。我们不会在这里深入研究群，因为我们只需要知道我们需要知道的。为了我们的目的，群是一个具有对称性的数学对象。从几何学的角度来说，这可能是最容易理解的，但你会发现，群是如此普遍，以至于许多不同的数学结构都可以从群论的角度受益。



![img](https://pic2.zhimg.com/80/v2-8ec317245d296e13f2608d76da5e6db1_1440w.jpg)



目测我们可以在这个三角形上看到一些不改变结构的可能的操作。我画了对称线，这表示你可以在这 ![[公式]](https://www.zhihu.com/equation?tex=3) 条线上镜像，最后还是得到同样的三角形结构。更重要的是，你可以在平面上转化这个三角形并且仍然是相同的结构。你也可以将三角形旋转 ![[公式]](https://www.zhihu.com/equation?tex=120) 度，它仍然保持三角形的结构。群论为管理这些类型的操作及其结果提供了精确的工具。

这是群的数学定义：

> **群：**是集合 ![[公式]](https://www.zhihu.com/equation?tex=G) 和一个二元运算 ![[公式]](https://www.zhihu.com/equation?tex=%2A) （或者其他你喜欢的符号），这一运算将 ![[公式]](https://www.zhihu.com/equation?tex=a%2C+b%5Cin+G) 映射到 ![[公式]](https://www.zhihu.com/equation?tex=c%5Cin+G) 上，写作 ![[公式]](https://www.zhihu.com/equation?tex=a%2Ab%3Dc%2C+%5Cforall+a%2C+b%2C+c+%5Cin+G) 。这个集合和它的运算被写成有序对 ![[公式]](https://www.zhihu.com/equation?tex=%28G%2C+%2A%29) 。另外，为了成为一个有效的群，集合和运算必须满足以下条件：
>
> \1. **结合律**： ![[公式]](https://www.zhihu.com/equation?tex=%5Cforall+a%2C+b%2C+c%5Cin+G%2C+%28a%2Ab%29%2Ac%3Da%2A%28b%2Ac%29)
>
> \2. **单位元**：存在一个元素 ![[公式]](https://www.zhihu.com/equation?tex=e+%5Cin+G) ，使得所有元素 ![[公式]](https://www.zhihu.com/equation?tex=a+%5Cin+G)，等式 ![[公式]](https://www.zhihu.com/equation?tex=a%2Ae%3De%2Aa%3Da) 成立。这样的元素是唯一的，被称为单位元素。
>
> \3. **逆元**：对于所有 ![[公式]](https://www.zhihu.com/equation?tex=a%5Cin+G) ，存在一个元素 ![[公式]](https://www.zhihu.com/equation?tex=b%5Cin+G) ，通常表示为 ![[公式]](https://www.zhihu.com/equation?tex=a%5E%7B-1%7D) ，使得 ![[公式]](https://www.zhihu.com/equation?tex=a%2Ab%3Db%2Aa%3De) ，其中 ![[公式]](https://www.zhihu.com/equation?tex=e) 是单位元素。
>
> *备注：运算 ![[公式]](https://www.zhihu.com/equation?tex=%2A) 不一定可以交换，即 ![[公式]](https://www.zhihu.com/equation?tex=a%2Ab%3Db%2Aa) 不一定成立。运算的顺序可能有关系。如果没有关系，则称之为交换群。集合 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+Z) （整数集）和加法是一个交换群，因为 ![[公式]](https://www.zhihu.com/equation?tex=1%2B2%3D2%2B1)* 。



“群”的概念看起来很随意让人不禁想问用途何在，但现在我们有希望把这个讲清楚些。记住，所有的数学对象都是由一些(看似任意的)公理集合(基本上定义结构的规则集必须遵守)。你可以在集合上定义任何您想要的结构(只要它们逻辑一致而且有统一规则)，那你将需要一些数学对象/结构。有些结构比其他的更有趣。有些集合有很多结构(比如很多规则)，而有的则很少。通常，具有许多规则的结构仅仅是更通用/抽象结构的专门化。群只是一些数学结构(被人为赋予了规则的集合)，具有有趣的属性，并且在很多领域都很有用。但是由于他们是如此的普遍，所以有时比较难确切地解释他们。

让我们看看我们能不能从上面“群化”我们的三角形示例。我们可以把三角形看作是一组被标记的顶点，就当它是 ![[公式]](https://www.zhihu.com/equation?tex=2) 维单纯形。既然我们已经标记了三角形的顶点，我们可以很容易地把它描述为集合:

![[公式]](https://www.zhihu.com/equation?tex=t%3D%7Ba%2C+b%2C+c%7D+%5C%5C)

但是我们如何定义 ![[公式]](https://www.zhihu.com/equation?tex=t) 的二元运算？我不确定，让我们试一试。我们将构建一个表，它会向我们展示当我们在 ![[公式]](https://www.zhihu.com/equation?tex=t) 中“运算”两个元素时会发生什么。我认真的完成以下二元运算 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7B%28a%2C+b%29%E7%9A%84%E6%98%A0%E5%B0%84%7Ca%2C+b%5Cin+t+%5C%7D) 并看看它是不是有效的群。如下图。



![img](https://pic4.zhimg.com/80/v2-b59ac3d4b27bd47fad13b04a2b2aec6f_1440w.jpg)



为了了解 ![[公式]](https://www.zhihu.com/equation?tex=a%2Ab) 是什么，你从第一行开始看，找到 ![[公式]](https://www.zhihu.com/equation?tex=a) ，然后在左列找到 ![[公式]](https://www.zhihu.com/equation?tex=b) ，它们相交的地方即是结果。在我举的例子中， ![[公式]](https://www.zhihu.com/equation?tex=a%2Ab%3Da) 。注意，我已经定义了这个操作是不可交换的，所以 ![[公式]](https://www.zhihu.com/equation?tex=a%2Ab%E2%89%A0b%2Aa) 。你必须从上一行开始，然后到左边的行(按顺序)。

现在你应该能够很快地看出，这实际上不是一个有效的群，因为它违反了群的公理。例如元素 ![[公式]](https://www.zhihu.com/equation?tex=b%5Cin+t) ，你找不到一个单位元素 ![[公式]](https://www.zhihu.com/equation?tex=e) ，使得 ![[公式]](https://www.zhihu.com/equation?tex=b%2Ae%3Db) 。

所以我们再来一次。这一次，我想要创建一个有效的群。



![img](https://pic2.zhimg.com/80/v2-805c744240ea25347354cda62ca93499_1440w.jpg)



你应该检查一下，这实际上是一个有效的群，这一次这个群是可交换的，所以我们称它为交换群。单位元素是 ![[公式]](https://www.zhihu.com/equation?tex=a) ，因为 ![[公式]](https://www.zhihu.com/equation?tex=a%2Ab%3Db%2Aa%3Db%EF%BC%8Ca%2Ac%3Dc%2Aa%3Dc) 。注意，表格本身目测就能看出来有对称性。

结果表明，有限的群，就像有限的拓扑空间一样，可以被表示为有向图，这有助于可视化(数学的模式很漂亮，不是吗？) 这些群的图有一个特殊的名称: Cayley图。构造一个Cayley图比构造拓扑空间的图要复杂得多。我们必须向Cayley图形添加另一个属性，除了直接使用箭头(边)，我们还为每个箭头分配一个运算。因为如果箭头从 ![[公式]](https://www.zhihu.com/equation?tex=a) 指向 ![[公式]](https://www.zhihu.com/equation?tex=b) ，表示在群中， ![[公式]](https://www.zhihu.com/equation?tex=a) 运算生成 ![[公式]](https://www.zhihu.com/equation?tex=b) 。并不是所有的箭头都是相同的运算，所以在可视化的帮助下，我们通常会使每一种类型的运算都与一个不同颜色的箭头相关联。

在我们构建一个Cayley图之前，我们需要了解一个群的生成集是什么。记住，群 ![[公式]](https://www.zhihu.com/equation?tex=%28G%2C+%2A%29) 是集合 ![[公式]](https://www.zhihu.com/equation?tex=G) 加二元运算 ![[公式]](https://www.zhihu.com/equation?tex=%2A) 。生成集是子集 ![[公式]](https://www.zhihu.com/equation?tex=S+%5Csubseteq+G) ，比如 ![[公式]](https://www.zhihu.com/equation?tex=G+%3D+%5C%7Ba+%2A+b+%5Cmid+a%2Cb+%5Cin+S%5C%7D) 。总之，这说明生成集 ![[公式]](https://www.zhihu.com/equation?tex=S) 是 ![[公式]](https://www.zhihu.com/equation?tex=G) 的子集，但如果我们在 ![[公式]](https://www.zhihu.com/equation?tex=S) 的元素中实现二元运算 ![[公式]](https://www.zhihu.com/equation?tex=%2A) ，很可能它就变成完整的集合 ![[公式]](https://www.zhihu.com/equation?tex=G) 。就好像说 ![[公式]](https://www.zhihu.com/equation?tex=S) 是 ![[公式]](https://www.zhihu.com/equation?tex=G) 的压缩。可能存在很多生成集。那我们的集合 ![[公式]](https://www.zhihu.com/equation?tex=t%3D%7Ba%2C+b%2C+c%7D) 和运算 ![[公式]](https://www.zhihu.com/equation?tex=%2A) 的生成集该怎么定义呢？好，看一下运算表部分，我已经用红色标出了。



![img](https://pic2.zhimg.com/80/v2-07be24607fa20388ca98917a0924296d_1440w.jpg)



你发现我标出了子集 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7Bb%2C+c%5C%7D) ，因为这两个元素可以生成完整的集合 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7Ba%2C+b%2C+c%5C%7D) 。但是事实上 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7Bb%5C%7D) 和 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7Bc%5C%7D) 也能分别生成完整的集合。例如， ![[公式]](https://www.zhihu.com/equation?tex=b%2Ab%3Dc%EF%BC%8Cb%2Ab%2Ab%3Da) （我们也可以写成 ![[公式]](https://www.zhihu.com/equation?tex=b%5E2%3Dc) 和 ![[公式]](https://www.zhihu.com/equation?tex=b%5E3%3Da) ）。同样， ![[公式]](https://www.zhihu.com/equation?tex=c%2Ac%3Db%EF%BC%8Cc%2Ac%2Ac%3Da) 。因此，通过在 ![[公式]](https://www.zhihu.com/equation?tex=b) 或者 ![[公式]](https://www.zhihu.com/equation?tex=c) 上反复使用运算 ![[公式]](https://www.zhihu.com/equation?tex=%2A) ，我们可以生成集合的三个元素。而a是集合的单位元素，所以 ![[公式]](https://www.zhihu.com/equation?tex=a%5En%3Da%2C+n+%5Cin+N) 。

因为存在两个可能性来生成元， ![[公式]](https://www.zhihu.com/equation?tex=b) 和 ![[公式]](https://www.zhihu.com/equation?tex=c) ，所以这里有两种不一样的箭头，代表不一样的运算。也就是说，我们有“ ![[公式]](https://www.zhihu.com/equation?tex=b) ”箭头和“ ![[公式]](https://www.zhihu.com/equation?tex=c) ”箭头（代表 ![[公式]](https://www.zhihu.com/equation?tex=%2Ab) 和 ![[公式]](https://www.zhihu.com/equation?tex=%2Ac) ）。为了构建群 ![[公式]](https://www.zhihu.com/equation?tex=%28G%2C+%2A%29) 的Cayley图的边集 ![[公式]](https://www.zhihu.com/equation?tex=E) ，生成集 ![[公式]](https://www.zhihu.com/equation?tex=S+%5Csubseteq+G) 是边集 ![[公式]](https://www.zhihu.com/equation?tex=E+%3D+%5C%7B%28a%2Cc%29+%5Cmid+c+%3D+a%2A+b+%5Cland+a%2Cc+%5Cin+G+%5Cland+b+%5Cin+S%5C%7D) 其中每条边都被 ![[公式]](https://www.zhihu.com/equation?tex=b%E2%88%88S) 用颜色标记。

Cayley图的结果是：



![img](https://pic2.zhimg.com/80/v2-967d377bc9942f01419062a873475581_1440w.jpg)



在这个Cayley图中，我们为两个生成元 ![[公式]](https://www.zhihu.com/equation?tex=b) 和 ![[公式]](https://www.zhihu.com/equation?tex=c) 画了两种箭头，然而，我们只需要选其一种，因为只要一种就能生成整个群。所以，一般来说，我们只要选最小的生成元来画Cayley图，这种情况下，我们只需要红色箭头就够了。

所以这个群是等边三角形的旋转对称，因为我们可以旋转三角形 ![[公式]](https://www.zhihu.com/equation?tex=120) 度而不改变它，那我们的群将 ![[公式]](https://www.zhihu.com/equation?tex=120) 度的每一次循环表示为“加”( ![[公式]](https://www.zhihu.com/equation?tex=%2A) )生成元 ![[公式]](https://www.zhihu.com/equation?tex=b) 的群运算。我们还可以加单位元素，这看起来就像不旋转它一样。在这我们能看出这基本集合 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7Ba%2C+b%2C+c%5C%7D) 中的元素“加” ![[公式]](https://www.zhihu.com/equation?tex=b) 是怎么看起来像顺时针旋转 ![[公式]](https://www.zhihu.com/equation?tex=120) 度的：



![img](https://pic4.zhimg.com/80/v2-cef49bef71d9935f04a2236e19f2c447_1440w.jpg)



这也被叫做与 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+Z_3) 同构的 ![[公式]](https://www.zhihu.com/equation?tex=3) 阶循环群。 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+Z_3) ？同构？你都在问什么？

同构主要是在保持结构的两个数学结构之间存在一对一(双射)映射。这就像它们是一样的结构但是有不同的标签一样。我们刚研究的三角形的旋转对称群是与整数模 ![[公式]](https://www.zhihu.com/equation?tex=3) （ ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+Z_3) ）同构的。模运算意味着在某一点上，运算循环回到开始。不像整数 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+Z) ，如果你一直加 ![[公式]](https://www.zhihu.com/equation?tex=1) ，你会得到一个更大的数，在模运算中，最终你加 ![[公式]](https://www.zhihu.com/equation?tex=1) 会回到起始元素(单位元素 ![[公式]](https://www.zhihu.com/equation?tex=0) )。想想时钟的时针，它是整数模 ![[公式]](https://www.zhihu.com/equation?tex=12) （ ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+Z_%7B12%7D) ），因为如果你再增加一个小时它最终会循环回来。

这是整数模 ![[公式]](https://www.zhihu.com/equation?tex=3) 的加法表:



![img](https://pic1.zhimg.com/80/v2-765ce614f8e32fa77fefa7722db629a4_1440w.jpg)



所以，在 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+Z_3) 中， ![[公式]](https://www.zhihu.com/equation?tex=1%2B1%3D2) ，但 ![[公式]](https://www.zhihu.com/equation?tex=2%2B2%3D1) ，而 ![[公式]](https://www.zhihu.com/equation?tex=1%2B2%3D0) 。整数模 ![[公式]](https://www.zhihu.com/equation?tex=x) 形成一个循环群(带有一个生成元)，其中元素 ![[公式]](https://www.zhihu.com/equation?tex=x) 和 ![[公式]](https://www.zhihu.com/equation?tex=0) 是单位元素。

好了，这就是群的基本知识，让我们继续讨论环和域。



## 环和域

现在我们继续学习一些关于环的知识，然后再轮到域。提前声明，域和环是群的特例，也就是说，它们是带有群规则和附加规则的集合。所有环都是群，所有域都是环。



> **环**：是定义了两个二元运算 ![[公式]](https://www.zhihu.com/equation?tex=%2A) 和 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbullet) 的集合 ![[公式]](https://www.zhihu.com/equation?tex=R) ，它满足下面这三个公理（环公理）：
> \1. ![[公式]](https://www.zhihu.com/equation?tex=R) 在 ![[公式]](https://www.zhihu.com/equation?tex=%2A) 运算下是交换群，也就是 ![[公式]](https://www.zhihu.com/equation?tex=%28R%2C+%2A%29) 满足群的公理。
> \2. 当运算 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbullet) 满足结合律 ![[公式]](https://www.zhihu.com/equation?tex=%28a%5Cbullet+%28b%5Cbullet+c%29%3D%28a%5Cbullet+b%29%5Cbullet+c%29) 并且 ![[公式]](https://www.zhihu.com/equation?tex=%28R%2C+%5Cbullet%29) 有单位元素 ![[公式]](https://www.zhihu.com/equation?tex=%28%5Cexists+e+%5Cin+R+%E6%BB%A1%E8%B6%B3e%5Cbullet+b%3Db%5Cbullet+e%3De%29) 时， ![[公式]](https://www.zhihu.com/equation?tex=%28R%2C+%5Cbullet%29) 构成一种叫幺半群的数学结构。
> \3. ![[公式]](https://www.zhihu.com/equation?tex=%2A) 和 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbullet) 满足分配律，即 ![[公式]](https://www.zhihu.com/equation?tex=a%2C+b%2C+c%5Cin+R) ，有：
> ![[公式]](https://www.zhihu.com/equation?tex=a%5Cbullet+%28b%2Ac%29%3D%28a%5Cbullet+b%29%2A%28a%5Cbullet+c%29) （左分配律）
> ![[公式]](https://www.zhihu.com/equation?tex=%28b%2Ac%29%5Cbullet+a%3D%28b%5Cbullet+a%29%2A%28c%5Cbullet+a%29) （右分配律）



最著名的环是 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+Z) ，因为它有两个熟悉的运算 ![[公式]](https://www.zhihu.com/equation?tex=%2B) 和 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctimes) 。因为环也是群，我们提到整数群的生成元。因为整数从 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7B+-n%E2%80%A6-3%2C+-2%2C+-1%2C+0%2C+1%2C+2%2C+3%E2%80%A6n%5C%7D) 跨度很大，所以在加法运算（ ![[公式]](https://www.zhihu.com/equation?tex=%2B) ）中，存在两个生成元 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7B+-1%2C+1%5C%7D) ，因为我不停 ![[公式]](https://www.zhihu.com/equation?tex=1%2B1%2B1%2B%E2%80%A6) 可以得到所有正整数，而 ![[公式]](https://www.zhihu.com/equation?tex=-1%2B-1%2B-1%2B%E2%80%A6) 可以得到所有负整数，并且 ![[公式]](https://www.zhihu.com/equation?tex=-1%2B1%3D0) 。



而下面是域的定义。



> **域**：是定义了两个二元运算 ![[公式]](https://www.zhihu.com/equation?tex=%2A) 和 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbullet) 的集合 ![[公式]](https://www.zhihu.com/equation?tex=F) ，对于 ![[公式]](https://www.zhihu.com/equation?tex=%5Cforall+a%2C+b%2C+c%5Cin+F) ，它满足以下条件。
> **结合律** ![[公式]](https://www.zhihu.com/equation?tex=a%2A%28b%2Ac%29%3D%28a%2Ab%29%2Ac%2C+a%5Cbullet+%28b%5Cbullet+c%29%3D%28a%5Cbullet+b%29%5Cbullet+c) **交换律** ![[公式]](https://www.zhihu.com/equation?tex=a%2Ab%3Db%2Aa%2C+a%5Cbullet+b%3Db%5Cbullet+a) **分配律** ![[公式]](https://www.zhihu.com/equation?tex=a%5Cbullet+%28b%2Ac%29%3D%28a%5Cbullet+b%29%2A%28a%5Cbullet+c%29%2C+%28b%2Ac%29%5Cbullet+a%3D%28b%5Cbullet+a%29%2A%28c%5Cbullet+a%29) **恒等** ![[公式]](https://www.zhihu.com/equation?tex=a%2A0%3Da%3D0%2Aa%2C+a%5Cbullet+1%3Da%3D1%5Cbullet+a) **逆元素** ![[公式]](https://www.zhihu.com/equation?tex=a%2A%28-a%29%3D0%3D-a%2Aa%2C+a%5Cbullet+a%5E%7B-1%7D%3D1%3Da%5E%7B-1%7D%5Cbullet+a) ，如果 ![[公式]](https://www.zhihu.com/equation?tex=a%E2%89%A00)
> 其中 ![[公式]](https://www.zhihu.com/equation?tex=0) 是 ![[公式]](https://www.zhihu.com/equation?tex=%2A) 运算下的单位元素， ![[公式]](https://www.zhihu.com/equation?tex=1) 是 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbullet) 运算下的单位元素。



显然，一个域的需求要比一个群多得多，为了注意，我一直在用符号 ![[公式]](https://www.zhihu.com/equation?tex=%2A) 和 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbullet) 作为群，环，域的二元运算，但更常用的是 ![[公式]](https://www.zhihu.com/equation?tex=%2B) 和 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctimes) 。我最初不使用这些符号的唯一原因是我想强调指出，这些符号并不仅仅适用于你所熟悉的数字，而是抽象的运算，它可以在满足要求的任何数学结构上起作用。但是现在你明白了，我们可以用更熟悉的符号。所以 ![[公式]](https://www.zhihu.com/equation?tex=%2B%3D%2A) ， ![[公式]](https://www.zhihu.com/equation?tex=%C3%97%3D%5Cbullet) ，而 ![[公式]](https://www.zhihu.com/equation?tex=a%C3%B7b%3Da%C3%97b%5E%7B-1%7D)



还记得整数集 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+Z) 是最熟悉的环吗？整数集不构成域，因为在 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+Z) 中，不是每个元素都有它的逆元素。例如，如果 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+Z) 是域，那么 ![[公式]](https://www.zhihu.com/equation?tex=5%C3%975%5E%7B-1%7D%3D1) ，然而 ![[公式]](https://www.zhihu.com/equation?tex=5%5E%7B-1%7D) 不是整数。如果我们考虑的是实数集 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+R) ，那 ![[公式]](https://www.zhihu.com/equation?tex=5%5E%7B-1%7D%3D1%2F5) 。因此，一个通过加（ ![[公式]](https://www.zhihu.com/equation?tex=%2B) ）和乘（ ![[公式]](https://www.zhihu.com/equation?tex=%C3%97) ）定义的域，同时也暗中定义了减（ ![[公式]](https://www.zhihu.com/equation?tex=-) ）和除（ ![[公式]](https://www.zhihu.com/equation?tex=%5Cdiv) ）。所以如果一个集合是域，除了单位元素，所有元素都能进行除运算；正如你从小学就知道 ![[公式]](https://www.zhihu.com/equation?tex=0) 不能作除数一样。而且这一切都是对称的，加法中 ![[公式]](https://www.zhihu.com/equation?tex=1) 的逆元素是 ![[公式]](https://www.zhihu.com/equation?tex=-1) ， ![[公式]](https://www.zhihu.com/equation?tex=2) 的逆元素是 ![[公式]](https://www.zhihu.com/equation?tex=-2) 。



![img](https://pic2.zhimg.com/80/v2-62f1724be28247b65860f8baaa27edfd_1440w.jpg)



注意到逆的对称性了吗？每一组互逆的数都与集合的中心，即为 ![[公式]](https://www.zhihu.com/equation?tex=0) ，距离相等。但因为 ![[公式]](https://www.zhihu.com/equation?tex=0) 是中心，所以它没有逆，也不能用除法来定义。



所以退一步说，群论是关于研究对称性的。任何具有对称特征的数学对象都可以被编组为群，然后用代数方法来确定哪些保持对称的操作可以在群中进行。如果我们不关心对称性，我们只想研究具有二元运算和结合律的集，那么我们可以研究幺半群。



## 为什么我们要学习群、环和域？

好的，我们学习了群、环、域的基础，但为什么呢？我已经说过了，我们需要理解群来理解那些需要计算单纯复形的同调的链群。但更普遍的是，群、环和域让我们在任何数学对象上都能使用我们熟悉的高中代数工具，这些工具满足群/环/域（不只是数字）相对宽松的需求。所以我们能对像单纯复形这种数学对象进行加，减（群），乘（环），除（域）。此外，我们还可以用一些未知的变量来解方程，包括不是数字的抽象数学对象。

直观上，向量是n维的数字列表，例如 ![[公式]](https://www.zhihu.com/equation?tex=%5B1.2%2C+4.3%2C+5.5%2C+4.1%5D) 。重要的是，我确信您已经了解了将向量相加并将它们乘以标量的基本规则。例如，

![[公式]](https://www.zhihu.com/equation?tex=%5B1.2%2C+4.3%2C+5.5%2C+4.1%5D%2B%5B1%2C+3%2C+2%2C+1%5D%3D%5B1.2%2B1%2C4.3%2B3%2C5.5%2B2%2C4.1%2B1%5D%3D%5B2.2%2C+7.3%2C+7.5%2C+5.1%5D+%5C%5C)

…总之，在向量加法中，它们必须是相同的长度，然后相应的元素相加。也就是说，每个向量的第一个元素加在一起，等等。而对于标量…

![[公式]](https://www.zhihu.com/equation?tex=2%C3%97%5B1.2%2C+4.3%2C+5.5%2C+4.1%5D%3D%5B2.2%2C+8.6%2C+11.0%2C+8.2%5D+%5C%5C)

…向量中的每个元素都乘以标量。

但等等！向量的定义没有提到任何信息说向量的元素是数量或列表。向量可以是一个符合域标准的任何有效的数学结构的集合。只要向量空间的元素可以通过域的元素（通常是实数或者整数）放大或缩小，并加在一起生成新的元素，而这新的元素仍是向量空间中的元素。

这是向量空间的正式定义，一种元素是向量的数学结构。

> **向量空间**：在域 ![[公式]](https://www.zhihu.com/equation?tex=F) 上的向量空间 ![[公式]](https://www.zhihu.com/equation?tex=V) 是被叫做向量的对象的集合，其中向量能和标量进行加减乘除运算。因为 ![[公式]](https://www.zhihu.com/equation?tex=V) 和加运算构成交换群，而对于任意 ![[公式]](https://www.zhihu.com/equation?tex=f+%5Cin+F%2C+v+%5Cin+V) ，我们都有元素 ![[公式]](https://www.zhihu.com/equation?tex=fv+%5Cin+V) （ ![[公式]](https://www.zhihu.com/equation?tex=f%C3%97v) 的结果仍在 ![[公式]](https://www.zhihu.com/equation?tex=V) 中）。标量乘法是分配律和结合律，而域的乘法恒等式在向量上恒等。

例如，最熟悉的数字向量来自域 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+R) 的向量空间。

那么，**模** 和向量空间一样，除了它是在环上定义的以外。记住，每个域都是环，所以相较于向量空间，模是一个更随意（更普遍）的数学结构。

（修改自< [http://www.math.uiuc.edu/~r-ash/Algebra/Chapter4.pdf](https://link.zhihu.com/?target=http%3A//www.math.uiuc.edu/~r-ash/Algebra/Chapter4.pdf) >）

我们还应该讨论向量空间(或模)的基础。

我们有一个有限集 ![[公式]](https://www.zhihu.com/equation?tex=S%3D%5C%7B+a%2C+b%2C+c%5C%7D) ，我们想用它来构建一个模（或者向量空间）。我们可以在环 ![[公式]](https://www.zhihu.com/equation?tex=R) 上用这个集合构建模。在这种情况下，我们的模在数学上的定义是： ![[公式]](https://www.zhihu.com/equation?tex=M+%3D+%5C%7B%28x%2A+a%2C+y%2A+b%2C+z%2A+c%29+%5Cmid+x%2Cy%2Cz+%5Cin+R%5C%7D+%5C%5C)

或者等效于

![[公式]](https://www.zhihu.com/equation?tex=M+%3D+%5C%7B%28x%2Ag%2C+y%2Ag%2C+z%2Ag%29+%5Cmid+x%2Cy%2Cz+%5Cin+R%2C+g+%5Cin+S%5C%7D+%5C%5C)

其中 ![[公式]](https://www.zhihu.com/equation?tex=%2A) 是我们模的二元“乘”运算。但因为 ![[公式]](https://www.zhihu.com/equation?tex=R) 是环，这必须还有第二个二元运算，我们可以叫它“加”，用 ![[公式]](https://www.zhihu.com/equation?tex=%2B) 表示。注意到我用的是圆括号，因此顺序很重要，例如 ![[公式]](https://www.zhihu.com/equation?tex=%28a%2C+b%2C+c%29%E2%89%A0%28b%2C+a%2C+c%29) 。

现在， ![[公式]](https://www.zhihu.com/equation?tex=M) 中的每一个元素是以 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7B+xa%2C+yb%2C+zc%5C%7D) 的形式（为了方便，我们省略了 ![[公式]](https://www.zhihu.com/equation?tex=%2A) ），因为它构成了这个模的基础。我们可以从它底层的环R中加和缩放每个元素 ![[公式]](https://www.zhihu.com/equation?tex=M) 。如果我们用整数集 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+Z) 代替环，那我们可以通过下面方法来加和缩放：

![[公式]](https://www.zhihu.com/equation?tex=m_1%2C+m_2+%7B%5Cin+M%7D+%5C%5C+m_1+%3D+%283a%2C+b%2C+5c%29+%5C%5C+m_2+%3D+%28a%2C+2b%2C+c%29+%5C%5C+m_1+%2B+m_2+%3D+%283a%2Ba%2C+b%2B2b%2C+5c%2Bc%29+%3D+%284a%2C+3b%2C+6c%29+%5C%5C+5%2Am_1+%3D+5+%2A+%283a%2C+b%2C+5c%29+%3D+%285%2A3a%2C+5%2Ab%2C+5%2A5c%29+%3D+%2815a%2C+5b%2C+25c%29)

如果我们只注意加运算，这个模也是一个群（因为每个模和向量空间都是一个群），但即使我们的生成集是一个像 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7B+a%2C+b%2C+c%5C%7D) 这样的有限集，一旦我们把它应用在像整数这样的无限环上，我们就构建了一个无限的模或向量空间。

通常情况下，一个向量空间，我们可以提出很多个基，然而，有一个数学定理告诉我们所有可能的基都是相同大小的。这就引出了维度的概念。向量空间(或模)的维数是其基的大小。因此，对于上面给出的例子，基的大小是 ![[公式]](https://www.zhihu.com/equation?tex=3) (基有三个元素)，因此模块的维度为 ![[公式]](https://www.zhihu.com/equation?tex=3) 。

再如， ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+R%5E2) 形成了一个向量空间，其中 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+R) 是实数集。这样定义： ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+R%5E2+%3D+%5C%7B%28x%2Cy%29+%5Cmid+x%2Cy+%5Cin+%5Cmathbb+R%5C%7D) 。基本上我们有所有可能的实数对组成的无限集。这个向量空间的一个基就是 ![[公式]](https://www.zhihu.com/equation?tex=%28x%2Cy%29+%5Cmid+x%2Cy+%5Cin+%5Cmathbb+R) ，这是理所应当的因为它最简单，但是没有人能阻止我们用 ![[公式]](https://www.zhihu.com/equation?tex=%282x%2B1.55%2C3y-0.718%29+%5Cmid+x%2Cy+%5Cin+%5Cmathbb+R) 作基，因为我们最终得到了相同的向量空间。但无论我们如何定义我们的基，它总是有 ![[公式]](https://www.zhihu.com/equation?tex=2) 个元素，因此它的维数是 ![[公式]](https://www.zhihu.com/equation?tex=2) 。

当我有一个像 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+R%5E2) 这样 ![[公式]](https://www.zhihu.com/equation?tex=2) 维的向量空间，我们可以像这样分离它们的成分：

![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+R_x+%3D+%5C%7B%28x%2C+0%29+%5Cmid+x+%5Cin+%5Cmathbb+R%5C%7D+%5C%5C+%5Cmathbb+R_y+%3D+%5C%7B%280%2C+y%29+%5Cmid+y+%5Cin+%5Cmathbb+R%5C%7D+%5C%5C+%5Cmathbb+R%5E2+%3D+%5Cmathbb+R_x+%5Coplus+%5Cmathbb+R_y)

我们引入了新的符号，叫做 **直和** ![[公式]](https://www.zhihu.com/equation?tex=%5Coplus) ，用来表示通过像 ![[公式]](https://www.zhihu.com/equation?tex=%28x%2C0%29%2B%280%2Cy%29%3D%28x%2B0%2C0%2By%29%3D%28x%2Cy%29+%5Cmid+x%2Cy+%5Cin+%5Cmathbb+R) 这样构建向量空间的维数的过程。因此我们可以简单的表示为 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+R%5E2+%3D+%5Cmathbb+R+%5Coplus+%5Cmathbb+R) 。

我们也可以说 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+R%5E2) 的基是集合 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7B+%281%2C+0%29%2C+%280%2C+1%29%5C%7D) 的生成子空间，记作 ![[公式]](https://www.zhihu.com/equation?tex=span%5C%7B+%281%2C+0%29%2C+%280%2C+1%29%5C%7D) 或者有时甚至更简单地用尖括号 ![[公式]](https://www.zhihu.com/equation?tex=%3C%281%2C+0%29%2C+%280%2C+1%29%3E) 表示。 ![[公式]](https://www.zhihu.com/equation?tex=span%5C%7B+%281%2C+0%29%2C+%280%2C+1%29%5C%7D) 是“由基 ![[公式]](https://www.zhihu.com/equation?tex=%281%2C+0%29) 和 ![[公式]](https://www.zhihu.com/equation?tex=%280%2C+1%29) 的所有线性组合构成的集合”的简写。

什么是线性组合？简单地说， ![[公式]](https://www.zhihu.com/equation?tex=x) 和 ![[公式]](https://www.zhihu.com/equation?tex=y) 的线性组合是 ![[公式]](https://www.zhihu.com/equation?tex=ax%2Bby) 形式的任何表达，其中 ![[公式]](https://www.zhihu.com/equation?tex=a%2C+b) 是域 ![[公式]](https://www.zhihu.com/equation?tex=F) 中的常数。

因此 ![[公式]](https://www.zhihu.com/equation?tex=%281%2C+0%29) 和 ![[公式]](https://www.zhihu.com/equation?tex=%280%2C+1%29) 的其中一个可能的线性组合是：

![[公式]](https://www.zhihu.com/equation?tex=5%281%2C+0%29%2B2%280%2C+1%29%3D%285%2A1%2C+5%2A0%29%2B%282%2A0%2C+2%2A1%29%3D%285%2C+0%29%2B%280%2C+2%29%3D%285%2B0%2C+0%2B2%29%3D%285%2C+2%29) 。但 ![[公式]](https://www.zhihu.com/equation?tex=%281%2C+0%29) 和 ![[公式]](https://www.zhihu.com/equation?tex=%280%2C+1%29) 的所有线性组合可以用 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7Ba%281%2C0%29+%2B+b%280%2C1%29+%5Cmid+a%2Cb+%5Cin+%5Cmathbb+R%5C%7D) 来表达，而这等同于 ![[公式]](https://www.zhihu.com/equation?tex=span%5C%7B+%281%2C+0%29%2C+%280%2C+1%29%5C%7D) 或者 ![[公式]](https://www.zhihu.com/equation?tex=%3C%281%2C+0%29%2C+%280%2C+1%29%3E) 。这个集合所有的实数都是由 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+R%5E2) 表示的。

关于向量空间的基，重要的是它们必须是线性无关的。这意味着一个元素不能被表示成另一个元素的线性组合。例如，基本元素 ![[公式]](https://www.zhihu.com/equation?tex=%281%2C+0%29) 不能被 ![[公式]](https://www.zhihu.com/equation?tex=%280%2C+1%29) 表示。 ![[公式]](https://www.zhihu.com/equation?tex=%5Cnot%5Cexists+a%2Cb+%5Cin+%5Cmathbb+R+%5Cland+a%2Cb%5Cneq+0) ，使得 ![[公式]](https://www.zhihu.com/equation?tex=a%280%2C1%29+%2B+b%281%2C0%29+%3D+%281%2C0%29) 。

因此总结，向量空间的基 ![[公式]](https://www.zhihu.com/equation?tex=V) 包含了元素 ![[公式]](https://www.zhihu.com/equation?tex=B) 的集合，使得每个元素 ![[公式]](https://www.zhihu.com/equation?tex=b%5Cin+B) 是线性无关的，而 ![[公式]](https://www.zhihu.com/equation?tex=B) 的生成子空间生成整个向量 ![[公式]](https://www.zhihu.com/equation?tex=V) 。因此向量空间的维度 ![[公式]](https://www.zhihu.com/equation?tex=dim%28V%29) 是 ![[公式]](https://www.zhihu.com/equation?tex=B) 中元素的个数。

(参考: The Napkin Project by Evan Chen <[Evan Chen &bullet; Napkin](https://link.zhihu.com/?target=http%3A//www.mit.edu/~evanchen/napkin.html)>)



## 链群

哎，好吧，我们克服了很多障碍，但是现在我们回到我们真正关心的问题：算出一个单纯复形的同调群。您可能还记得，我们没有讨论单纯复形的链群。我不想重复所有的事情，所以，如果你忘记了，就把这部分重新读一遍。我会等你的…

让 ![[公式]](https://www.zhihu.com/equation?tex=S%3D%5C%7B%5C%7Ba%5C%7D%2C+%5C%7Bb%5C%7D%2C+%5C%7Bc%5C%7D%2C+%5C%7Bd%5C%7D%2C+%5C%7Ba%2C+b%5C%7D%2C+%5C%7Bb%2C+c%5C%7D%2C+%5C%7Bc%2C+a%5C%7D%2C+%5C%7Bc%2C+d%5C%7D%2C+%5C%7Bd%2C+b%5C%7D%2C+%5C%7Ba%2C+b%2C+c%5C%7D%5C%7D) 成为一个从一些点云（如数据集）中构建的抽象的单纯复形（如下图所示）。 ![[公式]](https://www.zhihu.com/equation?tex=n) 链 ![[公式]](https://www.zhihu.com/equation?tex=%EF%BC%88C_n%28S%29%EF%BC%89) 是 ![[公式]](https://www.zhihu.com/equation?tex=n) 维单纯形 ![[公式]](https://www.zhihu.com/equation?tex=S) 的子集。比如 ![[公式]](https://www.zhihu.com/equation?tex=C_1%28S%29%3D+%5C%7Ba%2C+b%5C%7D%2C+%5C%7Bb%2C+c%5C%7D%2C+%5C%7Bc%2C+a%5C%7D%2C+%5C%7Bc%2C+d%5C%7D%2C+%5C%7Bd%2C+b%5C%7D%5C%7D) 和 ![[公式]](https://www.zhihu.com/equation?tex=C_2%28S%29%3D%5C%7Ba%2C+b%2C+c%5C%7D) 。

![img](https://pic2.zhimg.com/80/v2-37142a90180939db2e405b205242729d_1440w.jpg)



现在，如果我们给它定义一个满足群公理的二元加运算，让它可以变成一个链群。有了这个结构，我们可以把 ![[公式]](https://www.zhihu.com/equation?tex=C_n%28S%29) 中的 ![[公式]](https://www.zhihu.com/equation?tex=n) 维单纯形加在一起。更确切地说， ![[公式]](https://www.zhihu.com/equation?tex=n) 链群是群、环或域F中n链带系数地相加。我准备用和n链一样的 ![[公式]](https://www.zhihu.com/equation?tex=C_n) 符号来表示一个链群。 ![[公式]](https://www.zhihu.com/equation?tex=C_n%28S%29+%3D+%5Csum+a_i+%5Csigma_i+%5C%5C)

其中 ![[公式]](https://www.zhihu.com/equation?tex=%CF%83_i) 指的是 ![[公式]](https://www.zhihu.com/equation?tex=n) 链 ![[公式]](https://www.zhihu.com/equation?tex=C_n) 的第 ![[公式]](https://www.zhihu.com/equation?tex=i) 个单纯形， ![[公式]](https://www.zhihu.com/equation?tex=a_i) 是群、环或域的相应系数，而 ![[公式]](https://www.zhihu.com/equation?tex=S) 是一开始的单纯复形。

技术上，任何域/群/环都可以被用来为链组提供系数，然而，就我们目的来说，最好用最简单的群是循环群 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+Z_2) ，即模 ![[公式]](https://www.zhihu.com/equation?tex=2) 的整数。 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+Z_2) 只包含 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7B+0%2C+1%5C%7D) 使得 ![[公式]](https://www.zhihu.com/equation?tex=1%2B1%3D0) ，而且是域，因为我们可以定义一个加法和乘法运算来满足域的公理。这非常有用，因为我们只是希望能够说，在我们的 ![[公式]](https://www.zhihu.com/equation?tex=n) 链中存在一个单纯形(即它的系数为 ![[公式]](https://www.zhihu.com/equation?tex=1) )或者不存在(系数为 ![[公式]](https://www.zhihu.com/equation?tex=0) )，如果我们有一个重复的单纯形，当我们把它们加在一起时，它们就会消掉。这就是我们想要的性质。你可能会反对，说 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+Z_2) 不是个群，因为它没有逆，例如 ![[公式]](https://www.zhihu.com/equation?tex=-1) ，但事实上它是有的，例如， ![[公式]](https://www.zhihu.com/equation?tex=a) 的逆是 ![[公式]](https://www.zhihu.com/equation?tex=a) 。即在 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+Z_2) 中， ![[公式]](https://www.zhihu.com/equation?tex=a%3D-a) 因为 ![[公式]](https://www.zhihu.com/equation?tex=a%2Ba%3D0) 。这就是逆存在的所有要求，你只需要在你的团队中有某个元素，满足 ![[公式]](https://www.zhihu.com/equation?tex=%5Cexists+b+%5Cin+G%2C+a%2Bb%3D0%2C+%5Cforall+b+%5Cin+G) （ ![[公式]](https://www.zhihu.com/equation?tex=G) 是一个群）。

如果我们把 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+Z_2) 当作群系数，那我们基本上可以忽略单纯形的方向。这样就方便多了。但为了完整性，我想要融入方向，因为在学术论文和商业上，我见过很多人把整个整数集 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+Z) 当作系数；如果我们用一个像 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+Z) 一样带负数的域，那我们的单纯形需要有导向，就像 ![[公式]](https://www.zhihu.com/equation?tex=%5Ba%2C+b%5D%E2%89%A0%5Bb%2C+a%5D) 。这是因为如果我们用 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+Z) ，那么 ![[公式]](https://www.zhihu.com/equation?tex=%5Ba%2C+b%5D%3D-%5Bb%2C+a%5D) ，因为 ![[公式]](https://www.zhihu.com/equation?tex=%5Ba%2C+b%5D%2B%5Bb%2C+a%5D%3D0) 。

记住，我们的最终目标是在数学上找到单纯复形中的连通分支和 ![[公式]](https://www.zhihu.com/equation?tex=n) 维的圈圈。我们上面的单纯复形 ![[公式]](https://www.zhihu.com/equation?tex=S) ，目测，拥有一个连通分支和一个 ![[公式]](https://www.zhihu.com/equation?tex=2) 维的圈或者孔。记住单纯形 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7Ba%2C+b%2C+c%5C%7D+%5Cin+S) 是“被填充”的，中间没有洞，这是一个固体。

我们现在开始定义边界映射。看起来一个无导向的 ![[公式]](https://www.zhihu.com/equation?tex=n+) 维单纯形 ![[公式]](https://www.zhihu.com/equation?tex=X) 的边界映射（或者简称为边界）是 ![[公式]](https://www.zhihu.com/equation?tex=X) 的 ![[公式]](https://www.zhihu.com/equation?tex=%7B+%7BX%7D+%5Cchoose+%7Bn-1%7D%7D) 子集的集合。也就是说，边界是所有 ![[公式]](https://www.zhihu.com/equation?tex=X) 的 ![[公式]](https://www.zhihu.com/equation?tex=%EF%BC%88n-1%EF%BC%89) 子集的集合。例如， ![[公式]](https://www.zhihu.com/equation?tex=%5C%7B+a%2C+b%2C+c%5C%7D) 的边界是 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7B%5C%7Ba%2C+b%5C%7D%2C+%5C%7Ba%2C+c%5C%7D%2C+%5C%7Bb%2C+c%5C%7D%5C%7D) 。

让我们给出一个适用于定向单纯形的更精确的定义，并提供一些符号。

> **边界**：带有方向向量 ![[公式]](https://www.zhihu.com/equation?tex=%5Bv_0%2Cv_1%2Cv_2%2C...v_n%5D) 的 ![[公式]](https://www.zhihu.com/equation?tex=n) 维单纯形 ![[公式]](https://www.zhihu.com/equation?tex=X) 的边界（ ![[公式]](https://www.zhihu.com/equation?tex=%5Cpartial%28X%29) ）是： ![[公式]](https://www.zhihu.com/equation?tex=%5Cpartial%28X%29+%3D+%5Csum%5E%7Bn%7D_%7Bi%3D0%7D%28-1%29%5E%7Bi%7D%5Bv_0%2C+v_1%2C+v_2%2C+%5Chat%7Bv_i%7D+...+v_n%5D+%5C%5C)
> 其中第 ![[公式]](https://www.zhihu.com/equation?tex=i) 个向量被从序列中移除。
> 单个顶点的边界是 ![[公式]](https://www.zhihu.com/equation?tex=0) ， ![[公式]](https://www.zhihu.com/equation?tex=%5Cpartial%28%5Bv_i%5D%29+%3D+0) 。

例如，如果 ![[公式]](https://www.zhihu.com/equation?tex=X) 是 ![[公式]](https://www.zhihu.com/equation?tex=2) 维单纯形 ![[公式]](https://www.zhihu.com/equation?tex=%5Ba%2Cb%2Cc%5D) ，那么 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpartial%28X%29+%3D+%5Bb%2Cc%5D+%2B+%28-1%29%5Ba%2Cc%5D+%2B+%5Ba%2Cb%5D+%3D+%5Bb%2Cc%5D+%2B+%5Bc%2Ca%5D+%2B+%5Ba%2Cb%5D) 。

让我们来看看如何从边界的概念中得到上面 2 维复形中的一个简单的圈。我们看到 ![[公式]](https://www.zhihu.com/equation?tex=%5Bb%2Cc%5D%2B%5Bc%2Cd%5D%2B%5Bd%2Cb%5D) 是形成圈或者循环的 ![[公式]](https://www.zhihu.com/equation?tex=1) 维单纯形。如果我们得到这个带有系数域 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+Z) 的集合的边界，那么

![[公式]](https://www.zhihu.com/equation?tex=%5Cpartial%28%5Bb%2Cc%5D+%2B+%5Bc%2Cd%5D+%2B+%5Bd%2Cb%5D%29+%3D+%5Cpartial%28%5Bb%2Cc%5D%29+%2B+%5Cpartial%28%5Bc%2Cd%5D%29+%2B+%5Cpartial%28%5Bd%2Cb%5D%29+%5C%5C+%5Cpartial%28%5Bb%2Cc%5D%29+%2B+%5Cpartial%28%5Bc%2Cd%5D%29+%2B+%5Cpartial%28%5Bd%2Cb%5D%29+%3D+%5Bb%5D+%2B+%28-1%29%5Bc%5D+%2B+%5Bc%5D+%2B+%28-1%29%5Bd%5D+%2B+%5Bd%5D+%2B+%28-1%29%5Bb%5D+%5C%5C+%5Crequire%7Bcancel%7D+%5Ccancel%7B%5Bb%5D%7D+%2B+%5Ccancel%7B%28-1%29%5Bb%5D%7D+%2B+%5Ccancel%7B%28-1%29%5Bc%5D%7D+%2B+%5Ccancel%7B%5Bc%5D%7D+%2B+%5Ccancel%7B%28-1%29%5Bd%5D%7D+%2B+%5Ccancel%7B%5Bd%5D%7D+%3D+0)

这使我们有了一个更普遍的属性，一个 ![[公式]](https://www.zhihu.com/equation?tex=p) 圈是 ![[公式]](https://www.zhihu.com/equation?tex=C_n) 中的 ![[公式]](https://www.zhihu.com/equation?tex=n) 链，其中 ![[公式]](https://www.zhihu.com/equation?tex=C_n) 的边界 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpartial%28C_n%29+%3D+0) 。

也就是说，为了找到链群 ![[公式]](https://www.zhihu.com/equation?tex=C_n) 中的 ![[公式]](https://www.zhihu.com/equation?tex=p) 圈，我们需要解决代数方程 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpartial%28C_n%29+%3D+0) ，而它的解就是 ![[公式]](https://www.zhihu.com/equation?tex=p) 圈。不用担心，当我们过一些示例时，这些都会变得有意义和更直观的。

一个重要的结果是边界的边界总是 ![[公式]](https://www.zhihu.com/equation?tex=0) ，即 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpartial_n+%5Cpartial_%7Bn-1%7D+%3D+0) 。



## 链复形



我们刚看到边界运算是如何分配的，例如，两个单纯形 ![[公式]](https://www.zhihu.com/equation?tex=%5Csigma_1%2C+%5Csigma_2+%5Cin+S) ![[公式]](https://www.zhihu.com/equation?tex=%5Cpartial%28%5Csigma_1+%2B+%5Csigma_2%29+%3D+%5Cpartial%28%5Csigma_1%29+%2B+%5Cpartial%28%5Csigma_2%29+%5C%5C)

> **链复形**： ![[公式]](https://www.zhihu.com/equation?tex=S) 是一个单纯 ![[公式]](https://www.zhihu.com/equation?tex=p) 复形。 ![[公式]](https://www.zhihu.com/equation?tex=C_n%28S%29) 是 ![[公式]](https://www.zhihu.com/equation?tex=S) 的 ![[公式]](https://www.zhihu.com/equation?tex=n) 链， ![[公式]](https://www.zhihu.com/equation?tex=n%E2%89%A4p) ，链复形 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathscr+C%28S%29) 是 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathscr+C%28S%29+%3D+%5Csum%5E%7Bp%7D_%7Bn%3D0%7D%5Cpartial%28C_n%28S%29%29+%5C%5C)
> 换句话说
> ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathscr+C%28S%29+%3D+%5Cpartial%28C_0%28S%29%29+%2B+%5Cpartial%28C_1%28S%29%29+%5C+%2B+%5C+...+%5C+%2B+%5C+%5Cpartial%28C_p%28S%29%29+%5C%5C)

现在我们可以定义怎么在单纯复形中找到 ![[公式]](https://www.zhihu.com/equation?tex=p) 圈。

> **核**： ![[公式]](https://www.zhihu.com/equation?tex=%5Cpartial%28C_n%29) 的核（记作 ![[公式]](https://www.zhihu.com/equation?tex=Ker%28%5Cpartial%28C_n%29%29) ）是 ![[公式]](https://www.zhihu.com/equation?tex=n) 链 ![[公式]](https://www.zhihu.com/equation?tex=Z_n+%5Csubseteq+C_n) 的群，其中 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpartial%28Z_n%29+%3D+0) 。

我们已经快好了，我们需要再多两个概念，我们就可以定义单纯同调。

> **边界的像**：边界 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpartial_n) （一些 ![[公式]](https://www.zhihu.com/equation?tex=n) 链的边界）的像 ![[公式]](https://www.zhihu.com/equation?tex=Im%28%5Cpartial_n%29) 是边界的集合。

例如，如果 ![[公式]](https://www.zhihu.com/equation?tex=1) 链是 ![[公式]](https://www.zhihu.com/equation?tex=C_1+%3D+%5C%7B%5Bv_0%2C+v_1%5D%2C+%5Bv_1%2C+v_2%5D%2C+%5Bv_2%2C+v_0%5D%5C%7D)

然后 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpartial_1+%3D+%5Bv_0%5D+%2B+%28-1%29%5Bv_1%5D+%2B+%5Bv_1%5D+%2B+%28-1%29%5Bv_2%5D+%2B+%5Bv_2%5D+%2B+%28-1%29%5Bv_0%5D)

![[公式]](https://www.zhihu.com/equation?tex=Im+%5Cpartial_1+%3D+%5C%7B%5Bv_0-v_1%5D%2C%5Bv_1-v_2%5D%2C%5Bv_2-v_0%5D%5C%7D)

因此 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpartial_n) 和 ![[公式]](https://www.zhihu.com/equation?tex=Im%5Cpartial_n) 唯一的不同是边界的像是集合形式的，而边界是多项式形式的。

> **第 ![[公式]](https://www.zhihu.com/equation?tex=n) 个同调群**：第 ![[公式]](https://www.zhihu.com/equation?tex=n) 个同调群 ![[公式]](https://www.zhihu.com/equation?tex=H_n) 定义为 ![[公式]](https://www.zhihu.com/equation?tex=H_n%3DKer%5Cpartial_n%2FIm%5Cpartial_%7Bn%2B1%7D) 。



> **连通数**：第 ![[公式]](https://www.zhihu.com/equation?tex=n) 个连通数 ![[公式]](https://www.zhihu.com/equation?tex=b_n) 定义为 ![[公式]](https://www.zhihu.com/equation?tex=H_n) 的维度， ![[公式]](https://www.zhihu.com/equation?tex=b_n+%3D+dim%28H_n%29) 。



## 更多群论

在这个点，我们再次掉进一个坑里，需要一些更多的说明和铺垫。我随便用了 ![[公式]](https://www.zhihu.com/equation?tex=%EF%BC%8F) 来定义同调群 ![[公式]](https://www.zhihu.com/equation?tex=Ker%5Cpartial_n%2FIm%5Cpartial_%7Bn%2B1%7D) ，这个符号的数学用法是说，对于一些群 ![[公式]](https://www.zhihu.com/equation?tex=G) 和 ![[公式]](https://www.zhihu.com/equation?tex=G) 的子群 ![[公式]](https://www.zhihu.com/equation?tex=H) ， ![[公式]](https://www.zhihu.com/equation?tex=G%2FH) 是商群。那什么是商群？我们需要学习更多群论内容。不幸的是，这很难，但是我会尽力让它简单易懂。

> **商群**：对于群 ![[公式]](https://www.zhihu.com/equation?tex=G) 和 ![[公式]](https://www.zhihu.com/equation?tex=G) 的正规子群 ![[公式]](https://www.zhihu.com/equation?tex=N) （记作 ![[公式]](https://www.zhihu.com/equation?tex=N+%5Cunlhd+G) ）， ![[公式]](https://www.zhihu.com/equation?tex=G) 在 ![[公式]](https://www.zhihu.com/equation?tex=N) 上的商群，写作 ![[公式]](https://www.zhihu.com/equation?tex=G%2FN) ，读作“ ![[公式]](https://www.zhihu.com/equation?tex=G) 以 ![[公式]](https://www.zhihu.com/equation?tex=N) 为模”，是 ![[公式]](https://www.zhihu.com/equation?tex=N) 在 ![[公式]](https://www.zhihu.com/equation?tex=G) 中所有陪集的集合。

(来源：Weisstein, Eric W. "Quotient Group." From MathWorld--A Wolfram Web Resource. [Quotient Group -- from Wolfram MathWorld](https://link.zhihu.com/?target=http%3A//mathworld.wolfram.com/QuotientGroup.html))

现在你可以忽略一个正规子群的意思，因为我们将在 TDA 中处理的所有群都是交换群，而交换群的所有子群都是正规的。但这个定义是通过一个叫陪集的定义定义的。那什么是陪集呢？

> **陪集**：对于一个群 ![[公式]](https://www.zhihu.com/equation?tex=%28G%2C%2A%29) ，考虑一个子群 ![[公式]](https://www.zhihu.com/equation?tex=%28H%2C%2A%29) 以及它里面的元素 ![[公式]](https://www.zhihu.com/equation?tex=h_i) 和 ![[公式]](https://www.zhihu.com/equation?tex=G) 的元素 ![[公式]](https://www.zhihu.com/equation?tex=x) ，那么 ![[公式]](https://www.zhihu.com/equation?tex=x%2Ah_i%28%5Cforall+i%3D1%2C2%2C...%29) 组成子群 ![[公式]](https://www.zhihu.com/equation?tex=H) 关于 ![[公式]](https://www.zhihu.com/equation?tex=x) 的左陪集。

(来自：Weisstein, Eric W. "Left Coset." From MathWorld--A Wolfram Web Resource. [Left Coset -- from Wolfram MathWorld](https://link.zhihu.com/?target=http%3A//mathworld.wolfram.com/LeftCoset.html))

那么我们可以思考子集 ![[公式]](https://www.zhihu.com/equation?tex=H%E2%89%A4G) 关于一些元素 ![[公式]](https://www.zhihu.com/equation?tex=x%5Cin+G) 的左（或右）陪集是什么，而这只是一个陪集，但如果我们有所有左陪集的集合（即对每个元素 ![[公式]](https://www.zhihu.com/equation?tex=x%5Cin+G) 的陪集），那么我们就得到一个商群 ![[公式]](https://www.zhihu.com/equation?tex=G%2FH) 。

对我们来说，我们只需要关注左陪集，因为 TDA 只包括交换群，而对于交换群，左陪集和右陪集是等价的。（我们将举一个非交换群的例子）

我们会重新考虑等边三角形和其对称性，以更好的了解的子群、 商群和陪集。

![img](https://pic2.zhimg.com/80/v2-8ec317245d296e13f2608d76da5e6db1_1440w.jpg)



请记住，通过简单的可视化，我们确定了可以在保持其结构的等边三角形上执行的操作类型：我们可以将它旋转 ![[公式]](https://www.zhihu.com/equation?tex=0) 、 ![[公式]](https://www.zhihu.com/equation?tex=120) 、 ![[公式]](https://www.zhihu.com/equation?tex=240) 度，我们可以通过三条对称的边翻转它。任何其他的操作，如旋转 ![[公式]](https://www.zhihu.com/equation?tex=1) 度，都会在空间中产生不同的结构，例如， ![[公式]](https://www.zhihu.com/equation?tex=2) 维欧几里得空间。

我们可以构建一个包括这六种操作的集合： ![[公式]](https://www.zhihu.com/equation?tex=S+%3D+%5Ctext%7B%7B%24rot_0%24%2C+%24rot_%7B120%7D%24%2C+%24rot_%7B240%7D%24%2C+%24ref_a%24%2C+%24ref_b%24%2C+%24ref_c%24%7D%7D) ...其中 ![[公式]](https://www.zhihu.com/equation?tex=rot_0) （等等）代表以中心将这个三角形旋转0度，而 ![[公式]](https://www.zhihu.com/equation?tex=ref_a) 代表如上面图片，沿 ![[公式]](https://www.zhihu.com/equation?tex=a) 边翻转。

例如，我们可以对这个三角形进行两种操作，如 ![[公式]](https://www.zhihu.com/equation?tex=rot_%7B120%7D) 和 ![[公式]](https://www.zhihu.com/equation?tex=ref_a) 。



![img](https://pic3.zhimg.com/80/v2-0ac1df8aa81d2f6cfbed5145f45e2006_1440w.jpg)



所以 ![[公式]](https://www.zhihu.com/equation?tex=S) 形成有效的群了吗？它确实做到了，我们为它包含的每一对元素定义一个二元运算。而对任意 ![[公式]](https://www.zhihu.com/equation?tex=S) 中的两个元素进行 ![[公式]](https://www.zhihu.com/equation?tex=a%2Ab) 运算是指“先进行 ![[公式]](https://www.zhihu.com/equation?tex=a) 操作，然后进行 ![[公式]](https://www.zhihu.com/equation?tex=b) 操作”。 ![[公式]](https://www.zhihu.com/equation?tex=S) 的元素是我们对三角形采取的操作。我们可以构建一个乘法表（或者Cayley表）来展示应用于每对元素的操作的结果。

这是Cayley表。

![img](https://pic2.zhimg.com/80/v2-66213a7bd224e7ede48b200e89dbca49_1440w.jpg)

注意这定义了非交换(非abelian)群，因为 ![[公式]](https://www.zhihu.com/equation?tex=a%2Ab+%5Cneq+b%2Aa) 。

现在我们可以使用Cayley表来构建一个Cayley图，并可视化群 ![[公式]](https://www.zhihu.com/equation?tex=S) 。让我们回忆一下如何构建一个Cayley图。我们首先从顶点（或结点）开始，在群 ![[公式]](https://www.zhihu.com/equation?tex=S) 中一个对应六个动作。然后我们需要算出这个群的最小生成元，也就是 ![[公式]](https://www.zhihu.com/equation?tex=S) 最小的子集，在不同组合和重复应用群操作 ![[公式]](https://www.zhihu.com/equation?tex=%2A) 后会生成完整的含有 ![[公式]](https://www.zhihu.com/equation?tex=6) 个元素的集合 ![[公式]](https://www.zhihu.com/equation?tex=S) 。你只需要 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7Brot_%7B120%7D%2C+ref_a%5C%7D) 来生成整个集合，因此 ![[公式]](https://www.zhihu.com/equation?tex=2) 个元素的子集是最小生成集。

现在，生成集中的每个元素被分配了不同颜色的箭头，因此，从结点 ![[公式]](https://www.zhihu.com/equation?tex=a) 开始跟随特定的箭头到另一个元素 ![[公式]](https://www.zhihu.com/equation?tex=b) 说明 ![[公式]](https://www.zhihu.com/equation?tex=a%2Ag%3Db) ，其中 ![[公式]](https://www.zhihu.com/equation?tex=g) 是生成集中的元素。因此对于 ![[公式]](https://www.zhihu.com/equation?tex=S) ，我们将有一个有两种不同类型的箭头的图，而我将给 ![[公式]](https://www.zhihu.com/equation?tex=rot_%7B120%7D) 箭头涂蓝和给 ![[公式]](https://www.zhihu.com/equation?tex=ref_a) 涂红。然后我们使用来自上面的Cayley表来连接节点和两种类型的箭头。

这是得到的Cayley图:



![img](https://pic4.zhimg.com/80/v2-c1fd931468f2b00da1d02c30fc03a223_1440w.jpg)



原来这个群是最小的有限非交换群，叫做“ ![[公式]](https://www.zhihu.com/equation?tex=6) 阶二面体群”，除了在等边三角形上的对称动作外，它还可以用来表示其他一些东西。

我们将参考这两个Cayley表和Cayley图，以得到先前我们给子群、陪集和商群定义的感觉。

让我们回到子群的概念。群 ![[公式]](https://www.zhihu.com/equation?tex=%28G%2C%2A%29) 的子群 ![[公式]](https://www.zhihu.com/equation?tex=%28H%2C%2A%29) （一般用 ![[公式]](https://www.zhihu.com/equation?tex=H%5Cleq+G) 表示）只是 ![[公式]](https://www.zhihu.com/equation?tex=G) 和同一满足群公理的运算 ![[公式]](https://www.zhihu.com/equation?tex=%2A) 的子集。举个例子，每个群都有一个只有单位元素的子群（任何有效的子群都需要包含单位元素以满足组公理）。

考虑子群 ![[公式]](https://www.zhihu.com/equation?tex=W+%5Cleq+S+%3D+%5C%7Brot_0%2C+rot_%7B120%7D%2C+rot_%7B240%7D%5C%7D) 。它是一个有效的子群吗？是的，因为它是 ![[公式]](https://www.zhihu.com/equation?tex=S) 的子集，包含单位元素，它是相联的，每个元素都有一个逆。在这个例子中，子群 ![[公式]](https://www.zhihu.com/equation?tex=W+%5Cleq+S) 形成了Cayley图中的外部循环（绿色结点）：



![img](https://pic3.zhimg.com/80/v2-9f58a6ddd1a00807996a63a06bc47d42_1440w.jpg)



好的，一个子群相当直接了当。那么陪集呢？参照前面给出的定义，陪集是指一个特定的子群。因此我们考虑子群 ![[公式]](https://www.zhihu.com/equation?tex=W+%5Cleq+S) ，这个子群的陪集是什么？之前说过，我们只需要考虑左陪集，因为在 TDA 中这些群都是可交换的，这是事实，但等边三角形的对称群不是交换群，所以它们的左右陪集事实上是不一样的。我们只是用这个三角形来学习群论，一旦我们回到持续同调的链群，我们就会回到交换群。

回忆子集 ![[公式]](https://www.zhihu.com/equation?tex=H+%5Cleq+S) 的左陪集表示为 ![[公式]](https://www.zhihu.com/equation?tex=xH+%3D+%5C%7Bx%2A+h+%5Cmid+%5Cforall+h+%5Cin+H%3B+%5Cforall+x+%5Cin+G%5C%7D) 。为了完整性，右陪集表示为 ![[公式]](https://www.zhihu.com/equation?tex=Hx+%3D+%5C%7B%7Bh%7D+%2A+x+%5Cmid+%5Cforall+h+%5Cin+H%3B+%5Cforall+x+%5Cin+G%5C%7D) 。

回到我们的三角对称，群 ![[公式]](https://www.zhihu.com/equation?tex=S) 和它的子群 ![[公式]](https://www.zhihu.com/equation?tex=W) 。记得 ![[公式]](https://www.zhihu.com/equation?tex=W+%5Cleq+S+%3D+%5C%7Brot_0%2C+rot_%7B120%7D%2C+rot_%7B240%7D%5C%7D) 。为了计算出左陪集，我们将通过选择 ![[公式]](https://www.zhihu.com/equation?tex=x%5Cin+S) 开始，其中 ![[公式]](https://www.zhihu.com/equation?tex=x) 不在子群 ![[公式]](https://www.zhihu.com/equation?tex=W) 中。然后我们将拿 ![[公式]](https://www.zhihu.com/equation?tex=W) 中的每个元素乘以 ![[公式]](https://www.zhihu.com/equation?tex=x) 。我们将从 ![[公式]](https://www.zhihu.com/equation?tex=x+%3D+ref_a) 开始。

所以 ![[公式]](https://www.zhihu.com/equation?tex=ref_a+%5Cstar+%5C%7Brot_0%2C+rot_%7B120%7D%2C+rot_%7B240%7D%5C%7D+%3D+%5C%7Bref_a+%5Cstar+rot_0%2C+ref_a+%5Cstar+rot_%7B120%7D%2C+ref_a+%5Cstar+rot_%7B240%7D%5C%7D+%3D+%5C%7Bref_a%2C+ref_b%2C+ref_c%5C%7D) 。所以对于 ![[公式]](https://www.zhihu.com/equation?tex=ref_a) 的左陪集是集合 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7Bref_a%2C+ref_b%2C+ref_c%5C%7D) 。现在，我们应该对另一个元素 ![[公式]](https://www.zhihu.com/equation?tex=x+%5Cin+S%2C+x+%5Cnot%5Cin+W) 进行同样的操作，但如果我们这样做，我们会得到同样的集合： ![[公式]](https://www.zhihu.com/equation?tex=%5C%7Bref_a%2C+ref_b%2C+ref_c%5C%7D) 。因此我们只有一个左陪集。

对于我们这个子群，左右陪集是一样的，右陪集： ![[公式]](https://www.zhihu.com/equation?tex=%5C%7Brot_0%5Cstar+ref_a%2C+rot_%7B120%7D%5Cstar+ref_a%2C+rot_%7B240%7D%5Cstar+ref_a+%5C%7D+%3D+%5C%7Bref_a%2C+ref_b%2C+ref_c%5C%7D)

(参考: < [http://www.math.clemson.edu/~macaule/classes/m16_math4120/slides/math4120_lecture-3-02_handout.pdf](https://link.zhihu.com/?target=http%3A//www.math.clemson.edu/~macaule/classes/m16_math4120/slides/math4120_lecture-3-02_handout.pdf) >)

有趣的是，由于所有的Cayley图本身都具有对称性，所以在一般情况下，子群的左陪集就像在Cayley图中的子群的复制体一样。如果你考虑子集 ![[公式]](https://www.zhihu.com/equation?tex=W+%5Cleq+S+%3D+%5C%7Brot_0%2C+rot_%7B120%7D%2C+rot_%7B240%7D%5C%7D) ，它在Cayley图中形成这个外环，而左陪集则是构成图内部“环”的顶点集合。因此，这就像他们之间互相复制一样。这是另一个子群是 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7Brot_0%2C+ref_a%5C%7D) 的例子：



![img](https://pic3.zhimg.com/80/v2-e958211a84c528d608be2b1d79bb1176_1440w.jpg)



因此我们开始看到群的子群的左陪集是如何均匀地将群分成相同形式的几个子群的。子群是 ![[公式]](https://www.zhihu.com/equation?tex=W+%5Cleq+S+%3D+%5C%7Brot_0%2C+rot_%7B120%7D%2C+rot_%7B240%7D%5C%7D) ，我们可以将群 ![[公式]](https://www.zhihu.com/equation?tex=S) 分成两份都是 ![[公式]](https://www.zhihu.com/equation?tex=W) 形式的子群，而如果子群是 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7Brot_0%2C+ref_a%5C%7D) ,我们可以将群 ![[公式]](https://www.zhihu.com/equation?tex=S) 分成三份都是 ![[公式]](https://www.zhihu.com/equation?tex=W) 形式的子群。

这直接引出了商群的概念。回想一下前面给出的定义：

对于群 ![[公式]](https://www.zhihu.com/equation?tex=G) 和 ![[公式]](https://www.zhihu.com/equation?tex=G) 的正规子群 ![[公式]](https://www.zhihu.com/equation?tex=N) （记作 ![[公式]](https://www.zhihu.com/equation?tex=N+%5Cunlhd+G) ）， ![[公式]](https://www.zhihu.com/equation?tex=G) 在 ![[公式]](https://www.zhihu.com/equation?tex=N) 上的商群，写作 ![[公式]](https://www.zhihu.com/equation?tex=G%2FN) ，读作“ ![[公式]](https://www.zhihu.com/equation?tex=G) 以 ![[公式]](https://www.zhihu.com/equation?tex=N) 为模”，是 ![[公式]](https://www.zhihu.com/equation?tex=N) 在 ![[公式]](https://www.zhihu.com/equation?tex=G) 中所有陪集的集合。

一个正规子群是左右陪集相同的子群。因此，我们可以发现，子群 ![[公式]](https://www.zhihu.com/equation?tex=W+%5Cunlhd+S+%3D+%5C%7Brot_0%2C+rot_%7B120%7D%2C+rot_%7B240%7D%5C%7D) 是正规子群。我们可以用它来构造商群， ![[公式]](https://www.zhihu.com/equation?tex=S%2FW) 。

现在我们知道了什么是陪集，那么找到 ![[公式]](https://www.zhihu.com/equation?tex=S%2FW) 就变得很简单了，它只是关于 ![[公式]](https://www.zhihu.com/equation?tex=W) 的陪集的集合，而我们也将计算出来，陪集就是： ![[公式]](https://www.zhihu.com/equation?tex=S%2FW+%3D+%5C%7B%5C%7Brot_0%2C+rot_%7B120%7D%2C+rot_%7B240%7D%5C%7D%2C+%5C%7Bref_a%2C+ref_b%2C+ref_c%5C%7D%5C%7D) （我们将子群本身包含在集合中，因为子群的陪集严格地说包括了自己）。

这有两个非常有趣的原因， ![[公式]](https://www.zhihu.com/equation?tex=S%2FW) 的结果是含有两个元素的集合（元素本身是集合），所以某种意义上，我们取了一个六个元素的原始集合（整个群），并将它分成了含有三个元素的集合，每个元素是含有两个元素的集合。看起来很熟悉吗？其实这很像简单的算术 ![[公式]](https://www.zhihu.com/equation?tex=6%2F3%3D2) 。这是因为，实数的除法就是用陪集和商群定义的。第二个很有趣的原因是，商群中的两个元素是三角形的两种基本操作，即旋转操作和翻转操作。

我还想指出，我们得到的商群 ![[公式]](https://www.zhihu.com/equation?tex=S%2FW) 实际上是一个群，也就是说，它满足了所有的群公理，在这个例子中，它与模 ![[公式]](https://www.zhihu.com/equation?tex=2) 的整数（ ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+Z_2) ）同构。

所以直观地说，当你需要一些商群 ![[公式]](https://www.zhihu.com/equation?tex=A%2FB) ，其中 ![[公式]](https://www.zhihu.com/equation?tex=B%5C+%5Cunlhd+A) （ ![[公式]](https://www.zhihu.com/equation?tex=B) 是 ![[公式]](https://www.zhihu.com/equation?tex=A) 的子群），只需要问自己，“我怎么把 ![[公式]](https://www.zhihu.com/equation?tex=A) 分成像 ![[公式]](https://www.zhihu.com/equation?tex=B) 一样的小块呢？”并且分块可以重复。在这种情况下，我们的分块是不重叠的，即，商群中的每一个陪集都没有相同的元素，但情况并非总是如此。考虑由一个生成元生成的循环群 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+Z_4) ：



![img](https://pic4.zhimg.com/80/v2-2487098ef9330ca6b2decad51f4b5933_1440w.jpg)



我们可以把这个群分成 ![[公式]](https://www.zhihu.com/equation?tex=2) 个部分，但实际上有两种方法。我们可以生成一个子群 ![[公式]](https://www.zhihu.com/equation?tex=N+%5Cleq+%5Cmathbb+Z_4+%3D+%5C%7B0%2C2%5C%7D) ，这将空间分成两个部分（有两个左陪集，因此我们的商群大小为 ![[公式]](https://www.zhihu.com/equation?tex=2) ）。我们在下面描述了这一点，其中每个“部分”是在Cayley图中“彼此相对”的一对元素。

![img](https://pic3.zhimg.com/80/v2-37459a40a885fc01b5eacbe3140b8232_1440w.jpg)



![[公式]](https://www.zhihu.com/equation?tex=N+%3D+%5C%7B0%2C2%5C%7D+%5C%5C+N+%5Cleq+%5Cmathbb+Z_4+%5C%5C+%5Cmathbb+Z_4%5C+%2F%5C+N+%3D+%5C%7B%5C%7B0%2C2%5C%7D%2C%5C%7B1%2C3%5C%7D%5C%7D)



但我们也可以选择子群 ![[公式]](https://www.zhihu.com/equation?tex=N+%5Cleq+%5Cmathbb+Z_4+%3D+%5C%7B0%2C1%5C%7D) ，其中每一对元素都是相邻的。在这种情况下，我们可以将群分成 ![[公式]](https://www.zhihu.com/equation?tex=4) 个部分（即左陪集或商群的集合有 ![[公式]](https://www.zhihu.com/equation?tex=4) 个元素）。



![img](https://pic4.zhimg.com/80/v2-0058dfc3b32b7426e7380b74a871dfb3_1440w.jpg)



![[公式]](https://www.zhihu.com/equation?tex=N+%3D+%5C%7B0%2C1%5C%7D+%5C%5C+N+%5Cleq+%5Cmathbb+Z_4+%5C%5C+%5Cmathbb+Z_4%5C+%2F%5C+N+%3D+%5C%7B%5C%7B0%2C1%5C%7D%2C%5C%7B1%2C2%5C%7D%2C%5C%7B2%2C3%5C%7D%2C%5C%7B3%2C0%5C%7D%5C%7D)



最后我想说的是代数闭群和非闭群的概念。基本上，一个封闭的群是这个群中任何一个方程的解都包含在这个群里。例如，如果这是一个包含 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7B0%2C1%2C2%5C%7D) 的循环群 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+Z_2) ，那么等式 ![[公式]](https://www.zhihu.com/equation?tex=x%5E2+%3D+1) 的结果是 ![[公式]](https://www.zhihu.com/equation?tex=1) ，这在群 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7B0%2C1%2C2%5C%7D) 中。然而，如果我们可以找到一个方程，它的解不在 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+Z_2) 中，但在实数 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+R) 中，那么这个群非闭。事实上，这很容易，等式 ![[公式]](https://www.zhihu.com/equation?tex=x%2F3%3D1) 的结果是 ![[公式]](https://www.zhihu.com/equation?tex=3) ，而 ![[公式]](https://www.zhihu.com/equation?tex=3) 不在 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+Z_2) 中。

(参考：< [Algebraically closed group](https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/Algebraically_closed_group) >)



## 预告

实际上我们已经涵盖了大部分基本的数学知识，以便我们开始使用简单复形来计算拓扑特征，这就是我们下次要做的。



## 参考文献（网站）

\1. [Applying Topology to Data, Part 1: A Brief Introduction to Abstract Simplicial and Čech Complexes.](https://link.zhihu.com/?target=http%3A//dyinglovegrape.com/math/topology_data_1.php)

\2. [http://www.math.uiuc.edu/~r-ash/Algebra/Chapter4.pdf](https://link.zhihu.com/?target=http%3A//www.math.uiuc.edu/~r-ash/Algebra/Chapter4.pdf)

\3. [Group (mathematics)](https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/Group_(mathematics))

\4. [Homology Theory — A Primer](https://link.zhihu.com/?target=https%3A//jeremykun.com/2013/04/03/homology-theory-a-primer/)

\5. [http://suess.sdf-eu.org/website/lang/de/algtop/notes4.pdf](https://link.zhihu.com/?target=http%3A//suess.sdf-eu.org/website/lang/de/algtop/notes4.pdf)

\6. [Evan Chen &bullet; Napkin](https://link.zhihu.com/?target=http%3A//www.mit.edu/~evanchen/napkin.html)



## 参考文献（学术刊物）

\1. Basher, M. (2012). On the Folding of Finite Topological Space. International Mathematical Forum, 7(15), 745–752. Retrieved from [http://www.m-hikari.com/imf/imf-2012/13-16-2012/basherIMF13-16-2012.pdf](https://link.zhihu.com/?target=http%3A//www.m-hikari.com/imf/imf-2012/13-16-2012/basherIMF13-16-2012.pdf)

\2. Day, M. (2012). Notes on Cayley Graphs for Math 5123 Cayley graphs, 1–6.

\3. Doktorova, M. (2012). CONSTRUCTING SIMPLICIAL COMPLEXES OVER by, (June).

\4. Edelsbrunner, H. (2006). IV.1 Homology. Computational Topology, 81–87. Retrieved from [Computational Topology](https://link.zhihu.com/?target=http%3A//www.cs.duke.edu/courses/fall06/cps296.1/)

\5. Erickson, J. (1908). Homology. Computational Topology, 1–11.

\6. Evan Chen. (2016). An Infinitely Large Napkin.

\7. Grigor’yan, A., Muranov, Y. V., & Yau, S. T. (2014). Graphs associated with simplicial complexes. Homology, Homotopy and Applications, 16(1), 295–311. [HHA 16 (2014) No. 1 Article 16](https://link.zhihu.com/?target=http%3A//doi.org/10.4310/HHA.2014.v16.n1.a16)

\8. Kaczynski, T., Mischaikow, K., & Mrozek, M. (2003). Computing homology. Homology, Homotopy and Applications, 5(2), 233–256. [HHA 5 (2003) No. 2 Article 8](https://link.zhihu.com/?target=http%3A//doi.org/10.4310/HHA.2003.v5.n2.a8)

\9. Kerber, M. (2016). Persistent Homology – State of the art and challenges 1 Motivation for multi-scale topology. Internat. Math. Nachrichten Nr, 231(231), 15–33.

\10. Khoury, M. (n.d.). Lecture 6 : Introduction to Simplicial Homology Topics in Computational Topology : An Algorithmic View, 1–6.

\11. Kraft, R. (2016). Illustrations of Data Analysis Using the Mapper Algorithm and Persistent Homology.

\12. Lakshmivarahan, S., & Sivakumar, L. (2016). Cayley Graphs, (1), 1–9.

\13. Liu, X., Xie, Z., & Yi, D. (2012). A fast algorithm for constructing topological structure in large data. Homology, Homotopy and Applications, 14(1), 221–238. [HHA 14 (2012) No. 1 Article 11](https://link.zhihu.com/?target=http%3A//doi.org/10.4310/HHA.2012.v14.n1.a11)

\14. Naik, V. (2006). Group theory : a first journey, 1–21.

\15. Otter, N., Porter, M. A., Tillmann, U., Grindrod, P., & Harrington, H. A. (2015). A roadmap for the computation of persistent homology. Preprint ArXiv, (June), 17. Retrieved from [[1506.08903\] A roadmap for the computation of persistent homology](https://link.zhihu.com/?target=http%3A//arxiv.org/abs/1506.08903)

\16. Semester, A. (2017). § 4 . Simplicial Complexes and Simplicial Homology, 1–13.

\17. Singh, G. (2007). Algorithms for Topological Analysis of Data, (November).

\18. Zomorodian, A. (2009). Computational Topology Notes. Advances in Discrete and Computational Geometry, 2, 109–143. Retrieved from [CiteSeerX - Computational Topology](https://link.zhihu.com/?target=http%3A//citeseerx.ist.psu.edu/viewdoc/summary%3Fdoi%3D10.1.1.50.7483)

\19. Zomorodian, A. (2010). Fast construction of the Vietoris-Rips complex. Computers and Graphics (Pergamon), 34(3), 263–271. [Redirecting](https://link.zhihu.com/?target=http%3A//doi.org/10.1016/j.cag.2010.03.007)

\20. Symmetry and Group Theory 1. (2016), 1–18. [Redirecting](https://link.zhihu.com/?target=http%3A//doi.org/10.1016/B978-0-444-53786-7.00026-5)



















