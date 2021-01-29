---
title: Persistent-Homology-4
tag: TDA-PH
categories: TDA-PH
---

[![yPnSHO.jpg](https://s3.ax1x.com/2021/01/29/yPnSHO.jpg)](https://imgchr.com/i/yPnSHO)

------

> https://zhuanlan.zhihu.com/p/41264363

------

## **来自线性代数的拯救**

您可能已经注意到，对于比我们目前考虑的简单示例要大得多的东西，手工计算同调群和连通数是非常乏味和不切实际的。幸运的是，有更好的方法。特别是，我们可以用向量和矩阵的语言来进行同调群的大部分计算和表达，而计算机在处理向量和矩阵时是非常高效的。



现在，我们已经讲过向量是什么(向量空间中的元素)，但是什么是矩阵呢？你可能会想到一个矩阵有一个二维的数字网格，你知道你可以将矩阵乘以其他矩阵和向量。对于矩阵来说，一个数字网格当然是一个方便的符号，但这不是矩阵的本质。

## **什么是矩阵？**

至此，您应该对函数或映射的概念非常了解了。这两种方法都是将一种数学结构转换成另一种结构(或者至少将结构中的一个元素映射到同一结构中的不同元素)。特别是，我们花了大量的时间使用边界映射，将一个高维链群映射到一个更低维度的链群，以某种方式保留了原始组的结构(这是一种同态)。

就像我们可以有两个群之间的映射，也可以有两个向量空间之间的映射。我们称在向量空间之间的(线性)映射为矩阵。矩阵基本上是在向量空间(或单个向量元素)上应用线性变换产生新的向量空间。线性变换意味着我们只能通过常数和常数向量的加法来变换向量空间。

**定义（线性变换）**：

> 线性变换 ![[公式]](https://www.zhihu.com/equation?tex=M%5C+%3A%5C+V_1+%5Crightarrow+V_2) 是从 ![[公式]](https://www.zhihu.com/equation?tex=V_1) 到 ![[公式]](https://www.zhihu.com/equation?tex=V_2) 的映射 ![[公式]](https://www.zhihu.com/equation?tex=M) ，例如 ![[公式]](https://www.zhihu.com/equation?tex=M%28V_1+%2B+V_2%29+%3D+M%28V_1%29+%2B+M%28V_2%29) 和 ![[公式]](https://www.zhihu.com/equation?tex=M%28aV_1%29+%3D+aM%28V_1%29) ，其中 ![[公式]](https://www.zhihu.com/equation?tex=a) 是标量。



假如我们想将实数体 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+R%5E3) 映射到平面 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+R%5E2) 。

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+V_1+%26%3D+span%5C%7B%281%2C0%2C0%29%2C%280%2C1%2C0%29%2C%280%2C0%2C1%29%5C%7D+%5C%5C+V_2+%26%3D+span%5C%7B%281%2C0%29%2C%280%2C1%29%5C%7D+%5Cend%7Baligned%7D)

现在，如果想从 ![[公式]](https://www.zhihu.com/equation?tex=V_1) 映射到 ![[公式]](https://www.zhihu.com/equation?tex=V_2) ，就是说，我想将 ![[公式]](https://www.zhihu.com/equation?tex=V_1) 的每一个点对应到 ![[公式]](https://www.zhihu.com/equation?tex=V_2) 的点。我想做这样的事情有很多原因。例如，如果我要做一个图形应用程序，我想提供一个选项来旋转所绘制的图像，而这仅仅是一个在像素上应用线性变换的问题。



所以我们称映射 ![[公式]](https://www.zhihu.com/equation?tex=M%5C+%3A%5C+V_1+%5Crightarrow+V_2) 叫矩阵。注意到 ![[公式]](https://www.zhihu.com/equation?tex=V_1) 有三个元素，而 ![[公式]](https://www.zhihu.com/equation?tex=V_2) 有两个元素。为了从一个空间映射到另一个空间，我们只需要将一个基集映射到另一个基集。记住，因为这是一个线性映射，我们所能做的就是把它乘以一个标量，或者加上另一个向量，我们不能做奇异的变化，如平方项或者取对数。



我们称三个基础要素为 ![[公式]](https://www.zhihu.com/equation?tex=V_1%3A+B_1%2C+B_2%2C+B_3) 。所以， ![[公式]](https://www.zhihu.com/equation?tex=V_1+%3D+%5Clangle%7BB_1%2C+B_2%2C+B_3%7D%5Crangle) 。同理，我们称两个基本要素为 ![[公式]](https://www.zhihu.com/equation?tex=V_2%3A+%5Cbeta_1%2C+%5Cbeta_2) 。所以， ![[公式]](https://www.zhihu.com/equation?tex=V_2+%3D+%5Clangle%7B%5Cbeta_1%2C+%5Cbeta_2%7D%5Crangle) 。（尖括号代表张成，即这些元素的所有线性组合集合）。我们可以利用每个向量空间都可以由它们的基来定义的事实来设置方程，使得 ![[公式]](https://www.zhihu.com/equation?tex=V_1) 中的向量可以映射到 ![[公式]](https://www.zhihu.com/equation?tex=V_2) 。



**新符号（向量）**

> 为了防止标量符号和矢量的符号之间的混淆，我将在每个向量上加一个小箭头 ![[公式]](https://www.zhihu.com/equation?tex=%5Cvec%7Bv%7D) 表示它是一个向量，而不是一个标量。记住，标量只是来自定义向量空间的基础域 ![[公式]](https://www.zhihu.com/equation?tex=F) 的一个元素。



我们可以用下列方法定义映射 ![[公式]](https://www.zhihu.com/equation?tex=M%28V_1%29+%3D+V_2) ：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+M%28B_1%29+%26%3D+a%5Cvec+%5Cbeta_1+%2B+b%5Cvec+%5Cbeta_2+%5Cmid+a%2Cb+%5Cin+%5Cmathbb+R+%5C%5C+M%28B_2%29+%26%3D+c%5Cvec+%5Cbeta_1+%2B+d%5Cvec+%5Cbeta_2+%5Cmid+c%2Cd+%5Cin+%5Cmathbb+R+%5C%5C+M%28B_3%29+%26%3D+e%5Cvec+%5Cbeta_1+%2B+f%5Cvec+%5Cbeta_2+%5Cmid+e%2Cf+%5Cin+%5Cmathbb+R+%5C%5C+%5Cend%7Baligned%7D+)

也就是说， ![[公式]](https://www.zhihu.com/equation?tex=V_1) 的每个基的映射是由 ![[公式]](https://www.zhihu.com/equation?tex=V_2) 的基线性组合组成的。这需要我们定义总共 6 个新数据： ![[公式]](https://www.zhihu.com/equation?tex=a%2Cb%2Cc%2Cd%2Ce%2Cf+%5Cin+%5Cmathbb+R) 被用作映射。我们只需要注意 ![[公式]](https://www.zhihu.com/equation?tex=a%2Cb) 映射到 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbeta_1) 和 ![[公式]](https://www.zhihu.com/equation?tex=d%2Ce%2Cf) 映射到 ![[公式]](https://www.zhihu.com/equation?tex=%5Cbeta_2) 。怎样才能最方便的跟踪记录所有这些工作呢？是矩阵。

![[公式]](https://www.zhihu.com/equation?tex=M+%3D+%5Cbegin%7Bpmatrix%7D+a+%26+c+%26+e+%5C%5C+b+%26+d+%26+f+%5C%5C+%5Cend%7Bpmatrix%7D)

这是一种非常方便的表达映射 ![[公式]](https://www.zhihu.com/equation?tex=M%5C+%3A%5C+V_1+%5Crightarrow+V_2) 的方法。注意到矩阵的每一列都对应每个 ![[公式]](https://www.zhihu.com/equation?tex=M%28B_n%29) 的“映射方程”的系数。同时也要注意到这个矩阵的维数， ![[公式]](https://www.zhihu.com/equation?tex=2%5Ctimes3) 对应我们要映射的两个向量空间的维数。也就是说，任何映射 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+R%5En+%5Crightarrow+%5Cmathbb+R%5Em) 都能被表示为一个 ![[公式]](https://www.zhihu.com/equation?tex=m%5Ctimes+n) 矩阵。重要的是要记住，由于线性映射(以及因此矩阵)依赖于一个基的系数，那么如果一个人使用不同的基底，那么矩阵元素就会改变。



知道这一点，我们可以很容易地看出向量矩阵乘法是如何工作的，以及为什么矩阵和向量的维数必须一致。也就是说， ![[公式]](https://www.zhihu.com/equation?tex=n%5Ctimes+m) 的向量/矩阵乘以 ![[公式]](https://www.zhihu.com/equation?tex=j%5Ctimes+k) 的向量/矩阵要得到 ![[公式]](https://www.zhihu.com/equation?tex=n%5Ctimes+k) 的向量/矩阵，前提是使等式有效，必须 ![[公式]](https://www.zhihu.com/equation?tex=m%3Dj) 。



这就是我们拿矩阵映射 ![[公式]](https://www.zhihu.com/equation?tex=M) 乘以向量 ![[公式]](https://www.zhihu.com/equation?tex=V_1) ，得到向量 ![[公式]](https://www.zhihu.com/equation?tex=V_2) 的方法：



![[公式]](https://www.zhihu.com/equation?tex=M%28%5Cvec+v%5C+%5Cin%5C+V_1%29+%3D+%5Cunderbrace%7B+%5Cbegin%7Bbmatrix%7D+a+%26+c+%26+e+%5C%5C+b+%26+d+%26+f+%5C%5C+%5Cend%7Bbmatrix%7D%7D_%7BM%3AV_1%5Crightarrow+V_2%7D+%5Cunderbrace%7B+%5Cbegin%7Bpmatrix%7D+x+%5C%5C+y+%5C%5C+z+%5C%5C+%5Cend%7Bpmatrix%7D%7D_%7B%5Cvec+v%5C+%5Cin%5C+V_1%7D+%3D+%5Cunderbrace%7B+%5Cbegin%7Bbmatrix%7D+a+%2A+x+%26+c+%2A+y+%26+e+%2A+z+%5C%5C+b+%2A+x+%26+d+%2A+y+%26+f+%2A+z+%5C%5C+%5Cend%7Bbmatrix%7D%7D_%7BM%3AV_1%5Crightarrow+V_2%7D+%3D+%5Cbegin%7Bpmatrix%7D+a+%2A+x+%2B+c+%2A+y+%2B+e+%2A+z+%5C%5C+b+%2A+x+%2B+d+%2A+y+%2B+f+%2A+z+%5C%5C+%5Cend%7Bpmatrix%7D+%5Cin+V_2)



现在我们知道一个矩阵是两个向量空间之间的线性映射。但是如果把两个矩阵相乘会怎样呢？这只是映射的组成部分。例如，我们有三个向量空间 ![[公式]](https://www.zhihu.com/equation?tex=T%2C+U%2C+V) 和两个线性映射 ![[公式]](https://www.zhihu.com/equation?tex=m_1%2Cm_2) ：

![[公式]](https://www.zhihu.com/equation?tex=%5C%5CT+%5Cstackrel%7Bm_1%7D%7B%5Crightarrow%7D+U+%5Cstackrel%7Bm_2%7D%7B%5Crightarrow%7D+V)

为了从 ![[公式]](https://www.zhihu.com/equation?tex=T) 到 ![[公式]](https://www.zhihu.com/equation?tex=V) ，我们需要运用两次映射 ![[公式]](https://www.zhihu.com/equation?tex=m_2%28m_1%28T%29%29+%3D+V) 。因此，把两个矩阵相乘给了我们映射的组合 ![[公式]](https://www.zhihu.com/equation?tex=m_2+%5Ccirc+m_1) 。单位矩阵，即在除了对角线是 ![[公式]](https://www.zhihu.com/equation?tex=1) ，所有其他地方是 ![[公式]](https://www.zhihu.com/equation?tex=0) 的形式，是一个恒等映射（即不会改变输入值），例如：

![[公式]](https://www.zhihu.com/equation?tex=m%3D+%5Cbegin%7Bbmatrix%7D+%5Cddots+%26+0+%26+0+%26+0+%26+%E2%8B%B0+%5C%5C+0+%26+1+%26+0+%26+0+%26+0+%5C%5C+0+%26+0+%26+1+%26+0+%26+0+%5C%5C+0+%26+0+%26+0+%26+1+%26+0+%5C%5C+%E2%8B%B0+%26+0+%26+0+%26+0+%26+%5Cddots+%5C%5C+%5Cend%7Bbmatrix%7D+%5C%5C)

## ![[公式]](https://www.zhihu.com/equation?tex=%5C%5C+m+%5Cvec+v+%3D+%5Cvec+v%2C+%5Cforall+%5Cvec+v+%5Cin+V+)

## **（再次）回到单纯同调**

我们已经过了一遍以上矩阵的必须内容作为铺垫，那我们现在能将边界映射 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpartial%28C_n%29) 表达成矩阵的形式，这样我们才能使用线性代数的工具。这很容易理解，因为我们已经知道当我们允许标量相乘时，链群 ![[公式]](https://www.zhihu.com/equation?tex=C_n) 就能看作向量空间，那么在链向量空间之间的线性映射就是边界图，我们可以表示为一个矩阵。



我们表示一个 ![[公式]](https://www.zhihu.com/equation?tex=n) 维边界映射，即 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpartial%28C_n%29) ，为一个 ![[公式]](https://www.zhihu.com/equation?tex=k) 列， ![[公式]](https://www.zhihu.com/equation?tex=l) 行的矩阵，其中 ![[公式]](https://www.zhihu.com/equation?tex=n) 是链群的维度， ![[公式]](https://www.zhihu.com/equation?tex=k) 是 ![[公式]](https://www.zhihu.com/equation?tex=C_n) 中单形的数量，而 ![[公式]](https://www.zhihu.com/equation?tex=l) 是 ![[公式]](https://www.zhihu.com/equation?tex=C_%7Bn-1%7D) 中单形的数量。因此每一列代表 ![[公式]](https://www.zhihu.com/equation?tex=C_n) 的单形，每一行代表 ![[公式]](https://www.zhihu.com/equation?tex=C_%7Bn-1%7D) 的单形。如果这个列的单形映射到这一行的单形，我们就让矩阵的这一格为 ![[公式]](https://www.zhihu.com/equation?tex=1) 。例如，如果域是 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+Z) ，则 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpartial%28%5Ba%2Cb%5D%29+%3D+a+-+b) ，那我们 ![[公式]](https://www.zhihu.com/equation?tex=a) 列和 ![[公式]](https://www.zhihu.com/equation?tex=b) 列等于 ![[公式]](https://www.zhihu.com/equation?tex=1) ，因为 ![[公式]](https://www.zhihu.com/equation?tex=1) 维单形 ![[公式]](https://www.zhihu.com/equation?tex=%5Ba%2Cb%5D) 映射到两个 ![[公式]](https://www.zhihu.com/equation?tex=0) 维单形中。



让我们用矩阵和向量来计算前面的单纯复形的同调群。我们将倒回去用 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+Z_2) 作为我们的域（所以单形的方向可以忽略），因为这样做的计算效率更高。

![[公式]](https://www.zhihu.com/equation?tex=%5C%5CS+%3D+%5Ctext%7B+%7B%5Ba%5D%2C+%5Bb%5D%2C+%5Bc%5D%2C+%5Bd%5D%2C+%5Ba%2C+b%5D%2C+%5Bb%2C+c%5D%2C+%5Bc%2C+a%5D%2C+%5Bc%2C+d%5D%2C+%5Bd%2C+b%5D%2C+%5Ba%2C+b%2C+c%5D%7D+%7D)



![img](https://pic4.zhimg.com/80/v2-b89c282b915bcfdc86af4fa547097c33_1440w.jpg)



因为我们用的是（非常小）有限域 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+Z) ， 然后我们可以列出链(群)向量空间中的所有向量。我们有 ![[公式]](https://www.zhihu.com/equation?tex=3) 个链群，即 ![[公式]](https://www.zhihu.com/equation?tex=0) 维单形（顶点）， ![[公式]](https://www.zhihu.com/equation?tex=1) 维单形（边界）， ![[公式]](https://www.zhihu.com/equation?tex=2) 维单形（三角形）的群。



在我们的例子中，我们只有一个 ![[公式]](https://www.zhihu.com/equation?tex=2) 维单形： ![[公式]](https://www.zhihu.com/equation?tex=%5Ba%2Cb%2Cc%5D) ，因此它在域 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+Z_2) 上生成的群是 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7B0%2C+%5Ba%2Cb%2Cc%5D%5C%7D) ，这与 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+Z_2) 同构。一般来说，由单纯复形中p维单形的数量 ![[公式]](https://www.zhihu.com/equation?tex=n) 组成的群与 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+Z%5En_2) 同构。为了让电脑理解，我们可以用它们的系数 ![[公式]](https://www.zhihu.com/equation?tex=0) 或 ![[公式]](https://www.zhihu.com/equation?tex=1) 来对群元素进行编码。所以，例如，由 ![[公式]](https://www.zhihu.com/equation?tex=%5Ba%2Cb%2Cc%5D) 生成的群只需要用 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7B0%2C1%5C%7D) 就能表示。或者 ![[公式]](https://www.zhihu.com/equation?tex=0) 维单形 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7Ba%2C+b%2C+c%2C+d%5C%7D) 生成的群能用 ![[公式]](https://www.zhihu.com/equation?tex=4) 维向量表示，如，如果群的元素是 ![[公式]](https://www.zhihu.com/equation?tex=a%2Bb%2Bc) ，那么我们用 ![[公式]](https://www.zhihu.com/equation?tex=%281%2C1%2C1%2C0%29) 编码，其中每个位置分别表示 ![[公式]](https://www.zhihu.com/equation?tex=%28a%2Cb%2Cc%2Cd%29) 存在与否。



这里所有的链群都表示为带系数的向量（我没有把 ![[公式]](https://www.zhihu.com/equation?tex=C_1) 所有元素都列出来，因为这太多了[32]）：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%7D+C_0+%26%3D+%5Cleft%5C%7B+%5Cbegin%7Barray%7D%7Bll%7D+%280%2C0%2C0%2C0%29+%26+%281%2C0%2C0%2C0%29+%26+%280%2C1%2C1%2C0%29+%26+%280%2C1%2C0%2C1%29+%5C%5C+%280%2C1%2C0%2C0%29+%26+%280%2C0%2C1%2C0%29+%26+%280%2C0%2C1%2C1%29+%26+%280%2C1%2C1%2C1%29+%5C%5C+%280%2C0%2C0%2C1%29+%26+%281%2C1%2C0%2C0%29+%26+%281%2C0%2C0%2C1%29+%26+%281%2C0%2C1%2C1%29+%5C%5C+%281%2C1%2C1%2C0%29+%26+%281%2C1%2C1%2C1%29+%26+%281%2C0%2C1%2C0%29+%26+%281%2C1%2C0%2C1%29+%5C%5C+%5Cend%7Barray%7D+%5Cright.+%26+%5Ccong+%5Cmathbb+Z%5E4_2+%5C%5C+C_1+%26%3D+%5Cleft%5C%7B+%5Cbegin%7Barray%7D%7Bll%7D+%280%2C0%2C0%2C0%2C0%29+%26+%281%2C0%2C0%2C0%2C0%29+%26+%280%2C1%2C1%2C0%2C0%29+%26+%280%2C1%2C0%2C1%2C0%29+%5C%5C+%280%2C1%2C0%2C0%2C0%29+%26+%280%2C0%2C1%2C0%2C0%29+%26+%280%2C0%2C1%2C1%2C0%29+%26+%280%2C1%2C1%2C1%2C0%29+%5C%5C+%5Cdots+%5Cend%7Barray%7D+%5Cright.+%26+%5Ccong+%5Cmathbb+Z%5E5_2+%5C%5C+C_2+%26%3D+%5Cleft%5C%7B+%5Cbegin%7Barray%7D%7Bll%7D+0+%26+1+%5Cend%7Barray%7D+%5Cright.+%26+%5Ccong+%5Cmathbb+Z_2+%5Cend%7Balign%7D)



为了把 ![[公式]](https://www.zhihu.com/equation?tex=p) 维单形的群的边界映射(即线性映射)表示为一个矩阵，我们用列表示群中的每一个 ![[公式]](https://www.zhihu.com/equation?tex=p) 维单形，并用行表示每个 ![[公式]](https://www.zhihu.com/equation?tex=%28p-1%29) 维单形。我们让矩阵的每个位置等于 ![[公式]](https://www.zhihu.com/equation?tex=1) ，如果 ![[公式]](https://www.zhihu.com/equation?tex=%28p-1%29) 维单形行是 ![[公式]](https://www.zhihu.com/equation?tex=p) 维单形列的面。



我们用有序对 ![[公式]](https://www.zhihu.com/equation?tex=%28i%2Cj%29) 分别指代行和列。那么元素 ![[公式]](https://www.zhihu.com/equation?tex=a_%7B2%2C3%7D) 代表第 ![[公式]](https://www.zhihu.com/equation?tex=2) 行（从上开始）第 ![[公式]](https://www.zhihu.com/equation?tex=3) 列（从左开始）的元素。



因此，一般的边界矩阵（每列是 ![[公式]](https://www.zhihu.com/equation?tex=p) 维单形，每行是 ![[公式]](https://www.zhihu.com/equation?tex=%28p-1%29) 维单形）是这样的：



![[公式]](https://www.zhihu.com/equation?tex=%5C%5C%5Cbegin%7Balign%7D+%5Cpartial_p+%26%3D+%5Cbegin%7Bpmatrix%7D+a_%7B1%2C1%7D+%26+a_%7B1%2C2%7D+%26+a_%7B1%2C3%7D+%26+%5Ccdots+%26+a_%7B1%2Cj%7D+%5C%5C+a_%7B2%2C1%7D+%26+a_%7B2%2C2%7D+%26+a_%7B2%2C3%7D+%26+%5Ccdots+%26+a_%7B2%2Cj%7D+%5C%5C+a_%7B3%2C1%7D+%26+a_%7B3%2C2%7D+%26+a_%7B3%2C3%7D+%26+%5Ccdots+%26+a_%7B3%2Cj%7D+%5C%5C+%5Cvdots+%26+%5Cvdots+%26+%5Cvdots+%26+%5Cddots+%26+%5Cvdots+%5C%5C+a_%7Bi%2C1%7D+%26+a_%7Bi%2C2%7D+%26+a_%7Bi%2C3%7D+%26+%5Ccdots+%26+a_%7Bi%2Cj%7D+%5Cend%7Bpmatrix%7D+%5Cend%7Balign%7D)



我们将从把边界映射 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpartial%28C_2%29) 表示成矩阵开始。 ![[公式]](https://www.zhihu.com/equation?tex=C_2) 中只有一个 ![[公式]](https://www.zhihu.com/equation?tex=2) 维单形，所以矩阵只有一列，但 ![[公式]](https://www.zhihu.com/equation?tex=C_1) 有五个 ![[公式]](https://www.zhihu.com/equation?tex=1) 维单形，所以有五行。



![[公式]](https://www.zhihu.com/equation?tex=%5C%5C%5Cpartial_2+%3D+%5Cbegin%7Barray%7D%7Bc%7Clcr%7D+%5Cpartial+%26+%5Ba%2Cb%2Cc%5D+%5C%5C+%5Chline+%5Ba%2Cb%5D+%26+1+%5C%5C+%5Bb%2Cc%5D+%26+1+%5C%5C+%5Bc%2Ca%5D+%26+1+%5C%5C+%5Bc%2Cd%5D+%26+0+%5C%5C+%5Bd%2Cb%5D+%26+0+%5C%5C+%5Cend%7Barray%7D)



如果行元素是单形 ![[公式]](https://www.zhihu.com/equation?tex=%5Ba%2Cb%2Cc%5D) 的面，我们就让它等于 ![[公式]](https://www.zhihu.com/equation?tex=1) 。这个矩阵是有意义的线性映射因为如果我们把它乘以一个 ![[公式]](https://www.zhihu.com/equation?tex=C_2) 的向量元素（除了 ![[公式]](https://www.zhihu.com/equation?tex=0) 元素外，只有 ![[公式]](https://www.zhihu.com/equation?tex=1) 元素），我们得到了我们所期望的：



![[公式]](https://www.zhihu.com/equation?tex=%5C%5C%5Cbegin%7Balign%7D+%5Cbegin%7Bpmatrix%7D+1+%5C%5C+1+%5C%5C+1+%5C%5C+0+%5C%5C+0+%5C%5C+%5Cend%7Bpmatrix%7D+%2A+0+%5Cqquad+%26%3D+%5Cqquad+%5Cbegin%7Bpmatrix%7D+0+%5C%5C+0+%5C%5C+0+%5C%5C+0+%5C%5C+0+%5C%5C+%5Cend%7Bpmatrix%7D+%5C%5C+%5Cbegin%7Bpmatrix%7D+1+%5C%5C+1+%5C%5C+1+%5C%5C+0+%5C%5C+0+%5C%5C+%5Cend%7Bpmatrix%7D+%2A+1+%5Cqquad+%26%3D+%5Cqquad+%5Cbegin%7Bpmatrix%7D+1+%5C%5C+1+%5C%5C+1+%5C%5C+0+%5C%5C+0+%5C%5C+%5Cend%7Bpmatrix%7D+%5Cend%7Balign%7D)



好的，让我们继续构建边界矩阵 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpartial%28C_1%29) ：

![[公式]](https://www.zhihu.com/equation?tex=%5C%5C%5Cpartial_1+%3D+%5Cbegin%7Barray%7D%7Bc%7Clcr%7D+%5Cpartial+%26+%5Ba%2Cb%5D+%26+%5Bb%2Cc%5D+%26+%5Bc%2Ca%5D+%26+%5Bc%2Cd%5D+%26+%5Bd%2Cb%5D+%5C%5C+%5Chline+a+%26+1+%26+0+%26+1+%26+0+%26+0+%5C%5C+b+%26+1+%26+1+%26+0+%26+0+%26+1+%5C%5C+c+%26+0+%26+1+%26+1+%26+1+%26+0+%5C%5C+d+%26+0+%26+0+%26+0+%26+1+%26+1+%5C%5C+%5Cend%7Barray%7D)



这有意义吗？让我们用 **python**/**numpy** 检查一下。让我们从 ![[公式]](https://www.zhihu.com/equation?tex=1) 维链的群中取一个任意的元素，即 ![[公式]](https://www.zhihu.com/equation?tex=%5Ba%2Cb%5D%2B%5Bc%2Ca%5D%2B%5Bc%2Cd%5D) ，我们把它编码成 ![[公式]](https://www.zhihu.com/equation?tex=%281%2C0%2C1%2C1%2C0%29) 并应用边界矩阵，看看我们能得到什么。



```python3
import numpy as np
b1 = np.matrix([[1,0,1,0,0],[1,1,0,0,1],[0,1,1,1,0],[0,0,0,1,1]]) #boundary matrix C_1
el = np.matrix([1,0,1,1,0]) #random element from C_1
np.fmod(b1 * el.T, 2) # we want integers modulo 2
```



```text
matrix([[0],
        [1],
        [0],
        [1]])
```



将 ![[公式]](https://www.zhihu.com/equation?tex=%280%2C1%2C0%2C1%29) 转换成 ![[公式]](https://www.zhihu.com/equation?tex=b%2Bd) ，我们可以手工计算并比较边界：

![[公式]](https://www.zhihu.com/equation?tex=%5C%5C%5Cpartial%28%5Ba%2Cb%5D%2B%5Bc%2Ca%5D%2B%5Bc%2Cd%5D%29+%3D+a%2Bb%2Bc%2Ba%2Bc%2Bd+%3D+%5Ccancel%7Ba%7D%2Bb%2B%5Ccancel%7Bc%7D%2B%5Ccancel%7Ba%7D%2B%5Ccancel%7Bc%7D%2Bd+%3D+b%2Bd+%3D+%280%2C1%2C0%2C1%29)

这行得通！



最后，我们需要 ![[公式]](https://www.zhihu.com/equation?tex=C_0) 的边界矩阵，这很简单，因为 ![[公式]](https://www.zhihu.com/equation?tex=0) 维单形的边界总是映射到 ![[公式]](https://www.zhihu.com/equation?tex=0) ，所以：

![[公式]](https://www.zhihu.com/equation?tex=%5C%5C%5Cpartial_0+%3D+%5Cbegin%7Bpmatrix%7D+0+%26+0+%26+0+%26+0+%5C%5C+%5Cend%7Bpmatrix%7D)

是OK的，现在我们有了三个边界矩阵，我们怎么计算连通数呢？回想一下链群的子群序列： ![[公式]](https://www.zhihu.com/equation?tex=B_n+%5Cleq+Z_n+%5Cleq+C_n) ，它们分别是边界群，循环群和链群。



再重温一下连通数的定义： ![[公式]](https://www.zhihu.com/equation?tex=b_n+%3D+dim%28Z_n%5C+%2F%5C+B_n%29) 。但这是用群结构表示的集合，现在所有东西都被表示成向量和矩阵，所以我们重新定义了连通数 ![[公式]](https://www.zhihu.com/equation?tex=b_n+%3D+rank%28Z_n%29%5C+-%5C+rank%28B_n%29) 。”**rank**“ 是什么鬼东西？秩（**rank**）和维度差不多，但又不一样。如果我们把矩阵的列看成一组基向量： ![[公式]](https://www.zhihu.com/equation?tex=%5Cbeta_1%2C+%5Cbeta_2%2C+%5Cdots+%5Cbeta_k) ，那么这些列向量 ![[公式]](https://www.zhihu.com/equation?tex=%5Clangle+%5Cbeta_1%2C+%5Cbeta_2%2C+%5Cdots+%5Cbeta_k+%5Crangle) 张成的空间的维数就是矩阵的秩。结果是，你也可以使用行，它也会有相同的结果。但是，重要的是，维度是在最小的基元集合上定义的，即线性无关的基元。



矩阵的边界 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpartial_n) 包含链群和循环子群的信息，以及 ![[公式]](https://www.zhihu.com/equation?tex=B_%7Bn-1%7D) 边界子群，我们需要计算的连通数的所有信息。不幸的是，总的来说，我们构建边界矩阵的朴素方法并不是一种可以轻易得到的分群和子群信息的形式。我们需要修改边界矩阵，而不干扰它包含的映射信息，从而形成一种名为 **Smith** 标准型的新形式。基本上，矩阵的 **Smith** 标准型是沿着对角线从左上方开始全是1，而其他地方全是 ![[公式]](https://www.zhihu.com/equation?tex=0) 的形式。例如，

![[公式]](https://www.zhihu.com/equation?tex=%5C%5C%5Cbegin%7Balign%7D+%5Ctext%7BSmith+normal+form%7D+%26%3A%5C+%5Cbegin%7Bpmatrix%7D+1+%26+0+%26+0+%26+%5Ccdots+%26+0+%5C%5C+0+%26+1+%26+0+%26+%5Ccdots+%26+0+%5C%5C+0+%26+0+%26+1+%26+%5Ccdots+%26+0+%5C%5C+%5Cvdots+%26+%5Cvdots+%26+%5Cvdots+%26+%5Cddots+%26+%5Cvdots+%5C%5C+0+%26+0+%26+0+%26+%5Ccdots+%26+%3F+%5Cend%7Bpmatrix%7D+%5Cend%7Balign%7D)

注意到 ![[公式]](https://www.zhihu.com/equation?tex=1) 沿对角线不一定要一直延伸到右下方。而这才是在 **Smith** 标准型中可用的信息(红色对角线表示 ![[公式]](https://www.zhihu.com/equation?tex=1) ):

![img](https://pic2.zhimg.com/80/v2-a1473acd9bef9bd06c1d7bc7f0286201_1440w.jpg)

(来源: "COMPUTATIONAL TOPOLOGY" by Edelsbrunner and Harer, pg. 104)



那么怎样才能得到一个 **Smith** 标准型呢?我们可以根据规则进行一些操作，以下是矩阵中允许的两个操作：

\1. 你可以在矩阵中交换任意两列或任意两行。

\2. 您可以向另一个列添加一个列，或者将一行添加到另一行。



现在你只需要应用这些运算，直到你得到Smith标准型的矩阵。我要指出的是，当我们用域 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+Z_2) 时，这个过程并不容易。让我们用边界矩阵 ![[公式]](https://www.zhihu.com/equation?tex=C_1) 试一下。



![[公式]](https://www.zhihu.com/equation?tex=%5C%5C%5Cpartial_1+%3D+%5Cbegin%7Barray%7D%7Bc%7Clcr%7D+%5Cpartial+%26+%5Ba%2Cb%5D+%26+%5Bb%2Cc%5D+%26+%5Bc%2Ca%5D+%26+%5Bc%2Cd%5D+%26+%5Bd%2Cb%5D+%5C%5C+%5Chline+a+%26+1+%26+0+%26+1+%26+0+%26+0+%5C%5C+b+%26+1+%26+1+%26+0+%26+0+%26+1+%5C%5C+c+%26+0+%26+1+%26+1+%26+1+%26+0+%5C%5C+d+%26+0+%26+0+%26+0+%26+1+%26+1+%5C%5C+%5Cend%7Barray%7D)



我们已经在对角线上有 ![[公式]](https://www.zhihu.com/equation?tex=1) 了，但是我们有很多 ![[公式]](https://www.zhihu.com/equation?tex=1) 不在对角线上。



步骤：把第 ![[公式]](https://www.zhihu.com/equation?tex=3) 列加到第 ![[公式]](https://www.zhihu.com/equation?tex=5) 列上，再把第 ![[公式]](https://www.zhihu.com/equation?tex=4) 列加到第 ![[公式]](https://www.zhihu.com/equation?tex=5) 列上，然后把第 ![[公式]](https://www.zhihu.com/equation?tex=1) 列加到第 ![[公式]](https://www.zhihu.com/equation?tex=5) 列上：



![[公式]](https://www.zhihu.com/equation?tex=%5C%5C%5Cpartial_1+%3D+%5Cbegin%7Bpmatrix%7D+1+%26+0+%26+1+%26+0+%26+0+%5C%5C+1+%26+1+%26+0+%26+0+%26+0+%5C%5C+0+%26+1+%26+1+%26+1+%26+0+%5C%5C+0+%26+0+%26+0+%26+1+%26+0+%5C%5C+%5Cend%7Bpmatrix%7D)



步骤：把第 ![[公式]](https://www.zhihu.com/equation?tex=1) 列和第 ![[公式]](https://www.zhihu.com/equation?tex=2) 列加到第 ![[公式]](https://www.zhihu.com/equation?tex=3) 列上，把第1行加到第 ![[公式]](https://www.zhihu.com/equation?tex=2) 行上，把第 ![[公式]](https://www.zhihu.com/equation?tex=2) 行加到第 ![[公式]](https://www.zhihu.com/equation?tex=3) 行上，把第 ![[公式]](https://www.zhihu.com/equation?tex=3) 行加到第 ![[公式]](https://www.zhihu.com/equation?tex=4) 行上，第 ![[公式]](https://www.zhihu.com/equation?tex=3) 列和第 ![[公式]](https://www.zhihu.com/equation?tex=4) 列交换，最后矩阵如下：


![[公式]](https://www.zhihu.com/equation?tex=%5C%5C%5Ctext%7BSmith+normal+form%3A+%7D+%5Cbegin%7Bpmatrix%7D+1+%26+0+%26+0+%26+0+%26+0+%5C%5C+0+%26+1+%26+0+%26+0+%26+0+%5C%5C+0+%26+0+%26+1+%26+0+%26+0+%5C%5C+0+%26+0+%26+0+%26+0+%26+0+%5C%5C+%5Cend%7Bpmatrix%7D)



一旦我们有了 **Smith** 标准型的矩阵，我们就不再做任何运算了，当然，我们可以继续把行和列相加直到我们得到0的矩阵，但这并不是很有帮助！我随机添加了行/列，让它变成 **Smith** 标准型，但实际上有一种算法可以相对高效地完成它。



我将使用一个[现有的算法](https://link.zhihu.com/?target=https%3A//triangleinequality.wordpress.com/2014/01/23/computing-homology/)，而不是详细地实现 **Smith** 标准型的算法。



```text
def reduce_matrix(matrix):
    #Returns [reduced_matrix, rank, nullity]
    if np.size(matrix)==0:
        return [matrix,0,0]
    m=matrix.shape[0]
    n=matrix.shape[1]
    def _reduce(x):
        #We recurse through the diagonal entries.
        #We move a 1 to the diagonal entry, then
        #knock out any other 1s in the same  col/row.
        #The rank is the number of nonzero pivots,
        #so when we run out of nonzero diagonal entries, we will
        #know the rank.
        nonzero=False
        #Searching for a nonzero entry then moving it to the diagonal.
        for i in range(x,m):
            for j in range(x,n):
                if matrix[i,j]==1:
                    matrix[[x,i],:]=matrix[[i,x],:]
                    matrix[:,[x,j]]=matrix[:,[j,x]]
                    nonzero=True
                    break
            if nonzero:
                break
        #Knocking out other nonzero entries.
        if nonzero:
            for i in range(x+1,m):
                if matrix[i,x]==1:
                    matrix[i,:] = np.logical_xor(matrix[x,:], matrix[i,:])
            for i in range(x+1,n):
                if matrix[x,i]==1:
                    matrix[:,i] = np.logical_xor(matrix[:,x], matrix[:,i])
            #Proceeding to next diagonal entry.
            return _reduce(x+1)
        else:
            #Run out of nonzero entries so done.
            return x
    rank=_reduce(0)
    return [matrix, rank, n-rank]

# Source: < https://triangleinequality.wordpress.com/2014/01/23/computing-homology/ 
```



```text
reduce_matrix(b1)
#Returns the matrix in Smith normal form as well as rank(B_n-1) and rank(Z_n)
```



```text
[matrix([[1, 0, 0, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]), 3, 2]
```



正如你所见，我们得到了同样的结果，但是这个算法肯定更有效率。

因为每个边界映射给我们 ![[公式]](https://www.zhihu.com/equation?tex=Z_n) （循环）和 ![[公式]](https://www.zhihu.com/equation?tex=B_%7Bn-1%7D) （ ![[公式]](https://www.zhihu.com/equation?tex=%28n-1%29) 维链群的边界），为了计算链群 ![[公式]](https://www.zhihu.com/equation?tex=n) 的连通数，我们需要 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpartial_n) 和 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpartial_%7Bn%2B1%7D) 。记住，我们现在用 ![[公式]](https://www.zhihu.com/equation?tex=b_n+%3D+rank%28Z_n%29%5C+-%5C+rank%28B_n%29) 来计算连通数。

```text
#Initialize boundary matrices
boundaryMap0 = np.matrix([[0,0,0,0]])
boundaryMap1 = np.matrix([[1,0,1,0,0],[1,1,0,0,1],[0,1,1,1,0],[0,0,0,1,1]])
boundaryMap2 = np.matrix([[1,1,1,0,0]])

#Smith normal forms of the boundary matrices
smithBM0 = reduce_matrix(boundaryMap0)
smithBM1 = reduce_matrix(boundaryMap1)
smithBM2 = reduce_matrix(boundaryMap2)

#Calculate Betti numbers
betti0 = (smithBM0[2] - smithBM1[1])
betti1 = (smithBM1[2] - smithBM2[1])
betti2 = 0  #There is no n+1 chain group, so the Betti is 0

print(smithBM0)
print(smithBM1)
print(smithBM2)
print("Betti #0: %s \n Betti #1: %s \n Betti #2: %s" % (betti0, betti1, betti2))
```



```text
[matrix([[0, 0, 0, 0]]), 0, 4]
[matrix([[1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0]]), 3, 2]
[matrix([[1, 0, 0, 0, 0]]), 1, 4]
Betti #0: 1 
Betti #1: 1 
Betti #2: 0
```

很好，这行得通。

但我们跳过了一个重要的步骤。我们首先手动设计了边界矩阵，为了算法化整个过程，从构建单纯复形的数据到计算连通数，我们需要一个算法来用简单复形建立边界矩阵。现在让我们来解决它。



```text
#return the n-simplices in a complex
def nSimplices(n, complex):
    nchain = []
    for simplex in complex:
        if len(simplex) == (n+1):
            nchain.append(simplex)
    if (nchain == []): nchain = [0]
    return nchain

#check if simplex is a face of another simplex
def checkFace(face, simplex):
    if simplex == 0:
        return 1
    elif set(face) < set(simplex): #if face is a subset of simplex
        return 1
    else:
        return 0
#build boundary matrix for dimension n ---> (n-1) = p
def boundaryMatrix(nchain, pchain):
    bmatrix = np.zeros((len(nchain),len(pchain)))
    i = 0
    for nSimplex in nchain:
        j = 0
        for pSimplex in pchain:
            bmatrix[i, j] = checkFace(pSimplex, nSimplex)
            j += 1
        i += 1
    return bmatrix.T
```

这些是非常简单的辅助函数，用来构建边界矩阵然后使用之前描述的简化算法把它变成 **Smith** 标准型。别忘了，我们使用的简单复形的例子是这样的:

![img](https://pic1.zhimg.com/80/v2-c77bb35f0fe75585251c5d70bbd78774_1440w.jpg)



我准备用 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7B1%2C2%2C3%2C4%5C%7D) 代替 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7Ba%2Cb%2Cc%2Cd%5C%7D) ，这样 **python** 才能理解它。



```text
S = [{0}, {1}, {2}, {3}, {0, 1}, {1, 2}, {2, 0}, {2, 3}, {3, 1}, {0, 1, 2}] #this is our simplex from above

chain2 = nSimplices(1, S)
chain1 = nSimplices(0, S)
reduce_matrix(boundaryMatrix(chain2, chain1))
```



```text
[array([[ 1.,  0.,  0.,  0.,  0.],
        [ 0.,  1.,  0.,  0.,  0.],
        [ 0.,  0.,  1.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.]]), 3, 2]
```

现在让我们把所有的东西放在一起，做成一个函数，它会返回单纯复形的所有连通数。

```text
def betti(complex):
    max_dim = len(max(complex, key=len)) #get the maximum dimension of the simplicial complex, 2 in our example
    betti_array = np.zeros(max_dim) #setup array to store n-th dimensional Betti numbers
    z_n = np.zeros(max_dim) #number of cycles (from cycle group)
    b_n = np.zeros(max_dim) #b_(n-1) boundary group
    for i in range(max_dim): #loop through each dimension starting from maximum to generate boundary maps
        bm = 0 #setup n-th boundary matrix
        chain2 = nSimplices(i, complex) #n-th chain group
        if i==0: #there is no n+1 boundary matrix in this case
            bm = 0
            z_n[i] = len(chain2)
            b_n[i] = 0
        else:
            chain1 = nSimplices(i-1, complex) #(n-1)th chain group
            bm = reduce_matrix(boundaryMatrix(chain2, chain1))
            z_n[i] = bm[2]
            b_n[i] = bm[1] #b_(n-1)

    for i in range(max_dim): #Calculate betti number: Z_n - B_n
        if (i+1) < max_dim:
            betti_array[i] = z_n[i] - b_n[i+1]
        else:
            betti_array[i] = z_n[i] - 0 #if there are no higher simplices, the boundary group of this chain is 0

    return betti_array
```

好的，现在我们应该有计算任意以正确格式输入的单纯复形的连通数的集合需要的所有东西。请记住，所有这些代码都是为了学习目的而码的，所以我故意保持了它的简单性。它还不够完整。它基本上没有纠错功能，所以如果它得到了一些出乎意料的东西，它就会崩溃。



但是还是让我们的程序在各种简单复形上运行嘚瑟一下。



![[公式]](https://www.zhihu.com/equation?tex=%5C%5CH+%3D+%5Ctext%7B+%7B+%7B0%7D%2C+%7B1%7D%2C+%7B2%7D%2C+%7B3%7D%2C+%7B4%7D%2C+%7B5%7D%2C+%7B4%2C+5%7D%2C+%7B0%2C+1%7D%2C+%7B1%2C+2%7D%2C+%7B2%2C+0%7D%2C+%7B2%2C+3%7D%2C+%7B3%2C+1%7D%2C+%7B0%2C+1%2C+2%7D+%7D+%7D)



![img](https://pic1.zhimg.com/80/v2-c3ffc9d9717ef0dc0e85d5b216b53d94_1440w.jpg)



你可以看出这是我们之前一直在用的简单复形，除了现在它在右边有一个断开的边。因此，对于维度 ![[公式]](https://www.zhihu.com/equation?tex=0) ，我们应该得到连通数为 ![[公式]](https://www.zhihu.com/equation?tex=2) ，因为有 ![[公式]](https://www.zhihu.com/equation?tex=2) 个连接组件。

```text
H = [{0}, {1}, {2}, {3}, {4}, {5}, {4, 5}, {0, 1}, {1, 2}, {2, 0}, {2, 3}, {3, 1}, {0, 1, 2}]
betti(H)
```



```text
array([ 2.,  1.,  0.])
```



我们再试试另一个，这个有 ![[公式]](https://www.zhihu.com/equation?tex=2) 个周期和 ![[公式]](https://www.zhihu.com/equation?tex=2) 个连接的组件。

![[公式]](https://www.zhihu.com/equation?tex=%5C%5CY_1+%3D+%5Ctext%7B+%7B+%7B0%7D%2C+%7B1%7D%2C+%7B2%7D%2C+%7B3%7D%2C+%7B4%7D%2C+%7B5%7D%2C+%7B6%7D%2C+%7B0%2C+6%7D%2C+%7B2%2C+6%7D%2C+%7B4%2C+5%7D%2C+%7B0%2C+1%7D%2C+%7B1%2C+2%7D%2C+%7B2%2C+0%7D%2C+%7B2%2C+3%7D%2C+%7B3%2C+1%7D%2C+%7B0%2C+1%2C+2%7D+%7D+%7D)

![img](https://pic1.zhimg.com/80/v2-9346469c65242789c8c4076029b73270_1440w.jpg)

```text
Y1 = [{0}, {1}, {2}, {3}, {4}, {5}, {6}, {0, 6}, {2, 6}, {4, 5}, {0, 1}, {1, 2}, {2, 0}, {2, 3}, {3, 1}, {0, 1, 2}]
betti(Y1)
```



```text
array([ 2.,  2.,  0.])
```

再来一个。我只是添加了一个顶点：

![[公式]](https://www.zhihu.com/equation?tex=%5C%5CY_2+%3D+%5Ctext%7B+%7B+%7B0%7D%2C+%7B1%7D%2C+%7B2%7D%2C+%7B3%7D%2C+%7B4%7D%2C+%7B5%7D%2C+%7B6%7D%2C+%7B7%7D%2C+%7B0%2C+6%7D%2C+%7B2%2C+6%7D%2C+%7B4%2C+5%7D%2C+%7B0%2C+1%7D%2C+%7B1%2C+2%7D%2C+%7B2%2C+0%7D%2C+%7B2%2C+3%7D%2C+%7B3%2C+1%7D%2C+%7B0%2C+1%2C+2%7D+%7D+%7D)

![img](https://pic2.zhimg.com/80/v2-11ad2a4f2c28e2767b9f86caa8cc5fa9_1440w.jpg)



```text
Y2 = [{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {0, 6}, {2, 6}, {4, 5}, {0, 1}, {1, 2}, {2, 0}, {2, 3}, {3, 1}, {0, 1, 2}]
betti(Y2)
```



```text
array([ 3.,  2.,  0.])
```



最后一个。这是一个空心四面体：

![img](https://pic2.zhimg.com/80/v2-3fb13fda256aa38b3832178134b6e651_1440w.jpg)

```text
D = [{0}, {1}, {2}, {3}, {0,1}, {1,3}, {3,2}, {2,0}, {2,1}, {0,3}, {0,1,3}, {0,1,2}, {2,0,3}, {1,2,3}]
betti(D)
```



```text
array([ 1.,  0.,  1.]) 
```

果然如我们所愿！好的，看起来我们可以可靠地计算任意的单纯复形的连通数了。



## **预告**



前4篇文章都只是阐述了持续同调背后的数学思想和概念，但到目前为止，我们所做的只是（非持续性的）同调。还记得在第2部分中，我们编写了一个为数据构建简单复形的算法吗？回想一下，我们需要任意选择一个参数 ![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon) ，它决定了两个顶点是否足够近，以将它们连接起来。如果我们选择较小的 ![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon) ，那么我们会得到一个较为稀疏的图，如果我们选择较大的 ![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon) ，那么我们会得到一个拥有许多边的稠密的图。



问题是我们无法知道"正确"的 ![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon) 值应该是多少。根据不同的 ![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon) 值，我们将会得到不同单纯复形（因此有不同的同调群和连通数）。持续同调基本上是说：让我们从 ![[公式]](https://www.zhihu.com/equation?tex=0) 到最大值(所有顶点都是边缘连接)不断的扩大 ![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon) 值 ，并观察哪些拓扑特性持续时间最长。我们相信那些持续时间短的拓扑特征是噪音，而持续时间长的是数据真正的特征。所以下次我们将继续修改算法，在追踪计算的同调群的变化时，能不断变化 ![[公式]](https://www.zhihu.com/equation?tex=%5Cepsilon) 。



## **参考文献（网站）**



\1. [http://dyinglovegrape.com/math/topology_data_1.php](https://link.zhihu.com/?target=http%3A//dyinglovegrape.com/math/topology_data_1.php)

\2. [http://www.math.uiuc.edu/~r-ash/Algebra/Chapter4.pdf](https://link.zhihu.com/?target=http%3A//www.math.uiuc.edu/~r-ash/Algebra/Chapter4.pdf)

\3. [Group (mathematics)](https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/Group_(mathematics))

\4. [Homology Theory — A Primer](https://link.zhihu.com/?target=https%3A//jeremykun.com/2013/04/03/homology-theory-a-primer/)

\5. [http://suess.sdf-eu.org/website/lang/de/algtop/notes4.pdf](https://link.zhihu.com/?target=http%3A//suess.sdf-eu.org/website/lang/de/algtop/notes4.pdf)

\6. [http://www.mit.edu/~evanchen/napkin.html](https://link.zhihu.com/?target=http%3A//www.mit.edu/~evanchen/napkin.html)



## **参考文献（学术刊物）**



\1. Basher, M. (2012). On the Folding of Finite Topological Space. International Mathematical Forum, 7(15), 745–752. Retrieved from [http://www.m-hikari.com/imf/imf-2012/13-16-2012/basherIMF13-16-2012.pdf](https://link.zhihu.com/?target=http%3A//www.m-hikari.com/imf/imf-2012/13-16-2012/basherIMF13-16-2012.pdf)

\2. Day, M. (2012). Notes on Cayley Graphs for Math 5123 Cayley graphs, 1–6.

\3. Doktorova, M. (2012). CONSTRUCTING SIMPLICIAL COMPLEXES OVER by, (June).

\4. Edelsbrunner, H. (2006). IV.1 Homology. Computational Topology, 81–87. Retrieved from [http://www.cs.duke.edu/courses/fall06/cps296.1/](https://link.zhihu.com/?target=http%3A//www.cs.duke.edu/courses/fall06/cps296.1/)

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