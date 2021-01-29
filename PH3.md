---
title: Persistent-Homology-3
tag: TDA-PH
categories: TDA-PH
---

[![yPnSHO.jpg](https://s3.ax1x.com/2021/01/29/yPnSHO.jpg)](https://imgchr.com/i/yPnSHO)

------

> https://zhuanlan.zhihu.com/p/33376520

------

## 回到单纯同调



我们已经讨论了足够多的群论内容，能够完成我们在单纯复形上的同源群的计算。你们应该回忆一下，我们已经给出了第 ![[公式]](https://www.zhihu.com/equation?tex=n) 个同源群和第 ![[公式]](https://www.zhihu.com/equation?tex=n) 个连通数的定义。



连通数是我们的终极目标。他们很好地总结了单纯复形的拓扑性质。如果我们有一个形成单一圆形物体的单纯复形，那么 ![[公式]](https://www.zhihu.com/equation?tex=b_0) （第 ![[公式]](https://www.zhihu.com/equation?tex=0) 连通数）代表连接组件的数量（它是 ![[公式]](https://www.zhihu.com/equation?tex=1) ），而 ![[公式]](https://www.zhihu.com/equation?tex=b_1) 是 ![[公式]](https://www.zhihu.com/equation?tex=1) 维孔的数量（即圈），它也等于 ![[公式]](https://www.zhihu.com/equation?tex=1) ，但 ![[公式]](https://www.zhihu.com/equation?tex=b_n%2C+n%3E1) 是高维孔，而它等于 ![[公式]](https://www.zhihu.com/equation?tex=0) 。



我们来看看是否可以计算一个简单三角形单纯复形的连通数。



![[公式]](https://www.zhihu.com/equation?tex=%5Cmathcal+T+%3D+%5Ctext%7B+%7B%7Ba%7D%2C+%7Bb%7D%2C+%7Bc%7D%2C+%5Ba%2C+b%5D%2C+%5Bb%2C+c%5D%2C+%5Bc%2C+a%5D%7D+%7D) （在下面有描述）



![img](https://pic3.zhimg.com/80/v2-ec1e144732d902adce0c02876edc0cde_1440w.jpg)



目测 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathcal+T) 的连通数 ![[公式]](https://www.zhihu.com/equation?tex=b_0%3D1) （ ![[公式]](https://www.zhihu.com/equation?tex=1) 个连接的组件）， ![[公式]](https://www.zhihu.com/equation?tex=b_1%3D1) （ ![[公式]](https://www.zhihu.com/equation?tex=1) 个孔），我们只计算那些连通数。



让我们慢慢地完成整个步骤。首先我们要注意 ![[公式]](https://www.zhihu.com/equation?tex=n) 维链。



![[公式]](https://www.zhihu.com/equation?tex=0) 维链是 ![[公式]](https://www.zhihu.com/equation?tex=0) 维单形的集合： ![[公式]](https://www.zhihu.com/equation?tex=%5C%7B%5C%7Ba%5C%7D%2C%5C%7Bb%5C%7D%2C%5C%7Bc%5C%7D%5C%7D) 。 ![[公式]](https://www.zhihu.com/equation?tex=1) 维链是 ![[公式]](https://www.zhihu.com/equation?tex=1) 维单形的集合： ![[公式]](https://www.zhihu.com/equation?tex=%5Ba%2C+b%5D%2C+%5Bb%2C+c%5D%2C+%5Bc%2C+a%5D) 。该单纯复形没有更高维的 ![[公式]](https://www.zhihu.com/equation?tex=n) 维链了。



现在我们可以用 ![[公式]](https://www.zhihu.com/equation?tex=n) 维链定义链群。我们准备用 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+Z_2) 的系数， ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+Z_2) 是一个只有两个元素 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7B0%2C1%5C%7D) 的域，其中 ![[公式]](https://www.zhihu.com/equation?tex=1%2B1%3D0) 。



![[公式]](https://www.zhihu.com/equation?tex=0) 维链群被定义成： ![[公式]](https://www.zhihu.com/equation?tex=%5C%5C+C_0+%3D+%5C%7B%5C%7Bx%2A%28a%2C0%2C0%29%5C%7D%2C+%5C%7By%2A%280%2Cb%2C0%29%5C%7D%2C+%5C%7Bz%2A%280%2C0%2Cc%29%5C%7D+%5Cmid+x%2Cy%2Cz+%5Cin+%5Cmathbb+Z_2%5C%7D+)

这个群只定义了加法运算，但我们在 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+Z_2) 中用乘法构建了这个群。因此这个群与 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+Z_%7B2%7D%5E%7B3%7D+%3D+Z_%7B2%7D+%5Coplus+Z_%7B2%7D+%5Coplus+Z_%7B2%7D) 同构。



但我们也想把链群表示成向量空间。这意味着它成为一种结构，这个结构中的元素可以通过域中的元素放大或缩小(即乘法运算)，并加在一起，所有的结果仍然在结构中。如果我们只关注加法运算，那么我们看到的是群结构，但如果我们关注加法和乘法运算，那么我们可以把它看作向量空间。



0维链向量空间通过以下方法生成： ![[公式]](https://www.zhihu.com/equation?tex=%5C%5C+%5Cmathscr+C_0+%3D+%5C%7B%5C%7Bx%2A%28a%2C0%2C0%29%5C%7D%2C+%5C%7By%2A%280%2Cb%2C0%29%5C%7D%2C+%5C%7Bz%2A%280%2C0%2Cc%29%5C%7D+%5Cmid+x%2Cy%2Cz+%5Cin+%5Cmathbb+Z_2%5C%7D+%5C%5C)

（是的，它和上面的群是相同的集合）



向量空间是集合的一组元素，我们可以乘上 ![[公式]](https://www.zhihu.com/equation?tex=0) 或 ![[公式]](https://www.zhihu.com/equation?tex=1) ，然后把它们加起来。例如： ![[公式]](https://www.zhihu.com/equation?tex=1%2A%28a%2C0%2C0%29+%2B+1%2A%280%2C0%2Cc%29+%3D+%28a%2C0%2Cc%29) ，这个向量空间非常小（ ![[公式]](https://www.zhihu.com/equation?tex=2%5E3%3D8) 个元素），我们可以列出所有的元素。他们是：

![[公式]](https://www.zhihu.com/equation?tex=%5C%5C+%5Cmathscr%7BC_0%7D+%3D+%5Cbegin%7BBmatrix%7D+%28a%2C0%2C0%29%2C+%280%2Cb%2C0%29%2C+%280%2C0%2Cc%29%2C+%28a%2Cb%2C0%29+%5C%5C+%28a%2Cb%2Cc%29%2C+%280%2Cb%2Cc%29%2C+%28a%2C0%2Cc%29%2C+%280%2C0%2C0%29+%5Cend%7BBmatrix%7D)

你可以看到，我们可以在这个向量空间中添加任何两个元素，结果将是向量空间中的另一个元素。随便举个例子： ![[公式]](https://www.zhihu.com/equation?tex=%28a%2C0%2Cc%29+%2B+%28a%2Cb%2Cc%29+%3D+%28a%2Ba%2C0%2Bb%2Cc%2Bc%29+%3D+%280%2Cb%2C0%29) 。加法是分量方式的。我们也可以用向量乘以一个域 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+Z_2) 中的元素，但因为我们的域只有两个元素，结果比较无聊，只有 ![[公式]](https://www.zhihu.com/equation?tex=1%2A%28a%2Cb%2C0%29+%3D+%28a%2Cb%2C0%29) 和 ![[公式]](https://www.zhihu.com/equation?tex=0%2A%28a%2Cb%2C0%29+%3D+%280%2C0%2C0%29) ，但乘法运算仍然会在我们的向量空间中产生一个元素。

我们可以把这个向量空间表示为一个多项式，那么我们的0维链向量空间能等价地定义为：

![[公式]](https://www.zhihu.com/equation?tex=%5C%5C+%5Cmathscr%7BC_0%7D+%3D+%5C%7Bxa+%2B+yb+%2B+zc+%5Cmid+z%2Cy%2Cz+%5Cin+%5Cmathbb+Z_2%5C%7D)

我们可以很容易地把一个多项式 ![[公式]](https://www.zhihu.com/equation?tex=a%2Bb%2Bc) 翻译成它的有序集符号 ![[公式]](https://www.zhihu.com/equation?tex=%28a%2Cb%2Cc%29) ，或者 ![[公式]](https://www.zhihu.com/equation?tex=a%2Bb) 是 ![[公式]](https://www.zhihu.com/equation?tex=%28a%2Cb%2C0%29) 。作为多项式集合的向量空间是这样的： ![[公式]](https://www.zhihu.com/equation?tex=%5C%5C+%5Cmathscr%7BC_0%7D+%3D+%5Cbegin%7BBmatrix%7D+%5Ctext%7B+%7B0%7D%2C+%7Ba%7D%2C+%7Bb%7D%2C+%7Bc%7D%2C+%7Ba%2Bb%7D%2C+%7Bb%2Bc%7D%2C+%7Ba%2Bc%7D%2C+%7Ba%2Bb%2Bc%7D%7D+%5Cend%7BBmatrix%7D)

一般来说，使用多项式形式更方便，因为我们可以做一些熟悉的代数方程,像这样： ![[公式]](https://www.zhihu.com/equation?tex=a%2Bb%3D0+%5C%5Ca+%3D+-b+%5C%5Ca+%3D+b)

（记得 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+Z_2) 中的一个元素的逆是它自己，即 ![[公式]](https://www.zhihu.com/equation?tex=-b%3Db) ，其中“ ![[公式]](https://www.zhihu.com/equation?tex=-) ”表示负）。

> 注意： 知道我们讨论的是群体还是向量空间是非常重要的。我将用普通的 ![[公式]](https://www.zhihu.com/equation?tex=C) 代表链群而花体 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathscr+C) 代表链（向量）空间。它们具有相同的底层集，只是定义了不同的操作。如果我们讨论群形式，我们只能参考它的加法运算，而如果我们讨论向量空间形式，我们可以讨论它的乘法运算和加法运算。

我们对1维链进行相同的操作： ![[公式]](https://www.zhihu.com/equation?tex=%5Ba%2C+b%5D%2C+%5Bb%2C+c%5D%2C+%5Bc%2C+a%5D) 。我们可以用 ![[公式]](https://www.zhihu.com/equation?tex=1) 维链集定义另一个链群， ![[公式]](https://www.zhihu.com/equation?tex=C_1) 。它将与 ![[公式]](https://www.zhihu.com/equation?tex=C_0) 和 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+Z_%7B2%7D%5E%7B3%7D) 同构。 ![[公式]](https://www.zhihu.com/equation?tex=C_1+%3D+%5C%7B%5C+%28%5C+x%28%5Ba%2C+b%5D%29%2C+y%28%5Bb%2C+c%5D%29%2C+z%28%5Bc%2C+a%5D%29%5C+%29+%5Cmid+x%2Cy%2Cz+%5Cin+%5Cmathbb+Z_2%5C+%5C%7D) 我们可以用和定义 ![[公式]](https://www.zhihu.com/equation?tex=C_0) 同样的方法来使用这个链定义向量空间 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathscr+C_1) 。我将使用多项式形式。记住，链群和向量空间有相同的集合，只是向量空间有两个二进制运算而不是一个。这是向量空间中元素的完整列表： ![[公式]](https://www.zhihu.com/equation?tex=%5C%5C+%5Cmathscr%7BC_1%7D+%3D+%5Cbegin%7BBmatrix%7D+%5Ctext%7B+%7B%5Ba%2C+b%5D%7D%2C+%7B%5Bb%2C+c%5D%7D%2C+%7B%5Bc%2C+a%5D%7D%2C+%7B%5Ba%2C+b%5D+%2B+%5Bb%2C+c%5D%7D%2C+%7B%5Bb%2C+c%5D+%2B+%5Bc%2C+a%5D%7D%2C+%7B%5Ba%2C+b%5D+%2B+%5Bc%2C+a%5D%7D%2C+%7B%5Ba%2C+b%5D+%2B+%5Bb%2C+c%5D+%2B+%5Bc%2C+a%5D%7D%2C+%7B0%7D+%7D+%5Cend%7BBmatrix%7D+%5C%5C) 为了澄清边界图，这里有一个图。这说明了边界操作符如何映射 ![[公式]](https://www.zhihu.com/equation?tex=C_0) 的每个元素到 ![[公式]](https://www.zhihu.com/equation?tex=C_1) 的元素。

![img](https://pic3.zhimg.com/80/v2-aa8cda534bd414e775b89caaae857562_1440w.jpg)

现在我们可以开始计算第一个连通数， ![[公式]](https://www.zhihu.com/equation?tex=b_0) 。



回想一下连通数的定义：

> 第 ![[公式]](https://www.zhihu.com/equation?tex=n) 个连通数 ![[公式]](https://www.zhihu.com/equation?tex=b_n) 定义为 ![[公式]](https://www.zhihu.com/equation?tex=H_n) 的维度， ![[公式]](https://www.zhihu.com/equation?tex=b_n+%3D+dim%28H_n%29) 。

再回忆一下同调群的定义：

> 第 ![[公式]](https://www.zhihu.com/equation?tex=n) 个同调群 ![[公式]](https://www.zhihu.com/equation?tex=H_n) 定义为 ![[公式]](https://www.zhihu.com/equation?tex=H_n%3DKer%5Cpartial_n%2FIm%5Cpartial_%7Bn%2B1%7D) 。

最后，回顾一下内核的定义：

> ![[公式]](https://www.zhihu.com/equation?tex=%5Cpartial%28C_n%29) 的核（记作 ![[公式]](https://www.zhihu.com/equation?tex=Ker%28%5Cpartial%28C_n%29%29) ）是 ![[公式]](https://www.zhihu.com/equation?tex=n) 链 ![[公式]](https://www.zhihu.com/equation?tex=Z_n+%5Csubseteq+C_n) 的群，其中 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpartial%28Z_n%29+%3D+0) 。

首先，我们需要边界 ![[公式]](https://www.zhihu.com/equation?tex=C_0) 的核 ![[公式]](https://www.zhihu.com/equation?tex=Ker%5Cpartial%28C_0%29) 。记得边界映射 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpartial) 为我们提供了 ![[公式]](https://www.zhihu.com/equation?tex=C_n+%5Crightarrow+C_%7Bn-1%7D) 的映射。



在所有情况下，边界的 ![[公式]](https://www.zhihu.com/equation?tex=0) 维链是 ![[公式]](https://www.zhihu.com/equation?tex=0) ，因此 ![[公式]](https://www.zhihu.com/equation?tex=Ker%5Cpartial%28C_0%29) 是整个 ![[公式]](https://www.zhihu.com/equation?tex=0) 维链。 ![[公式]](https://www.zhihu.com/equation?tex=Ker%5Cpartial%28C_0%29+%3D+%5C%7Ba%2C+b%2C+c%5C%7D) 这形成了另一个 ![[公式]](https://www.zhihu.com/equation?tex=0) 圈的群，记为 ![[公式]](https://www.zhihu.com/equation?tex=Z_0) （或者 ![[公式]](https://www.zhihu.com/equation?tex=Z_n) ）， ![[公式]](https://www.zhihu.com/equation?tex=Z_0) 是 ![[公式]](https://www.zhihu.com/equation?tex=C_0) 的子群，即 ![[公式]](https://www.zhihu.com/equation?tex=Z_n+%5Cleq+C_n) 。加上 ![[公式]](https://www.zhihu.com/equation?tex=Z_2) 的定义， ![[公式]](https://www.zhihu.com/equation?tex=Z_0) 也与 ![[公式]](https://www.zhihu.com/equation?tex=Z_2) 同构，因此它和 ![[公式]](https://www.zhihu.com/equation?tex=C_0) 一样。



另一件事是我们需要找到同调群 ![[公式]](https://www.zhihu.com/equation?tex=H_0) 的 ![[公式]](https://www.zhihu.com/equation?tex=Im%5Cpartial_%7B1%7D) 。这形成了 ![[公式]](https://www.zhihu.com/equation?tex=Z_0) 的子群，记作 ![[公式]](https://www.zhihu.com/equation?tex=B_0) ，它是 ![[公式]](https://www.zhihu.com/equation?tex=%28n%2B1%29) 维链的边界的群。因此 ![[公式]](https://www.zhihu.com/equation?tex=B_n+%5Cleq+Z_n+%5Cleq+C_n) 。 ![[公式]](https://www.zhihu.com/equation?tex=%5Cpartial%7BC_1%7D+%3D+%5Cpartial%28%7B%5Ba%2C+b%5D%2C+%5Bb%2C+c%5D%2C+%5Bc%2C+a%5D%7D+%3D+%28a+%2B+b%29+%2B+%28b+%2B+c%29+%2B+%28c+%2B+a%29+%5C%5C+%5Cpartial%7BC_1%7D+%3D+%28a+%2B+b%29+%2B+%28b+%2B+c%29+%2B+%28c+%2B+a%29+%3D+%28a+%2B+a%29+%2B+%28b+%2B+b%29+%2B+%28c+%2B+c%29+%3D+%280%29+%2B+%280%29+%2B+%280%29+%3D+0+%5C%5C+%5Cpartial%7BC_1%7D+%3D+0)

所以 ![[公式]](https://www.zhihu.com/equation?tex=Im%5Cpartial_%7B1%7D+%3D+0) 。



因此我们计算子群 ![[公式]](https://www.zhihu.com/equation?tex=H_0+%3D+Z_0%5C+%2F%5C+B_0) ，这种情况下： ![[公式]](https://www.zhihu.com/equation?tex=Z_0+%3D+%5Ctext+%7B+%7B+%7Ba%2C+b%2C+c%7D%2C+%7B0%7D+%7D+%7D+%5C%5C+B_0+%3D+%5C%7B0%5C%7D)

所以我们用 ![[公式]](https://www.zhihu.com/equation?tex=Z_0) 中两个元素中的 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7B0%5C%7D) 的左陪集来得到商群： ![[公式]](https://www.zhihu.com/equation?tex=%5C%5C+Z_0%5C+%2F%5C+B_0+%3D+%5Ctext+%7B+%7B+%7Ba%2C+b%2C+c%7D%2C+%7B0%7D+%7D+%7D+%3D+Z_0)

总之，如果 ![[公式]](https://www.zhihu.com/equation?tex=B_n+%3D+%5C%7B0%5C%7D) ，那么 ![[公式]](https://www.zhihu.com/equation?tex=Z_n%5C+%2F%5C+B_n+%3D+Z_n) ，所以 ![[公式]](https://www.zhihu.com/equation?tex=H_0+%3D+Z_0) 。



连通数 ![[公式]](https://www.zhihu.com/equation?tex=b_0) 是 ![[公式]](https://www.zhihu.com/equation?tex=H_0+%3D+Z_0) 的维度。什么是 ![[公式]](https://www.zhihu.com/equation?tex=H_0) 的维度？它有两个元素，但是维度被定义为一个群中最小的生成元集合，因为这个群与 ![[公式]](https://www.zhihu.com/equation?tex=Z_2) 同构，所以它只有 ![[公式]](https://www.zhihu.com/equation?tex=1) 个生成元。因为整个群可以通过反复加 ![[公式]](https://www.zhihu.com/equation?tex=1) 来形成，即 ![[公式]](https://www.zhihu.com/equation?tex=1%2B1%3D0%2C+1%2B1%2B1+%3D+1) ，然后我们就能得到整个 ![[公式]](https://www.zhihu.com/equation?tex=Z_2) ，所以 ![[公式]](https://www.zhihu.com/equation?tex=Z_2) 的生成元是 ![[公式]](https://www.zhihu.com/equation?tex=1) 。



所以 ![[公式]](https://www.zhihu.com/equation?tex=b_0+%3D+dim%28H_0%29+%3D+1) ，这就是我们所期望的，这个单纯复形有 ![[公式]](https://www.zhihu.com/equation?tex=1) 个相连的分量。



现在开始计算 ![[公式]](https://www.zhihu.com/equation?tex=1) 维连通 ![[公式]](https://www.zhihu.com/equation?tex=b_1) 。这次它可能会有些不同，因为计算 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctext%7BKer%7D%5Cpartial%28C_1%29) 将变得更复杂。我们需要做一些代数运算。



所以，第一，我们需要 ![[公式]](https://www.zhihu.com/equation?tex=Z_1) ， ![[公式]](https://www.zhihu.com/equation?tex=1) 维圈的群。这是边界为 ![[公式]](https://www.zhihu.com/equation?tex=0) 的 ![[公式]](https://www.zhihu.com/equation?tex=1) 维单形的集合。记得 ![[公式]](https://www.zhihu.com/equation?tex=1) 维链是 ![[公式]](https://www.zhihu.com/equation?tex=%5C%7B+%5Ba%2Cb%5D%2C+%5Bb%2Cc%5D%2C+%5Bc%2Ca%5D%5C%7D) ，而当它应用在 ![[公式]](https://www.zhihu.com/equation?tex=Z_1) 上，它形成了 ![[公式]](https://www.zhihu.com/equation?tex=1) 维链群 ![[公式]](https://www.zhihu.com/equation?tex=C_1) 。我们将构建这样一个等式：

![[公式]](https://www.zhihu.com/equation?tex=%5C%5C+%5Cmathscr+C_1+%3D+%5Clambda_0%28%5Ba%2Cb%5D%29+%2B+%5Clambda_1%28%5Bb%2Cc%5D%29+%5Clambda_2%28%5Bc%2Ca%5D%29+%5Ctext%7B+...+%E5%85%B6%E4%B8%AD+%24%5Clambda_n+%5Cin+%5Cmathbb+Z_2%24%EF%BC%8C+%E8%BF%99%E6%98%AF%E5%90%91%E9%87%8F%E7%A9%BA%E9%97%B4%24%5Cmathscr+C_1%24%E4%B8%AD%E4%BB%BB%E4%BD%95%E5%85%83%E7%B4%A0%E7%9A%84%E4%B8%80%E8%88%AC%E5%BD%A2%E5%BC%8F+%7D+%5C%5C+%5Clambda_0%28%5Ba%2Cb%5D%29+%2B+%5Clambda_1%28%5Bb%2Cc%5D%29+%2B+%5Clambda_2%28%5Bc%2Ca%5D%29+%3D+0+%5Ctext%7B+...+%E7%84%B6%E5%90%8E%E5%8F%96%E8%BE%B9%E7%95%8C%7D+%5C%5C+%5Cpartial%28%5Clambda_0%28%5Ba%2Cb%5D%29+%2B+%5Clambda_1%28%5Bb%2Cc%5D%29+%2B+%5Clambda_2%28%5Bc%2Ca%5D%29%29+%3D+0+%5C%5C+%5Clambda_0%28a%2Bb%29+%2B+%5Clambda_1%28b%2Bc%29+%2B+%5Clambda_2%28c%2Ba%29+%3D+0+%5C%5C+%5Clambda_0%7Ba%7D+%2B+%5Clambda_0%7Bb%7D+%2B+%5Clambda_1%7Bb%7D+%2B+%5Clambda_1%7Bc%7D+%2B+%5Clambda_2%7Bc%7D+%2B+%5Clambda_2%7Ba%7D+%3D+0+%5C%5C+a%28%5Clambda_0+%2B+%5Clambda_2%29+%2B+b%28%5Clambda_0+%2B+%5Clambda_1%29+%2B+c%28%5Clambda_1+%2B+%5Clambda_2%29+%3D+0+%5Ctext%7B+...%E6%8F%90%E5%87%BA%E5%9B%A0%E6%95%B0+a%2Cb%2Cc%7D+%5C%5C)

满足这个方程后，需要把每一项的所有系数 ![[公式]](https://www.zhihu.com/equation?tex=%5Clambda_n) 加起来等于 ![[公式]](https://www.zhihu.com/equation?tex=0) ，来让 ![[公式]](https://www.zhihu.com/equation?tex=a%2Cb%2Cc) 变成 ![[公式]](https://www.zhihu.com/equation?tex=0) 。也就是说，如果整个等式等于 ![[公式]](https://www.zhihu.com/equation?tex=0) ，那么每一项都等于 ![[公式]](https://www.zhihu.com/equation?tex=0) ，如 ![[公式]](https://www.zhihu.com/equation?tex=a%28%5Clambda_0+%2B+%5Clambda_2%29+%3D+0) ，因此 ![[公式]](https://www.zhihu.com/equation?tex=%5Clambda_0+%2B+%5Clambda_2+%3D+0) 。现在我们有了一个线性方程组：

![[公式]](https://www.zhihu.com/equation?tex=%5Clambda_0+%2B+%5Clambda_2+%3D+0+%5C%5C+%5Clambda_0+%2B+%5Clambda_1+%3D+0+%5C%5C+%5Clambda_1+%2B+%5Clambda_2+%3D+0+%5C%5C+%5Ctext%7B...%E5%BE%97%E5%87%BA...%7D+%5C%5C+%5Clambda_0+%3D+%5Clambda_2+%5C%5C+%5Clambda_0+%3D+%5Clambda_1+%5C%5C+%5Clambda_1+%3D+%5Clambda_1+%5C%5C+%5Clambda_0+%3D+%5Clambda_1+%3D+%5Clambda_2)

对于上面的方程，所有的系数都是相等的。我们用一个符号替换所有的 ![[公式]](https://www.zhihu.com/equation?tex=%5Clambda) ，即 ![[公式]](https://www.zhihu.com/equation?tex=%5Clambda_0%2C+%5Clambda_1%2C+%5Clambda_2+%3D+%5Cphi) 。



现在，让我们回到1维链向量空间的一般表达式 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathscr+C_1+%3D+%5Clambda_0%28%5Ba%2Cb%5D%29+%2B+%5Clambda_1%28%5Bb%2Cc%5D%29+%2B+%5Clambda_2%28%5Bc%2Ca%5D%29) 。当我们把它的边界设为 ![[公式]](https://www.zhihu.com/equation?tex=0) 时，我们会得到 ![[公式]](https://www.zhihu.com/equation?tex=%5Clambda_0+%3D+%5Clambda_1+%3D+%5Clambda_2) ，而我建议我们用符号 ![[公式]](https://www.zhihu.com/equation?tex=%5Cphi) 代替。



因此，循环群：

![[公式]](https://www.zhihu.com/equation?tex=%5C%5C+Z_1+%5Cleq+%5Cmathscr+C_1+%3D+%5Cphi%28%5Ba%2Cb%5D%29+%2B+%5Cphi%28%5Bb%2Cc%5D%29+%2B+%5Cphi%28%5Bc%2Ca%5D%29+%5C%5C+Z_1+%3D+%5Cphi%28%5Ba%2Cb%5D+%2B+%5Bb%2Cc%5D+%2B+%5Bc%2Ca%5D%29%2C+%5Ctext%7B+...%E8%AE%B0%E4%BD%8F+%24%5Cphi%24+%E6%9D%A5%E8%87%AA+%24%5Cmathbb+Z_2%24%EF%BC%8C%E6%89%80%E4%BB%A5%E5%AE%83%E6%98%AF0%E6%88%961%E3%80%82%7D)

因此循环群只包含两个元素，它和 ![[公式]](https://www.zhihu.com/equation?tex=Z_2) 同构。

> 我会引入新的符号。如果数学结构 ![[公式]](https://www.zhihu.com/equation?tex=G_1%2CG_2) 同构，那么我们记作 ![[公式]](https://www.zhihu.com/equation?tex=G_1+%5Ccong+G_2) 。

如果 ![[公式]](https://www.zhihu.com/equation?tex=%5Cphi+%3D+0) ，那么 ![[公式]](https://www.zhihu.com/equation?tex=%5Cphi%28%5Ba%2Cb%5D+%2B+%5Bb%2Cc%5D+%2B+%5Bc%2Ca%5D%29+%3D+0%28%5Ba%2Cb%5D+%2B+%5Bb%2Cc%5D+%2B+%5Bc%2Ca%5D%29+%3D+0) ，但如果 ![[公式]](https://www.zhihu.com/equation?tex=%5Cphi+%3D+1) ，那么 ![[公式]](https://www.zhihu.com/equation?tex=%5Cphi%28%5Ba%2Cb%5D+%2B+%5Bb%2Cc%5D+%2B+%5Bc%2Ca%5D%29+%3D+1%28%5Ba%2Cb%5D+%2B+%5Bb%2Cc%5D+%2B+%5Bc%2Ca%5D%29+%3D+%5Ba%2Cb%5D+%2B+%5Bb%2Cc%5D+%2B+%5Bc%2Ca%5D) ，所以整个群是： ![[公式]](https://www.zhihu.com/equation?tex=%5C%5C+Z_1+%3D+%5Cbegin%7BBmatrix%7D+%5Ba%2Cb%5D+%2B+%5Bb%2Cc%5D+%2B+%5Bc%2Ca%5D+%5C%5C+0+%5Cend%7BBmatrix%7D)

边界群 ![[公式]](https://www.zhihu.com/equation?tex=B_1+%3D+Im%5Cpartial%28C_2%29) ，但因为 ![[公式]](https://www.zhihu.com/equation?tex=C_2) 是空集，所以 ![[公式]](https://www.zhihu.com/equation?tex=B_1+%3D+%5C%7B0%5C%7D) 。



我们可以再次计算出同源群： ![[公式]](https://www.zhihu.com/equation?tex=%5C%5CH_1+%3D+Z_1+%2F+B_1+%3D+%5Cbegin%7BBmatrix%7D+%5Ba%2Cb%5D+%2B+%5Bb%2Cc%5D+%2B+%5Bc%2Ca%5D+%5C%5C+0+%5Cend%7BBmatrix%7D)

而连通数 ![[公式]](https://www.zhihu.com/equation?tex=b_1+%3D+dim%28H_1%29+%3D+1) ，因为在群 ![[公式]](https://www.zhihu.com/equation?tex=H_1) 中，我们只有一个生成元。



所以这就是非常简单的单纯复形。我们将转到一个更大的复形。这次我将不会详细介绍，并将使用许多我已经定义或描述过的简化符号和约定俗成。



让我们来完成和之前差不多，但更复杂的单纯复形： ![[公式]](https://www.zhihu.com/equation?tex=%5C%5C+S+%3D+%5Ctext%7B%7B%5Ba%5D%2C+%5Bb%5D%2C+%5Bc%5D%2C+%5Bd%5D%2C+%5Ba%2C+b%5D%2C+%5Bb%2C+c%5D%2C+%5Bc%2C+a%5D%2C+%5Bc%2C+d%5D%2C+%5Bd%2C+b%5D%2C+%5Ba%2C+b%2C+c%5D%7D%7D)

（下面是描述）

![img](https://pic2.zhimg.com/80/v2-37142a90180939db2e405b205242729d_1440w.jpg)

注意现在我们有一个 ![[公式]](https://www.zhihu.com/equation?tex=2) 维单形 ![[公式]](https://www.zhihu.com/equation?tex=%5Ba%2Cb%2Cc%5D) ，被描绘成实心三角形。



这次我们用整个整数域 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+Z) 作为我们的系数，所以由此产生的向量空间将是无限的，而不是我们可以列出的有限空间。既然我们用了 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+Z) ，我们就必须定义“负”单形是什么意思，即 ![[公式]](https://www.zhihu.com/equation?tex=-%5Bc%2Ca%5D) 的意义？我们之前讨论过了，基本上，我们定义了两种方法，一个单纯形可以被定向，而对原定义的相反方向则被赋予一个单纯形的“负”值。



所以 ![[公式]](https://www.zhihu.com/equation?tex=%5Ba%2Cc%5D+%3D+-%5Bc%2Ca%5D) 。但 ![[公式]](https://www.zhihu.com/equation?tex=%5Ba%2Cb%2Cc%5D) 呢？有两种方法来排列 ![[公式]](https://www.zhihu.com/equation?tex=3) 个元素列表，但它只有两个方向。



如果你见过以前的定向单形:



这只有两种方法可以“围绕”循环，顺时针或者逆时针。

![[公式]](https://www.zhihu.com/equation?tex=%5Ba%2Cb%2Cc%5D) 是顺时针。

![[公式]](https://www.zhihu.com/equation?tex=%5Bc%2Ca%2Cb%5D) 也是顺时针。

![[公式]](https://www.zhihu.com/equation?tex=%5Ba%2Cc%2Cb%5D) 是逆时针，所以 ![[公式]](https://www.zhihu.com/equation?tex=%5Ba%2Cb%2Cc%5D+%3D+%5Bc%2Ca%2Cb%5D+%3D+-%5Ba%2Cc%2Cb%5D+%3D+-%5Bb%2Cc%2Ca%5D) 。



让我们从列出我们的链组开始。 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5CC_0+%3D+%5Clangle%7Ba%2Cb%2Cc%2Cd%7D%5Crangle+%5Ccong+%5Cmathbb+Z%5E4+%5C%5C+C_1+%3D+%5Clangle%7B%5Ba%2C+b%5D%2C+%5Bb%2C+c%5D%2C+%5Bc%2C+a%5D%2C+%5Bc%2C+d%5D%2C+%5Bd%2C+b%5D%7D%5Crangle+%5Ccong+%5Cmathbb+Z%5E5+%5C%5C+C_2+%3D+%5Clangle%7B%5Ba%2C+b%2C+c%5D%7D%5Crangle+%5Ccong+%5Cmathbb+Z+%5C%5C)

回想一下方括号的含义，这显然比我们在最后一个例子中构建群的方式要简单得多。注意每个群都与向量空间 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+Z%5En) 同构，其中 ![[公式]](https://www.zhihu.com/equation?tex=n) 是 ![[公式]](https://www.zhihu.com/equation?tex=n) 维链中单形的数量。



我们可以这样描述我们的链结构： ![[公式]](https://www.zhihu.com/equation?tex=%5C%5C+C_2+%5Cstackrel%7B%5Cpartial_1%7D%5Crightarrow+C_1+%5Cstackrel%7B%5Cpartial_0%7D%5Crightarrow+C_0)

我们知道，我们可以很轻易地将这个单纯复形可视化，因为它有一个连接组件和一个1维循环（一个1维孔）。因此，连通数 ![[公式]](https://www.zhihu.com/equation?tex=b_0+%3D+1%EF%BC%8C+b_1+%3D+1) ，但我们需要自己计算。



让我们从高维链群开始，即 ![[公式]](https://www.zhihu.com/equation?tex=2) 维链群。



记住， ![[公式]](https://www.zhihu.com/equation?tex=Z_n+%3D+%5Ctext%7BKer%7D%5Cpartial%28C_n%29) ![[公式]](https://www.zhihu.com/equation?tex=n) 维循环的群，它是 ![[公式]](https://www.zhihu.com/equation?tex=C_n) 的子群。而 ![[公式]](https://www.zhihu.com/equation?tex=B_n+%3D+%5Ctext%7BIm%7D%5Cpartial%28C_%7Bn%2B1%7D%29) 是 ![[公式]](https://www.zhihu.com/equation?tex=n) 维边界的群，它是 ![[公式]](https://www.zhihu.com/equation?tex=n) 维循环的子集。因此 ![[公式]](https://www.zhihu.com/equation?tex=B_n+%5Cleq+Z_n+%5Cleq+C_n) 。还记得同调群 ![[公式]](https://www.zhihu.com/equation?tex=H_n+%3D+Z_n%5C+%2F%5C+B_n) 而第 ![[公式]](https://www.zhihu.com/equation?tex=n) 个连通数是 ![[公式]](https://www.zhihu.com/equation?tex=n) 维同调群的维度。



为了得到 ![[公式]](https://www.zhihu.com/equation?tex=Z_n) ，我们需要为 ![[公式]](https://www.zhihu.com/equation?tex=C_n) 中的一般元素设置表达式。 ![[公式]](https://www.zhihu.com/equation?tex=%5C%5C+%5Cbegin%7Baligned%7D+C_2+%26%3D+%5Clambda_0%7B%5Ba%2Cb%2Cc%5D%7D%2C+%5Clambda_0+%5Cin+%5Cmathbb%7BZ%7D+%5C%5C+Z_2+%26%3D+%5Ctext%7BKer%7D%5Cpartial%7B%28C_2%29%7D+%5C%5C+%5Cpartial%7B%28C_2%29%7D+%26%3D+%5Clambda_0%7B%28%5Bb%2Cc%5D%29%7D+-+%5Clambda_0%7B%28%5Ba%2Cc%5D%29%7D+%2B+%5Clambda_0%7B%28%5Ba%2Cb%5D%29%7D+%5Ctext%7B+...%E8%AE%A9%E5%AE%83%E7%AD%89%E4%BA%8E0%E4%BB%A5%E5%BE%97%E5%88%B0%E6%A0%B8%7D+%5C%5C+%5Clambda_0%7B%28%5Bb%2Cc%5D%29%7D+-+%5Clambda_0%7B%28%5Ba%2Cc%5D%29%7D+%2B+%5Clambda_0%7B%28%5Ba%2Cb%5D%29%7D+%26%3D+0+%5C%5C+%5Clambda_0%7B%28%5Bb%2Cc%5D+-+%5Ba%2Cc%5D+%2B+%5Ba%2Cb%5D%29%7D+%26%3D+0+%5C%5C+%5Clambda_0+%26%3D+0+%5Ctext%7B+...+%24%5Clambda_0+%3D+0%24%E5%8F%AA%E6%9C%89%E4%B8%80%E4%B8%AA%E8%A7%A3%EF%BC%8C%E6%89%80%E4%BB%A5+%240%3D0%24.+%7D+%5C%5C+%5Clambda_0%7B%5Ba%2Cb%2Cc%5D%7D+%26%3D+0%2C+%5Clambda_0+%3D+0+%5Ctext%7B+...+%E6%89%80%E4%BB%A5%24C_2%24%E4%B8%AD%E6%B2%A1%E6%9C%89%E4%B8%80%E4%B8%AA%E5%85%83%E7%B4%A0%E7%AD%89%E4%BA%8E0%EF%BC%8C%E5%9B%A0%E6%AD%A4%E6%A0%B8%E4%B8%AD%E5%8F%AA%E6%9C%89%7B0%7D%7D+%5C%5C+...+%5C%5C+%5Ctext%7BKer%7D%5Cpartial%7B%28C_2%29%7D+%26%3D+%5C%7B0%5C%7D+%5C%5C+%5Cend%7Baligned%7D%5C%5C)

因为没有 ![[公式]](https://www.zhihu.com/equation?tex=3) 维单形或更高， ![[公式]](https://www.zhihu.com/equation?tex=B_2+%3D+%7B0%7D) 。因此连通数 ![[公式]](https://www.zhihu.com/equation?tex=b_2+%3D+dim%28%5C%7B0%5C%7D+%2F+%5C%7B0%5C%7D%29+%3D+0) 。这就是我们所期望的，单纯复形里没有 ![[公式]](https://www.zhihu.com/equation?tex=2) 维洞。



让我们用同样的方法处理 ![[公式]](https://www.zhihu.com/equation?tex=C_1) 。

![[公式]](https://www.zhihu.com/equation?tex=%5C%5C+%5Cbegin%7Baligned%7D+C_1+%26%3D+%5Clambda_0%5Ba%2C+b%5D+%2B+%5Clambda_1%5Bb%2C+c%5D+%2B+%5Clambda_2%5Bc%2C+a%5D+%2B+%5Clambda_3%5Bc%2C+d%5D+%2B+%5Clambda_4%5Bd%2C+b%5D%2C+%5Clambda_n+%5Cin+%5Cmathbb%7BZ%7D+%5C%5C+Z_1+%26%3D+%5Ctext%7BKer%7D%5Cpartial%7B%28C_1%29%7D+%5C%5C+%5Cpartial%7B%28C_1%29%7D+%26%3D+%5Clambda_0%7B%28a+-+b%29%7D+%2B+%5Clambda_1%7B%28b+-+c%29%7D+%2B+%5Clambda_2%7B%28c+-+a%29%7D+%2B+%5Clambda_3%7B%28c+-+d%29%7D+%2B+%5Clambda_4%7B%28d+-+b%29%7D+%5C%5C+%5Ctext%7B+...%E8%AE%A9%E5%AE%83%E7%AD%89%E4%BA%8E0%E4%BB%A5%E5%BE%97%E5%88%B0%E6%A0%B8%7D+%5C%5C+%5Clambda_0%7B%28a+-+b%29%7D+%2B+%5Clambda_1%7B%28b+-+c%29%7D+%2B+%5Clambda_2%7B%28c+-+a%29%7D+%2B+%5Clambda_3%7B%28c+-+d%29%7D+%2B+%5Clambda_4%7B%28d+-+b%29%7D+%26%3D+0+%5C%5C+%5Clambda_0a+-+%5Clambda_0b+%2B+%5Clambda_1b+-+%5Clambda_1c+%2B+%5Clambda_2c+-+%5Clambda_2a+%2B+%5Clambda_3c+-+%5Clambda_3d+%2B+%5Clambda_4d+-+%5Clambda_4b+%26%3D+0+%5Ctext+%7B+...%E6%8C%87%E5%87%BA+a%2Cb%2Cc%2Cd%7D%5C%5C+a%28%5Clambda_0+-+%5Clambda_2%29+%2B+b%28%5Clambda_1+-+%5Clambda_0+-+%5Clambda_4%29+%2B+c%28%5Clambda_2+-+%5Clambda_1+%2B+%5Clambda_3%29+%2B+d%28%5Clambda_4+-+%5Clambda_3%29+%26%3D+0+%5C%5C+%5Ctext%7B%E7%8E%B0%E5%9C%A8%E6%88%91%E4%BB%AC%E5%8F%AF%E4%BB%A5%E5%BB%BA%E7%AB%8B%E4%B8%80%E4%B8%AA%E7%BA%BF%E6%80%A7%E6%96%B9%E7%A8%8B%E7%BB%84...%7D+%5C%5C+%5Clambda_0+-+%5Clambda_2+%26%3D+0+%5C%5C+%5Clambda_1+-+%5Clambda_0+-+%5Clambda_4+%26%3D+0+%5C%5C+%5Clambda_2+-+%5Clambda_1+%2B+%5Clambda_3+%26%3D+0+%5C%5C+%5Clambda_4+-+%5Clambda_3+%26%3D+0+%5C%5C+%5Ctext%7B%E8%A7%A3%E6%96%B9%E7%A8%8B%EF%BC%8C%E6%8A%8A%E7%AD%94%E6%A1%88%E6%94%BE%E5%9B%9E%24C_1%24%E8%A1%A8%E8%BE%BE%E5%BC%8F%E4%B8%AD...%7D+%5C%5C+%5Clambda_0%28%5Ba%2Cb%5D+%2B+%5Bb%2Cc%5D+%2B+%5Bc%2Ca%5D%29+%2B+%5Clambda_3%28%5Bb%2Cc%5D+%2B+%5Ccancel%7B%5Ba%2Cc%5D%7D+%2B+%5Bc%2Cd%5D+%2B+%5Ccancel%7B%5Bc%2Ca%5D%7D+%2B+%5Bd%2Cb%5D%29+%26%3D+%5Ctext%7BKer%7D%5Cpartial%7B%28C_1%29%7D+%5C%5C+%5Ctext%7BKer%7D%5Cpartial%7B%28C_1%29%7D+%26%3D+%5Clambda_0%28%5Ba%2Cb%5D+%2B+%5Bb%2Cc%5D+%2B+%5Bc%2Ca%5D%29+%2B+%5Clambda_3%28%5Bb%2Cc%5D+%2B+%5Bc%2Cd%5D+%2B+%5Bd%2Cb%5D%29+%5C%5C+Z_1+%3D+%5Ctext%7BKer%7D%5Cpartial%7B%28C_1%29%7D+%26%5Ccong+%5Cmathbb+Z%5E2+%5Cend%7Baligned%7D%5C%5C)

现在开始求边界 ![[公式]](https://www.zhihu.com/equation?tex=B_1+%3D+Im%5Cpartial%28C_2%29) 。

![[公式]](https://www.zhihu.com/equation?tex=%5C%5C+%5Cbegin%7Baligned%7D+%5Cpartial%28C_2%29+%26%3D+%5Clambda_0%7B%28%5Bb%2Cc%5D%29%7D+-+%5Clambda_0%7B%28%5Ba%2Cc%5D%29%7D+%2B+%5Clambda_0%7B%28%5Ba%2Cb%5D%29%7D+%5Ctext+%7B...+%E8%AE%B0%E4%BD%8F+%24-%5Ba%2Cc%5D+%3D+%5Bc%2Ca%5D%24+...%7D+%5C%5C+%5Cpartial%28C_2%29+%26%3D+%5Clambda_0%7B%28%5Bb%2Cc%5D+%2B+%5Bc%2Ca%5D+%2B+%5Ba%2Cb%5D%29%7D+%5C%5C+B_1+%3D+Im%5Cpartial%28C_2%29+%26%3D+%5C%7B%5Clambda_0%7B%28%5Bb%2Cc%5D+%2B+%5Bc%2Ca%5D+%2B+%5Ba%2Cb%5D%29%7D%5C%7D%2C+%5Clambda_0+%5Cin+%5Cmathbb+Z+%5C%5C+B_1+%26%5Ccong+Z+%5C%5C+H_1+%3D+Z_1%5C+%2F%5C+B_1+%26%3D+%5C%7B%5Clambda_0%28%5Ba%2Cb%5D+%2B+%5Bb%2Cc%5D+%2B+%5Bc%2Ca%5D%29+%2B+%5Clambda_3%28%5Bb%2Cc%5D+%2B+%5Bc%2Cd%5D+%2B+%5Bd%2Cb%5D%29%5C%7D%5C+%2F%5C+%5C%7B%5Clambda_0%7B%28%5Bb%2Cc%5D+%2B+%5Bc%2Ca%5D+%2B+%5Ba%2Cb%5D%29%7D%5C%7D+%5C%5C+H_1+%26%3D+%5C%7B%5Clambda_3%28%5Bb%2Cc%5D+%2B+%5Bc%2Cd%5D+%2B+%5Bd%2Cb%5D%29%5C%7D+%5Ccong+%5Cmathbb+Z+%5Cend%7Baligned%7D)

另一种更容易计算出商群 ![[公式]](https://www.zhihu.com/equation?tex=H_1+%3D+Z_1%5C+%2F%5C+B_1) 的方法是注意 ![[公式]](https://www.zhihu.com/equation?tex=Z_n%2C+B_n) 和 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+Z%5En) 中的什么同构。在这种情况下：

![[公式]](https://www.zhihu.com/equation?tex=%5C%5CZ_1+%5Ccong+%5Cmathbb+Z%5E2+%5C%5C+B_1+%5Ccong+%5Cmathbb+Z%5E1+%5C%5C+H_1+%3D+%5Cmathbb+Z%5E2%5C+%2F%5C+%5Cmathbb+Z+%3D+%5Cmathbb+Z)

所以因为 ![[公式]](https://www.zhihu.com/equation?tex=H_1+%5Ccong+%5Cmathbb+Z) ， ![[公式]](https://www.zhihu.com/equation?tex=H_1) 的连通数是 ![[公式]](https://www.zhihu.com/equation?tex=1) ，因为 ![[公式]](https://www.zhihu.com/equation?tex=%5Cmathbb+Z) 的维度是 ![[公式]](https://www.zhihu.com/equation?tex=1) （它只有一个生成元）。



我想你们现在已经明白了，我不打算讲所有计算连通数 ![[公式]](https://www.zhihu.com/equation?tex=b_0) 的细节了，它应该是 ![[公式]](https://www.zhihu.com/equation?tex=1) ，因为只有一个连接的分量。



## 预告

我们已经掌握了如何手工计算简单的单纯复形的同调群和连通数。但是我们需要开发一些新的工具，这样我们就可以让计算机算法来处理一些真实的，通常更大的单纯复形的计算。下一节我们将会看到线性代数是如何为我们提供一种有效的方法的。



## 参考文献（网站）



\1. [Applying Topology to Data, Part 1: A Brief Introduction to Abstract Simplicial and Čech Complexes.](https://link.zhihu.com/?target=http%3A//dyinglovegrape.com/math/topology_data_1.php)

\2. [http://www.math.uiuc.edu/~r-ash/Algebra/Chapter4.pdf](https://link.zhihu.com/?target=http%3A//www.math.uiuc.edu/~r-ash/Algebra/Chapter4.pdf)

\3. [Group (mathematics)](https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/Group_(mathematics))

\4. [Homology Theory — A Primer](https://link.zhihu.com/?target=https%3A//jeremykun.com/2013/04/03/homology-theory-a-primer/)

\5. [http://suess.sdf-eu.org/website/lang/de/algtop/notes4.pdf](https://link.zhihu.com/?target=http%3A//suess.sdf-eu.org/website/lang/de/algtop/notes4.pdf)

\6. [http://www.mit.edu/~evanchen/napkin.html](https://link.zhihu.com/?target=http%3A//www.mit.edu/~evanchen/napkin.html)



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

