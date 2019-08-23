# 基于几何基元的匹配方法（Geometric primitive based matching method）

[TOC]

**步骤**

1、提取轮廓

一个目标物体轮廓上所有点的点集（坐标位置） S={ (xi,yi) | i=1,2,3...m}

2、轮廓分割

将点集分割成线段、圆弧

3、基元拟合

用点集拟合出用几何参数表达的形状方程

4、基元匹配

利用几何方程参数进行匹配

## 一、提取轮廓（contour）

```
contours，hierarchy = cv2.find_contours(thresh, 0.5)
#contours是图像中所有轮廓的Python列表。每个单独的轮廓是对象的边界点的（x，y）坐标的Numpy阵列。
```

> Suzuki S , Be K . Topological structural analysis of digitized binary images by border following[J]. Computer Vision Graphics and Image Processing, 1985, 30(1):32-46.

## 二、轮廓分割

对于给定数量的基元（n），可以是弧或线段，目标是划分边界数据集（按逆时针顺序排列）：

轮廓是几何基元的集合，关键是找到分段点

| 分割点                               | 图示                                                         |
| ------------------------------------ | ------------------------------------------------------------ |
| 拐点：轮廓的切线是不连续的           | ![1566347296152](/home/hechenxu/.config/Typora/typora-user-images/1566347296152.png) |
| 平滑连接：其切线是连续的，曲率不连续 | ![1566347308809](/home/hechenxu/.config/Typora/typora-user-images/1566347308809.png) |

### 降噪

链码表示中，代表切线方向的“代码”被分配给曲线上的每个点，因此差分链码是两个相邻点之间的切线方向的变化。8链码对噪声非常敏感，使用噪声过滤技术来平滑数据。

每个轮廓上点的坐标取其领域h点集的坐标均值。
$$
\begin{array}{l}{x_{j}=\frac{1}{2 h+1} \sum_{j-h \leq l \leq j+h} x_{l}} \\ {y_{j}=\frac{1}{2 h+1} \sum_{j-h \leq l \leq j+h} y_{l}}\end{array}
$$
其中h是点pi处领域宽度，邻域平均的应用具有使边界率变化变弱。

### 链码

> 从[边界](https://baike.baidu.com/item/边界/458161)（曲线）起点S开始，按[顺时针](https://baike.baidu.com/item/顺时针/9965844)方向观察每一线段走向，并用相应的指向符表示，结果就形成表示该边界（曲线）的数码[序](https://baike.baidu.com/item/序/1302588)列，称为原链码。

$$
\zeta_{j, k}=\tan ^{-1}\left[\frac{y_{j}-y_{j-k}}{x_{j}-x_{j-k}}\right], \quad j=1,2, \ldots, m
$$
m个点的序列描述闭合轮廓。S的链码由以下序列组成，k为支撑长度，下图为k=1的情况。

k越大噪声影响越小

![1566348560455](/home/hechenxu/.config/Typora/typora-user-images/1566348560455.png)

### 差分链码

$$
\begin{aligned} \delta_{j, k} &=\tan ^{-1}\left[\frac{y_{j+k}-y_{j}}{x_{j+k}-x_{j}}\right]-\tan ^{-1}\left[\frac{y_{j}-y_{j-k}}{x_{j}-x_{j-k}}\right] \\ j &=1,2, \ldots, m \end{aligned}
$$
其中δj，k是在点pj处具有支撑长度k的切线角的变化。

差分链码可以反应切线方向的变化率。

### **具体流程**

| 原始轮廓                                                     | 均值滤波                                                     | 链码                                                         | 差分链码                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![1566520582547](/home/hechenxu/.config/Typora/typora-user-images/1566520582547.png) | ![1566520595139](/home/hechenxu/.config/Typora/typora-user-images/1566520595139.png) | ![1566520610035](/home/hechenxu/.config/Typora/typora-user-images/1566520610035.png) | ![1566520624041](/home/hechenxu/.config/Typora/typora-user-images/1566520624041.png) |
| 右下角为U1，逆时针依次为U2、U3、U4                           |                                                              | 平滑连接成为倾斜段的两个端点                                 | δj，k序列中的两个尖峰表示拐点                                |

### 拐点的提取

![1566349244676](/home/hechenxu/.config/Typora/typora-user-images/1566349244676.png)

差分链码减去均值u的绝对值、代表偏离平均值的绝对值，用绝对值建立高斯分布，在某个置信范围内选取阈值。将偏离均值最大的一部分点作为分段点。
$$
T_{b}=z_{1-2 \alpha} \sigma_{b}
$$

**阈值**由标准差求得                                ![1566349395082](/home/hechenxu/.config/Typora/typora-user-images/1566349395082.png)

**非极大值抑制**判断标准：大于阈值且为邻域最大值
$$
\begin{array}{l}{\tilde{\delta}_{j, k} \geq T_{\tilde{b}}} \\ {\text { and: }} \\ {\tilde{\delta}_{j, k}>\tilde{\delta}_{l, k}, \quad \text { for } j-k \leq l \leq j+k, \quad \text { and } l \neq j}\end{array}
$$
符合条件的点视为拐点。

### 平滑连接的分段点

为了稳定后续操作，首先通过将比例因子乘以切线$，k来标准X和Y方向上的比例。
$$
\overline{\zeta}_{j, k}=\frac{m}{2 \pi} \zeta_{j, k}, \quad j=1,2, \ldots, m
$$

分段线性逼近，得到平滑连接点。数据和近似函数之间的最大误差是检测分裂点的标准。
$$
E_{\infty}\left(u_{1}, u_{i+1}\right)=\max _{j} \epsilon_{i, j}
$$
逼近误差，其中ei，j是区间i的第j个点与其近似函数之间的欧几里德距离，并且（U1，Ui+1）是区间的两个端点。

达到分段数量或者误差低于某一个值时停止分裂。

|                                                              |                                                              |                                                              |                                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![1566520533967](/home/hechenxu/.config/Typora/typora-user-images/1566520533967.png) | ![1566520517424](/home/hechenxu/.config/Typora/typora-user-images/1566520517424.png) | ![1566520542683](/home/hechenxu/.config/Typora/typora-user-images/1566520542683.png) | ![1566520548865](/home/hechenxu/.config/Typora/typora-user-images/1566520548865.png) |



## 三、基元拟合

圆弧：近似段不平行于X轴

线段：近似段与X轴平行（斜率为零）。
$$
\begin{array}{l}{e_{j}^{2}=\left(f\left(x_{j}, y_{j}\right)\right)^{2}} \\ {\text { where: }} \\ {f(x, y)=A x^{2}+B x y+C y^{2}+D x+E y+F=0}\end{array}
$$
直线的误差
$$
\begin{aligned} \epsilon_{i, j}^{2} &=\left(x_{i, j} \cos \theta_{i}+y_{i, j} \sin \theta_{i}-d_{i}\right)^{2}, \quad \text { for } \\ j &=1,2, \ldots, m_{i}, \quad \text { and } i \in \mathcal{I}_{l} \end{aligned}
$$
圆弧的误差
$$
\begin{aligned} e_{i, j}^{2} &=\left(\left(\left(x_{i, j}-a_{i}\right)^{2}+\left(y_{i, j}-b_{i}\right)^{2}\right)^{1 / 2}-r_{i}\right)^{2}, \quad \text { for } \\ j &=1,2, \ldots, m_{i}, \quad \text { and } i \in \mathcal{I}_{c} \end{aligned}
$$

### 调整分段点

![1566351842866](/home/hechenxu/.config/Typora/typora-user-images/1566351842866.png)

![1566351886634](/home/hechenxu/.config/Typora/typora-user-images/1566351886634.png)

选择最小的步长，D = 1.在步骤1中，断点向左或向右逐点移动，直到每对相邻间隔的误差范数最小化。步骤2检查在前一次迭代中是否有任何改进。如果误差范数已经减小，则再执行一次迭代;否则，算法终止。由于这是严格减少的过程，因此算法无法循环。

改进算法先用大步长![1566352036710](/home/hechenxu/.config/Typora/typora-user-images/1566352036710.png)

### 相似实验结果

1.提取轮廓

2.[多边形近似](https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm)

```python
epsilon = elength* cv2.arcLength(cnt, True)  # 多边形拟合偏差epsilon =0.05*周长
```

3.轮廓分割

阈值T=mean_disten点集平均距离

当两点之间距离大于mean_disten时视为线段，点集数量大于6的时候拟合为椭圆

4.基元拟合用最小二乘法



![1566373734375](/home/hechenxu/.config/Typora/typora-user-images/1566373734375.png)



结果为几何基元
$$
C P=\left\{G F_{i} | i=1,2, \ldots, n\right\}
$$

$$
G F_{i}=\left\{\begin{array}{ll}{L_{i},} & {i \in I_{1}} \\ {C_{i},} & {i \in I_{c}}\end{array}\right.
$$

## 四、基元匹配

**The shape model**

描述CP
$$
L_{i}=\left\{k_{i}, \alpha_{i}\right\}, i \in I_{1}
\left(k_{i}=l_{i} / l_{1}\right)
$$

$$
C_{i}=\left\{\lambda_{i}, \alpha_{i}, \pm \phi_{i}\right\}, i \in I_{\mathrm{c}}
\left(\lambda_{i}=r_{i} / l_{1}\right)
$$

![1566354251768](/home/hechenxu/.config/Typora/typora-user-images/1566354251768.png)

圆弧弦长
$$
k_{i}=2 \times \lambda_{i} \times \sin \left(\phi_{i} / 2\right), i \in I_{\mathrm{c}}
$$


> [法线](https://baike.baidu.com/item/法线)式：**x·cosα+ysinα-p=0【适用于不平行于坐标轴的直线】**
>
> 过原点向直线做一条的垂线段，该垂线段所在直线的倾斜角为α，p是该线段的长度

$$
L_{i} : X \cos \omega_{i}+Y \sin \omega_{i}=\delta_{i}, i=1,2, \ldots, n
$$

$$
C_{i} :\left(X-u_{i}\right)^{2}+\left(Y-v_{i}\right)^{2}=\lambda_{i}^{2}, i \in I_{\mathrm{c}}
$$



假设L1为单位线段
$$
\left(\begin{array}{l}{X^{s}} \\ {Y^{s}}\end{array}\right)=l_{1}\left(\begin{array}{cc}{\cos \theta_{1}} & {-\sin \theta_{1}} \\ {\sin \theta_{1}} & {\cos \theta_{1}}\end{array}\right)\left(\begin{array}{l}{X} \\ {Y}\end{array}\right)+\left(\begin{array}{l}{x_{1}} \\ {y_{1}}\end{array}\right)
$$

$$
\begin{array}{l}{L_{i} :\left(X^{s}-x_{1}\right) \cos \left(\omega_{i}+\theta_{1}\right)+\left(Y^{s}-y_{1}\right) \sin \left(\omega_{i}+\theta_{1}\right)} \\ {=\delta_{i} l_{1}, i \in I_{c}}\end{array}
$$

$$
C_{i} :\left(X^{s}-\overline{u}_{i} l_{1}-x_{1}\right)^{2}+\left(Y_{s}-\overline{v}_{i} l_{1}-y_{1}\right)^{2}=\left(\lambda_{i} l_{1}\right)^{2}, i \in I_{c}
$$

形状仿射变换变换后线方程 与（比例）（角度）（位移）参数有关
$$
E_{i, j}=\left\{\begin{array}{ll}{\left|\left(x_{i, j}-x_{1}\right) \cos \left(\theta_{1}+\omega_{i}\right)+\left(y_{i, j}-y_{1}\right) \sin \left(\theta_{1}+\omega_{i}\right)-\delta_{i} l_{1}\right|,} & {j=1,2, \ldots, m_{i} \quad i \in I_{i}} \\ {\left|\left(x_{i, j}-\overline{u}_{i} l_{1}-x_{1}\right) \cos \left(\psi_{i, j}\right)+\left(y_{i, j}-\overline{v}_{i} l_{1}-y_{1}\right) \sin \left(\psi_{i, j}\right)-\lambda_{i} l_{1}\right|,} & {j=1,2, \ldots, m_{i} \quad i \in I_{c}}\end{array}\right.
$$

$$
\begin{array}{l}{\min _{x_{1} \cdot y_{1}, t_{1}, E_{m} \geq 0, \theta_{1}} E_{\max }} \\ {\text { subject to } E_{i, j} \leq E_{\max }, j=1,2, \ldots, m_{i} ; i=1,2, \ldots, n}\end{array}
$$

最小化误差。得到匹配形状的位置、比例、角度



> [1] Ventura J A , Wan W . Accurate matching of two-dimensional shapes using the minimal tolerance zone error[J]. Image and Vision Computing, 1997, 15(12):889-899.
>
> [2] Chen J M , Ventura J A , Wu C H . Segmentation of planar curves into circular arcs and line segments[J]. Image and Vision Computing, 1996, 14(1):71-83.