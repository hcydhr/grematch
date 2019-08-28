# 基于几何基元的匹配

## 一、轮廓提取

```python
contours = measure.find_contours(thresh, 0.5)
```

| 原图                                                         | 轮廓提取                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![1566951273012](/home/hechenxu/PycharmProjects/grematch/1566951273012.png) | ![1566951260981](/home/hechenxu/PycharmProjects/grematch/1566951260981.png) |



## 二、轮廓分割

```python
approx = cv2.approxPolyDP(cnt, epsilon, True)  # 多边形拟合
```

| 多边形拟合                                                   | 轮廓分割                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![1566951586020](/home/hechenxu/PycharmProjects/grematch/1566951586020.png) | ![1566951799318](/home/hechenxu/PycharmProjects/grematch/1566951799318.png) |



## 三、基元拟合

![1566951883167](/home/hechenxu/PycharmProjects/grematch/1566951883167.png)
$$
C P=\left\{G F_{i} | i=1,2, \ldots, n\right\}
$$

$$
G F_{i}=\left\{\begin{array}{ll}{L_{i},} & {i \in I_{1}} \\ {C_{i},} & {i \in I_{c}}\end{array}\right.
$$



## 四、基元匹配

|      | 模板                                                         | 目标                                                         |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 原图 | ![1566953172172](/home/hechenxu/PycharmProjects/grematch/1566953172172.png) | ![1566953186875](/home/hechenxu/PycharmProjects/grematch/1566953186875.png) |
|      |                                                              |                                                              |
|      |                                                              |                                                              |

