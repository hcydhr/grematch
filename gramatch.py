import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from skimage import measure, draw
from scipy.optimize import curve_fit, leastsq
import math

# happy


def get_ellipse(e_x, e_y, a, b, e_angle):
    angles_circle = np.arange(0, 2 * np.pi, 0.01)
    x = []
    y = []
    for angles in angles_circle:
        or_x = a * math.cos(angles)
        or_y = b * math.sin(angles)
        length_or = math.sqrt(or_x * or_x + or_y * or_y)
        or_theta = math.atan2(or_y, or_x)
        new_theta = or_theta + e_angle / 180 * 3.14159
        new_x = e_x + length_or * math.cos(new_theta)
        new_y = e_y + length_or * math.sin(new_theta)
        x.append(new_x)
        y.append(new_y)

    return x, y


def ellipsecurve(x, y, a, b, c, d, e, f):  # 椭圆方程
    return a * x * x + b * x * y + c * y * y + d * x + e * y + f


def ellipsefit(unlines):
    ellipses = []
    for unline in unlines:
        if len(unline) >= 6:
            # x=unline[:,0,0]
            # y = unline[:, 0, 1]
            ellipse = cv2.fitEllipse(unline)
            leftmost = tuple(unline[unline[:, :, 0].argmin()][0])
            rightmost = tuple(unline[unline[:, :, 0].argmax()][0])
            topmost = tuple(unline[unline[:, :, 1].argmin()][0])
            bottommost = tuple(unline[unline[:, :, 1].argmax()][0])
            startplot = unline[0]
            stopplot = unline[unline.shape[0] - 1]
            ellipses.append(
                [ellipse, leftmost, rightmost, topmost, bottommost, startplot, stopplot]
            )

            # print(ellipse)
    return ellipses


def unlinesget(cnt, mean_disten):
    unlines = []
    unline = []
    c = False
    linelocl = 0
    for i, plot in enumerate(cnt):
        disten = np.linalg.norm(cnt[i] - cnt[i - 1])  # 计算线段长度
        if disten > mean_disten:
            linelocl = i
            c = True
            break
    for i in range(linelocl):
        temp = cnt[-1].copy()
        cnt[1 : cnt.shape[0]] = cnt[0 : cnt.shape[0] - 1]
        cnt[0] = temp

    if c == False:
        for i, plot in enumerate(cnt):
            unline.append(cnt[i])
        unlines.append(cnt)
    if c == True:
        i = cnt.shape[0] - 1
        disten1 = np.linalg.norm(cnt[i - 1] - cnt[i])
        while i > 0 or disten1 < mean_disten:
            disten1 = np.linalg.norm(cnt[i - 1] - cnt[i])
            if disten1 < mean_disten:
                unline.append(cnt[i])
            else:
                unline.append(cnt[i])
                unline = np.array(unline)
                unlines.append(unline)
                unline = []
            i = i - 1

    for i, unline in enumerate(unlines):
        if len(unline) == 2:
            del unlines[i]
    unlines = np.array(unlines)

    # print(unlines)
    return unlines


def lineget(cnt, mean_disten):
    lines = []  # 初始化线段表
    for i, plot in enumerate(cnt):
        distens1 = np.linalg.norm(cnt[i] - cnt[i - 1])
        distens2 = np.linalg.norm(cnt[i - 2] - cnt[i - 3])
        # print(distens1)
        if distens1 > (mean_disten):
            lines.append(np.array([cnt[i], cnt[i - 1]]))
            if distens2 > (mean_disten):
                lines.append(np.array([cnt[i - 1], cnt[i - 2]]))
    # if np.array([cnt[i],cnt[i-1]]) in lines:
    # if cnt[i] in lines:
    # lines.append(np.array([cnt[i+1], cnt[i]]))
    # for i,line in enumerate(lines):
    #    a=lines[i-1][0,0]
    #    b=lines[i][1,0]
    #    if  a==b :
    #        print(lines[i-1][0].all)
    return lines


def mean_disten(cnt):  # 点集平均距离
    mean_disten = 0
    plotlast = cnt[0]
    for plot in cnt:
        mean_disten = mean_disten + np.linalg.norm(plot - plotlast)
    mean_disten = mean_disten / cnt.shape[0]

    return mean_disten


def gra(img):  # 求几何基元
    elength = 0.003  # 多边形拟合误差=周长×e
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换灰度模式
    imgx = cv2.imread("b.png")
    edges = cv2.Canny(imgray, 50, 150, apertureSize=3)
    plt.imshow(np.rot90(imgx))
    # blur = cv2.GaussianBlur(imgray, (5, 5), 0)
    blur = imgray
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓搜索不连续
    contours = measure.find_contours(thresh, 0.5)

    for i, cnt in enumerate(contours):
        mean_dis = mean_disten(cnt)
        # plt.plot( [0, mean_dis],[i, i])#平均长度图示
        # print(cnt.shape[0])
        cnt = cnt.reshape((cnt.shape[0], 1, 2))
        cnt = np.int32(cnt)
        epsilon = elength * cv2.arcLength(cnt, True)  # 多边形拟合偏差
        approx = cv2.approxPolyDP(cnt, epsilon, True)  # 多边形拟合
        x = cnt[:, 0, 0]
        y = cnt[:, 0, 1]
        # plt.scatter(x, y)#轮廓点
        # print(approx.shape)
        xa = approx[:, 0, 0]  # 拟合后
        ya = approx[:, 0, 1]
        # plt.scatter(xa, ya)#拟合点

        # 提取直线
        lines = lineget(approx, mean_dis)
        print(lines)
        # 画直线
        for i, line in enumerate(lines):
            plt.plot(
                [line[0, 0, 0], line[1, 0, 0]], [line[0, 0, 1], line[1, 0, 1]], "r"
            )
        # 提取弧线
        unlines = unlinesget(approx, mean_dis)
        # 画弧线
        for i, unline in enumerate(unlines):
            if unline.shape[0] > 2:
                for j, plot in enumerate(unline):
                    if j > 0:
                        plt.plot(
                            [unline[j][0, 0], unline[j - 1][0, 0]],
                            [unline[j][0, 1], unline[j - 1][0, 1]],
                            "r",
                        )
        # 拟合椭圆
        ellipses = ellipsefit(unlines)
        print(ellipses)
        # 画椭圆
        for ellipse in ellipses:
            ox = ellipse[0][0][0]
            oy = ellipse[0][0][1]
            rs = ellipse[0][1][0]
            rl = ellipse[0][1][1]
            theta = ellipse[0][2]
            # print(ellipse[1])
            # plt.plot([ox,ox+rs/2], [oy,oy])
            # plt.plot([ox, ox ], [oy, oy + rl/2])
            # x = y = np.arange(-4, 4, 0.1)
            # x, y = np.meshgrid(x, y)
            # plt.contour(x, y, x ** 2 / 9 + y ** 2 - 1, [0])
            x, y = get_ellipse(ox, oy, rs / 2, rl / 2, theta)
            ell = [x, y]
            for i, xline in enumerate(x):
                if xline < ellipse[1][0] or xline > ellipse[2][0]:
                    del ell[0][i]
                    del ell[1][i]
            for i, xline in enumerate(y):
                if xline < ellipse[1][1] or xline > ellipse[2][1]:
                    del ell[0][i]
                    del ell[1][i]
            plt.plot(ell[0], ell[1], "b")
            plt.plot(
                [ellipse[0][0][0], ellipse[6][0, 0]],
                [ellipse[0][0][1], ellipse[6][0, 1]],
                "r--",
            )
            plt.plot(
                [ellipse[5][0, 0], ellipse[0][0][0]],
                [ellipse[5][0, 1], ellipse[0][0][1]],
                "r--",
            )

        # print(line)

    plt.show()


if __name__ == "__main__":
    img1 = cv2.imread("a.png")
    img2 = cv2.imread("p1.png")
    gra(img1)
