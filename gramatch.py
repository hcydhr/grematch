#_*_coding:utf-8_*_
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from skimage import measure, draw
from scipy.optimize import curve_fit, leastsq
import math
import copy
def drawMatches(img1, gras1, img2, gras2):
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]
    out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype="uint8")
    out[:rows1, :cols1] = np.dstack([img1[:, :, 0], img1[:, :, 0], img1[:, :, 0]])
    out[:rows2, cols1:] = np.dstack([img2[:, :, 0], img2[:, :, 0], img2[:, :, 0]])
    num = 0
    for i,gra in enumerate(gras1):
        if len(gra)==2:
            [[x1, y1]] = gra[0]
            if len(gras2[i]) == 2:
                [[x2, y2]] = gras2[i][0]
        if len(gra)==1:
            [x1, y1] = gra[0][0][0]
            if len(gras2[i]) == 1:
                [x2, y2] = gras2[i][0][0][0]
        #j = kp2[num]
        num += 1
        #[[x1, y1]] = i
        #[[x2, y2]] = j
        a = np.random.randint(0, 256)
        b = np.random.randint(0, 256)
        c = np.random.randint(0, 256)
        cv2.line(
            out,
            (int(np.round(x1)), int(np.round(y1))),
            (int(np.round(x2) + cols1), int(np.round(y2))),
            (a, b, c),
            10,
            shift=0,
        )

    for i,gra in enumerate(gras1):
        if len(gra)==2:
            [[x1, y1]] = gra[1]
            if len(gras2[i]) == 2:
                [[x2, y2]] = gras2[i][1]
        #j = kp2[num]
        num += 1
        #[[x1, y1]] = i
        #[[x2, y2]] = j
        a = np.random.randint(0, 256)
        b = np.random.randint(0, 256)
        c = np.random.randint(0, 256)
        cv2.line(
            out,
            (int(np.round(x1)), int(np.round(y1))),
            (int(np.round(x2) + cols1), int(np.round(y2))),
            (a, b, c),
            10,
            shift=0,
        )
    return out

def fami(gras1,gras2):
    for i,gra in enumerate(gras1):
        if len(gra)==2:

            print('')
    plt.show()

def result(x,y,gras1,gras2):#将几何基元列表重新排序，使得对应基元序号对应
    for i in range(x+1):
        l = len(gras1)
        tmp=gras1[0]
        gras1[0:l-1]=gras1[1:l]
        gras1[-1]=tmp
    for i in range(y+1):
        l = len(gras2)
        tmp=gras2[0]
        gras2[0:l-1]=gras2[1:l]
        gras2[-1]=tmp
    print(gras1,gras2)
    return gras1,gras2


def matsslist(mats):#得到排序后的几何基元列表
    # nor=mats[0][0]
    # for i, mat in enumerate(mats):
    #     mats[i][0] = (mats[i][0] * 100 / nor)
    matss = []
    for i,mat in enumerate(mats):
        l=len(mats)
        tmp=mats[0]
        mats[0:l-1]=mats[1:l]
        mats[-1]=tmp
        #matsc=mats.copy()
        matsc=copy.deepcopy(mats)
        nor1 = matsc[0][0]
        for i,mat in enumerate(matsc):
            matsc[i][0]=(matsc[i][0]*100/nor1)
        #print(mats)
        matss.append(copy.deepcopy(matsc))
    return matss

def match(mats1,mats2):#匹配几何基元，输出相似度最大的几何基元位置
    matss1=matsslist(mats1)
    #print(mats1)
    matss2 = matsslist(mats2)
    #print(mats1)
    x=0
    y=0
    costmin=1000
    cost=1000
    for i,matss1s in enumerate(matss1):
        for j,matss2s in enumerate(matss2):
            if len(matss1s)>=len(matss2s):
                # f=0
                if cost<costmin:
                    costmin=cost
                    x=i
                    y=j-1
                # # print(matss1s[k],mat)
                cost = 0
                for k,mat in enumerate(matss2s):
                    if mat.shape[0] == 2 and matss1s[k].shape[0] == 2:
                        cost = cost + np.linalg.norm(mat - matss1s[k])
                    if mat.shape[0] == 2 and matss1s[k].shape[0] == 3:
                        cost = cost +  matss1s[k][2]
                    if mat.shape[0] == 3 and matss1s[k].shape[0] == 2:
                        cost = cost +  mat[2]
                    if mat.shape[0] == 3 and matss1s[k].shape[0] == 3:
                        cost = cost +  np.linalg.norm(matss1s[k][2] - mat[2])
                print(cost,'---',matss1s,'---',matss2s)
        print('--------------')
    print(matss1[x],matss2[y])
    # nor2 = mats2[0][0]
    # for i, mat in enumerate(mats2):
    #     mats2[i][0] = (mats2[i][0] * 100 / nor2)
    # matss=[]
    # for i,mat in enumerate(mats1):
    #     l=len(mats1)
    #     tmp=vmats1[0]
    #     mats1[0:l-1]=mats1[1:l]
    #     mats1[-1]=tmp
    #     mats1c=mats1.copy()
    #     nor1 = mats1c[0][0]
    #     for i,mat in enumerate(mats1c):
    #         mats1c[i][0]=(mats1c[i][0]*100/nor1)
    #     #print(mats)
    #     matss.append(copy.deepcopy(mats1c))
        # cost=0
        # for i,mat1 in enumerate(mats1c):
        #     if mats1c[i].shape[0] == 2 and mats2[i].shape[0] == 2:
        #         cost = cost + np.linalg.norm(mats1c[i] - mats2[i])
        #     if mats1c[i].shape[0] == 2 and mats2[i].shape[0] == 3:
        #         cost = cost +  mats2[i][2]
        #     if mats1c[i].shape[0] == 3 and mats2[i].shape[0] == 2:
        #         cost = cost +  mats1c[i][2]
        #     if mats1c[i].shape[0] == 3 and mats2[i].shape[0] == 3:
        #         cost = cost +  np.linalg.norm(mats1c[i][2] - mats2[i][2])
        # print(cost,'++++++++++++++++')
    #
    # for i,mat in enumerate(mats2):
    #     for j,mat1 in enumerate(matss ):
    #         cost = 0
    #         for k, mat2 in enumerate(matss[j]):
    #             if matss[j][k].shape[0] == 2 and mats2[k].shape[0] == 2:
    #                 cost = cost + np.linalg.norm(matss[j][k] - mats2[k])
    #         print(cost)







    #    for i,mats in enumerate(matss):


    return x,y


def angle(v1, v2):
  dx1 = v1[2] - v1[0]
  dy1 = v1[3] - v1[1]
  dx2 = v2[2] - v2[0]
  dy2 = v2[3] - v2[1]
  angle1 = math.atan2(dy1, dx1)
  angle1 = int(angle1 * 180/math.pi)
  # print(angle1)
  angle2 = math.atan2(dy2, dx2)
  angle2 = int(angle2 * 180/math.pi)
  # print(angle2)
  if angle1*angle2 >= 0:
    included_angle = abs(angle1-angle2)
  else:
    included_angle = abs(angle1) + abs(angle2)
    if included_angle > 180:
      included_angle = 360 - included_angle
  return included_angle


def degree(a,b,o):#求圆周角
    x=a-o
    y=b-o
    #x = np.array([2, -5,0,0])
    #y = np.array([-4, 0,0,0])
    x=np.array([x[0],x[1],0,0])
    y = np.array([y[0], y[1], 0, 0])
    # 两个向量
    ang=angle(x,y)
    # 变为角度
    #print(ang)
    return (ang)


# happy
def matgra(gras):
    mats=[]
    matsla=[]
    for i,gra in enumerate(gras):#格式转换
        #print(gra)
        if len(gra)==2:
            mats.append(np.array([gra[0][0],gra[1][0]]))
            #print(mats)
        if len(gra) == 1:
            degree(np.array(gra[0][1]),np.array( gra[0][2]),np.array(gra[0][0][0]))
            mats.append(np.array([gra[0][8],gra[0][9],gra[0][7]]))

    tmpmat = mats[-1]
    for j,gra in enumerate(mats):
        #print(gra[0])
        length= np.linalg.norm(gra[0]- gra[1])
        alfa=degree(mats[j][1],tmpmat[0],mats[j][0])
        #print(alfa)
        tmpmat = mats[j].copy()
        mats[j][0]=np.array(length)
        mats[j][1]=np.array(alfa)
    #print(mats)
    for k,gra in enumerate(mats):
        if gra.shape[0]==2:
            matsla.append(np.array([gra[0][0],gra[1][0]]))
        if gra.shape[0]==3:
            matsla.append(np.array([gra[0],gra[1],gra[2]]))
    #print(matsla)
    return matsla





def get_ellipse(e_x, e_y, a, b, e_angle):#由椭圆参数得到椭圆点集、用于画图
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


def ellipsefit(unlines):#椭圆拟合
    ellipses = []
    for unline in unlines:
        if len(unline) >= 6:
            # x=unline[:,0,0]
            # y = unline[:, 0, 1]
            ellipse = cv2.fitEllipse(unline)#ellipse（圆心坐标）（长短轴长度）（旋转角度））
            leftmost = tuple(unline[unline[:, :, 0].argmin()][0])#左极点、右极点、上极点、下极点
            rightmost = tuple(unline[unline[:, :, 0].argmax()][0])
            topmost = tuple(unline[unline[:, :, 1].argmin()][0])
            bottommost = tuple(unline[unline[:, :, 1].argmax()][0])
            angle=degree(unline[0,0],unline[-1,0],np.array( ellipse[0]))
            startplot = unline[0]
            stopplot = unline[unline.shape[0] - 1]
            ellipses.append(
                [ellipse, leftmost, rightmost, topmost, bottommost, startplot, stopplot,angle,unline[0,0],unline[-1,0]]
            )

            # print(ellipse)
    return ellipses


def unlinesget(cnt, mean_disten):#获取圆弧、输入轮廓、轮廓点集平均距离
    gra=[]#几何基元
    unlines = []#弧线集、是弧线点集的列表
    unline = []#一条弧线、由弧线点集组成
    unlinesalone = []  # 弧线集、是弧线点集的列表
    c = False
    linelocl = 0
    for i, plot in enumerate(cnt):
        disten = np.linalg.norm(cnt[i] - cnt[i - 1])  # 计算线段长度
        if disten > mean_disten:#大于点集平均距离的两点被判断为直线
            linelocl = i
            c = True
            break
    for i in range(linelocl):
        temp = cnt[-1].copy()
        cnt[1 : cnt.shape[0]] = cnt[0 : cnt.shape[0] - 1]
        cnt[0] = temp

    if c == False:#如果轮廓中不存在线段
        gra.append(ellipsefit(cnt))
        for i, plot in enumerate(cnt):
            unline.append(cnt[i])
        unlines.append(cnt)
    if c == True:#如果轮廓中存在线段
        i = cnt.shape[0] - 1
        disten1 = np.linalg.norm(cnt[i - 1] - cnt[i])
        #gra.append([cnt[0], cnt[1]])
        while i >= 0 or disten1 < mean_disten*0.25:#遇到线段时停止分割
            disten1 = np.linalg.norm(cnt[i - 1] - cnt[i])
            if disten1 < mean_disten*0.25:
                unline.append(cnt[i])
            else:

                unline.append(cnt[i])
                unline = np.array(unline)
                unlines.append(unline)
                unlinesalone=[]
                unlinesalone=[unline]
                unline = []
                ellipse=ellipsefit(unlinesalone)
                if ellipse!=[]:
                    gra.append(ellipse)
                gra.append([cnt[i ], cnt[i-1]])

            i = i - 1

    for i, unline in enumerate(unlines):
        if len(unline) == 2:
            del unlines[i]
    unlines = np.array(unlines)
    if len(gra[0])==2 and len(gra[-1])==2:
        if ((gra[0][0]==gra[-1][0]).all()):
            del gra[-1]
    #print((gra[0][0]==gra[-1][0]))

    # print(unlines)
    return unlines,gra


def lineget(cnt, mean_disten):#获取线段、输入一个轮廓、输出线段列表、线段由起始点终止点两个点组成
    lines = []  # 初始化线段表
    for i, plot in enumerate(cnt):
        distens1 = np.linalg.norm(cnt[i] - cnt[i - 1])
        distens2 = np.linalg.norm(cnt[i - 2] - cnt[i - 3])
        # print(distens1)
        if distens1 > (mean_disten*0.25):
            lines.append(np.array([cnt[i], cnt[i - 1]]))
            if distens2 > (mean_disten):
                lines.append(np.array([cnt[i - 1], cnt[i - 2]]))
    # if np.array([cnt[i],cnt[i-1]]) in lines:

    # lines.append(np.array([cnt[i+1], cnt[i]]))
    # for i,line in enumerate(lines):
    #    a=lines[i-1][0,0]
    #    b=lines[i][1,0]
    #    if  a==b :
    #        print(lines[i-1][0].all)
    return lines


def mean_disten(cnt):  # 点集平均距离、输入轮廓、输出距离
    mean_disten = 0
    plotlast = cnt[0]
    for plot in cnt:
        mean_disten = mean_disten + np.linalg.norm(plot - plotlast)
    mean_disten = mean_disten / cnt.shape[0]

    return mean_disten


def gras(img):  # 求几何基元
    elength = 0.003  # 多边形拟合误差=周长×e
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换灰度模式
    imgx = cv2.imread("b.png")
    edges = cv2.Canny(imgray, 50, 150, apertureSize=3)
    #plt.imshow(np.flipud( np.rot90(imgx)))
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
        # plt.scatter(xa, ya)#拟合点1

        # 提取直线
        lines = lineget(approx, mean_dis)
        #print(lines)
        #画直线
        # for i, line in enumerate(lines):
        #
        #     plt.plot(
        #         [line[0, 0, 0], line[1, 0, 0]], [line[0, 0, 1], line[1, 0, 1]], "r"
        #     )
        #提取弧线
        unlines,gra = unlinesget(approx, mean_dis)
        # 画弧线
        #print(gra)
        # for i, unline in enumerate(unlines):
        #     if unline.shape[0] > 2:
        #         for j, plot in enumerate(unline):
        #             if j > 0:
        #                 plt.plot(
        #                     [unline[j][0, 0], unline[j - 1][0, 0]],
        #                     [unline[j][0, 1], unline[j - 1][0, 1]],
        #                     "r",
        #                 )
        # # 拟合椭圆
        ellipses = ellipsefit(unlines)
        #print(ellipses)
        #画椭圆
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
            # plt.plot(ell[0], ell[1], "b")
            # plt.plot(
            #     [ellipse[0][0][0], ellipse[6][0, 0]],
            #     [ellipse[0][0][1], ellipse[6][0, 1]],
            #     "r--",
            # )
            # plt.plot(
            #     [ellipse[5][0, 0], ellipse[0][0][0]],
            #     [ellipse[5][0, 1], ellipse[0][0][1]],
            #     "r--",
            # )

        # print(line)

    #plt.show()
    return gra


if __name__ == "__main__":
    img1 = cv2.imread("a.png")
    img2 = cv2.imread("b.png")
    gras1=gras(img1)#获取几何基元
    mats1=matgra(gras1)#获取形状
    gras2 = gras(img2)  # 获取几何基元
    mats2 = matgra(gras2)  # 获取形状
    x,y= match(mats1,mats2)#匹配
    gras1new, gras2new= result(x,y,gras1,gras2)#匹配后的基元列表，一一对应
    #fami(gras1new,gras2new)
    img1=np.flipud( np.rot90(img1))
    img2 = np.flipud(np.rot90(img2))
    out= drawMatches(img1,gras1new,img2,gras2new)
    plt.imshow(out)
    plt.show()
    print(gras1new,gras2new)
