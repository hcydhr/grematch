import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from skimage import measure,draw
from scipy.optimize import curve_fit,leastsq

def ellipsecurve(x,y, a, b, c,d,e,f):#椭圆方程

 return a*x*x+b*x*y+c*y*y+d*x+e*y+f

def ellipsefit(unlines):
    ellipses=[]
    for unline in unlines:
        if len(unline)>6:
            #x=unline[:,0,0]
            #y = unline[:, 0, 1]
            ellipse = cv2.fitEllipse(unline)
            ellipses.append(ellipse)
            print(ellipse)
    return ellipses



def unlinesget(cnt,mean_disten):
    unlines=[]
    unline=[]
    c=False
    linelocl=0
    for i,plot in enumerate(cnt):
        disten = np.linalg.norm(cnt[i]-cnt[i-1])  # 计算线段长度
        if disten>mean_disten:
            linelocl=i
            c=True
            break
    for i in range(linelocl):
        temp=cnt[-1].copy()
        cnt[1:cnt.shape[0]] = cnt[0:cnt.shape[0]-1]
        cnt[0] = temp


    if c==False:
        for i, plot in enumerate(cnt):
            unline.append(cnt[i])
        unlines.append(cnt)
    if c==True:
        i=cnt.shape[0]-1
        disten1=np.linalg.norm(cnt[i-1]-cnt[i])
        while (i>0 or disten1<mean_disten):
            disten1 = np.linalg.norm(cnt[i - 1] - cnt[i])
            if disten1<mean_disten:
                unline.append(cnt[i])
            else:
                unline.append(cnt[i])
                unlines.append(unline)
                unline=[]
            i=i-1

    for i,unline in enumerate(unlines):
        if len(unline)==2:
            del unlines[i]









    #print(unlines)
    return unlines




def lineget(cnt,mean_disten):
    lines=[]#初始化线段表
    for i,plot in enumerate(cnt):
        distens1=np.linalg.norm(cnt[i]-cnt[i-1])
        distens2=np.linalg.norm(cnt[i-2]-cnt[i-3])
        print(distens1)
        if distens1>(mean_disten):
            lines.append(np.array([cnt[i], cnt[i - 1]]))
            if distens2>(mean_disten):
                lines.append(np.array([cnt[i-1], cnt[i - 2]]))
    #if np.array([cnt[i],cnt[i-1]]) in lines:
    #if cnt[i] in lines:
    #lines.append(np.array([cnt[i+1], cnt[i]]))
    #for i,line in enumerate(lines):
    #    a=lines[i-1][0,0]
    #    b=lines[i][1,0]
    #    if  a==b :
    #        print(lines[i-1][0].all)
    return lines




def mean_disten(cnt):#点集平均距离
    mean_disten=0
    plotlast=cnt[0]
    for plot in cnt:
        mean_disten=mean_disten+ np.linalg.norm(plot-plotlast)
    mean_disten=mean_disten/cnt.shape[0]

    return mean_disten


def gra(img):#求几何基元
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换灰度模式
    edges = cv2.Canny(imgray, 50, 150, apertureSize=3)
    blur = cv2.GaussianBlur(imgray, (5, 5), 0)
    ret,thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓搜索不连续
    contours = measure.find_contours(thresh, 0.5)

    for i,cnt in enumerate(contours):
        mean_dis = mean_disten(cnt)
        #plt.plot( [0, mean_dis],[i, i])#平均长度图示
        #print(cnt.shape[0])
        cnt=cnt.reshape((cnt.shape[0],1,2))
        cnt=np.int32(cnt)
        epsilon = 0.008 * cv2.arcLength(cnt, True)  # 多边形拟合偏差
        approx = cv2.approxPolyDP(cnt, epsilon, True)  # 多边形拟合
        x=cnt[:,0,0]
        y=cnt[:,0,1]
        #plt.scatter(x, y)#轮廓点
        #print(approx.shape)
        xa = approx[:,0,0]  #拟合后
        ya = approx[:,0,1]
        #plt.scatter(xa, ya)#拟合点


        #提取直线
        lines=lineget(approx,mean_dis)
        #画直线
        #for i,line in enumerate(lines):
        #    plt.plot([line[0,0,0],line[1,0,0]],[line[0,0,1],line[1,0,1]])
        #提取弧线
        unlines=unlinesget(approx,mean_dis)
        #画弧线
        for i, unline in enumerate(unlines):
            for j, plot in enumerate(unline):
                plt.plot([unline[j][0,0],unline[j-1][0,0]],[unline[j][0,1],unline[j-1][0,1]])
        #拟合椭圆
        ellipses=ellipsefit(unlines)
        #画椭圆
        for ellipse in ellipses:
            ox=ellipse[0][0]
            oy=ellipse[0][1]
            print(ellipse[0])
            plt.plot([ox,ox+10], [oy,oy+10])







        #print(line)

    plt.show()





if __name__ == '__main__':
    img1 = cv2.imread("p.png")
    img2 = cv2.imread("p1.png")
    gra(img1)


