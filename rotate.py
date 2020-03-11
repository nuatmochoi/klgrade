import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy
import os
from skimage.transform import match_histograms
import scipy.ndimage


#최대 넓이를 가진 contour 추출
def getMaxContour(contours):
    MaxArea = 0
    Location = 0
    for idx in range(0, len(contours)):
      Area = cv2.contourArea(contours[idx])
      if Area > MaxArea:
          MaxArea = Area
          Location = idx
    MaxContour = np.array(contours[Location])
    return MaxContour, MaxArea

#By Using Extreme point, Set angle
def condition(prev_angle,c0,x,y,set_flag):
    box=cv2.boundingRect(c0)
    x_c=box[0]; y_c=box[1];w_c=box[2];h_c=box[3]
    M = cv2.moments(c0)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    leftmost = tuple(c0[c0[:, :, 0].argmin()][0])
    rightmost = tuple(c0[c0[:, :, 0].argmax()][0])
    topmost = tuple(c0[c0[:, :, 1].argmin()][0])
    bottommost = tuple(c0[c0[:, :, 1].argmax()][0])
    angle=None


    if bottommost[1] > y * (14 / 15) and not (topmost[1] < 10) \
            and not (leftmost[0] < 10) and not (rightmost[0] > x * (29 / 30)):  # 바닥에 붙어있음
        angle = 0
        print("attach bottom")

    elif topmost[1] < 10 and not (bottommost[1] > y * (29 / 30)) \
            and not (rightmost[0] > x * (29 / 30)) and not (leftmost[0] < 10):  # 위에 붙어있음
        angle = 180
        print("attach top")

    if (bottommost[0] == leftmost[0] and leftmost[1] == bottommost[1]):  # use frame
        angle = 0
        print("include dummy")
    """
    if bottommost[1]>y*(14/15) and rightmost[0]>x*(14/15) and abs(bottommost[1]-rightmost[1])<int(y/3)\
           and abs(bottommost[0]-rightmost[0])<int(w_c/2) and not topmost[0]>x*(29/30) and leftmost[0]<x*(1/3) :  #5시->11시
        if (topmost[0] < cx): #엄지가 밑으로 for flip
            angle=-40
        else:
            angle=-40
        print("little tilt minus")
    if bottommost[1]>y*(14/15) and leftmost[0]<x*(1/15) and abs(bottommost[1]-leftmost[1])<int(y/3)\
           and abs(bottommost[0]-leftmost[0])<int(w_c/2) and not topmost[0]<x*(1/30) and rightmost[0]>x*(2/3):  #7시->1시
        if (topmost[0] < cx): #엄지가 위로 for flip
            angle=40
        else:
            angle=40
        print("little tilt plus")
    if topmost[1]<y*(1/15) and leftmost[0]<x*(1/15) and abs(topmost[1]-leftmost[1])<int(y/3)\
           and abs(topmost[0]-leftmost[0])<int(w_c/2) and not bottommost[0]<x*(1/30): #and rightmost[0]>x*(2/3):  #11시->5시
        if (bottommost[0] < cx):
            angle=130
        else:
            angle=130
        print("little tilt reverse")

    if topmost[1]<y*(1/15) and leftmost[0]>x*(14/15) and abs(topmost[1]-leftmost[1])<int(y/3)\
           and abs(topmost[0]-leftmost[0])<int(w_c/2) and not bottommost[0]>x*(29/30): #and rightmost[0]>x*(2/3):  #1시->7시
        if (bottommost[0] < cx):
            angle=-130
        else:
            angle=-130
        print("little tilt reverse minus")
    """
    if angle!=None:
        set_flag=1
        return angle,set_flag

    if h_c>=w_c:
        if abs(leftmost[1] - rightmost[1]) < 150:
            if cy > rightmost[1] and cy > leftmost[1] and not(abs(rightmost[1]-bottommost[1])<10):
                angle = 0
                print("right,left > center")
            elif cy < rightmost[1] and cy < leftmost[1] and not(abs(rightmost[1]-bottommost[1])<10):
                angle = 180
                print("right,left <center")
        if angle!=None:
            set_flag=1
            return angle,set_flag

    elif w_c>h_c:

        if abs(topmost[0] - bottommost[0] < 150):
            if cx < topmost[0] and cx < bottommost[0]:
                angle = 90
                print("top,bottom>center")
            elif cx > topmost[0] and cx > bottommost[0]:
                angle = -90
                print("top,bottom<center")

        else:
            if rightmost[0] > x * (29 / 30) and not (bottommost[0]>x*(29/30)) and not (topmost[0]<x*(29/30)) :
                angle = -90
                print("attach right")
            elif leftmost[0] < x * (1 / 30) and not (bottommost[0]<x*(1/30)) and not (topmost[0]<x*(1/30)):
                angle = 90
                print("attach left")

            elif cy > rightmost[1] and cy > leftmost[1]:
                angle = 0
                print("right,left < center2")
            elif cy < rightmost[1] and cy < leftmost[1]:
                angle = 180
                print("right,left >center2")
        if angle != None:
            set_flag=1
            return angle,set_flag
        else:
            return prev_angle,set_flag


def Process(Src,count,set_flag):
    r=10
    h, w = Src.shape
    tl = Src[0:int(round(h / r)), 0:int(round(w / r))]
    tr = Src[0:int(round(h / r)), int(round((r - 1) * w / r)):w]
    bl = Src[int(round((r - 1) * h / r)):h, 0:int(round(w / r))]
    br = Src[int(round((r - 1) * h / r)):h, int(round((r - 1) * w / r)):w]
    center=Src[int((2/5)*h):int((3/5)*h),int((2/5)*w):int((3/5)*w)]
    sum_cen=0
    for i in range(center.shape[0]):
        for j in range(center.shape[1]):
            sum_cen+=center[i][j]
    avg_cen=sum_cen/(center.shape[0]*center.shape[1])
    sum=0
    for i in range(tl.shape[0]):
        for j in range(tl.shape[1]):
            sum += tl[i][j]
    for i in range(tr.shape[0]):
        for j in range(tr.shape[1]):
            sum += tr[i][j]
    for i in range(bl.shape[0]):
        for j in range(bl.shape[1]):
            sum += bl[i][j]
    for i in range(br.shape[0]):
        for j in range(br.shape[1]):
            sum += br[i][j]
    small_h, small_w = tl.shape
    small_area = small_h * small_w
    avg_pixel = sum / (small_area * 4)


    #IMAGE PREPROCESSING
    hist1=cv2.calcHist([Src],[0],None,[256],[0,256])

    SrcImg=Src[int(h/10):int(9*h/10),int(w/10):int(9*w/10)]
    normalizedImg = np.zeros((800, 800))
    SrcImg = cv2.normalize(SrcImg, normalizedImg, 0, 255, cv2.NORM_MINMAX)
    y,x=SrcImg.shape
    try:
        reference = cv2.imread("kah_add_390000.jpg",cv2.IMREAD_GRAYSCALE)
        reference = cv2.resize(reference,(y,x),interpolation=cv2.INTER_AREA)
    except Exception as e:
        print(str(e))
    hist2=cv2.calcHist([reference],[0],None,[256],[0,256])
    compare=cv2.compareHist(hist1,hist2,cv2.HISTCMP_CORREL)
    print("compare hist=",compare)
    if compare <-0.13:
        SrcImg = match_histograms(SrcImg, reference, multichannel=False).astype('uint8')
        SrcImg = cv2.bitwise_not(SrcImg)

    SrcImg = match_histograms(SrcImg, reference, multichannel=False).astype('uint8')

    mask = np.zeros((y + 2, x + 2), np.uint8)
    cv2.floodFill(SrcImg, mask, (int(x / 2), int(y / 2)), 255, flags=(4 | 255 << 8))
    if count>=1:
        if avg_pixel>avg_cen:
            _, BinImg = cv2.threshold(SrcImg,avg_pixel,255, cv2.THRESH_BINARY_INV)
        else:
            _, BinImg = cv2.threshold(SrcImg, 0, 255, cv2.THRESH_OTSU)
    else:
        _, BinImg = cv2.threshold(SrcImg, 0, 255, cv2.THRESH_OTSU)

    #BinImg=cv2.bitwise_and(SrcImg,SrcImg,mask=BinImg)

    kernel=np.ones((3,3),np.uint8)

    BinImg4box=cv2.dilate(BinImg,kernel,iterations=50)
    BinImg = cv2.dilate(BinImg, kernel, iterations=11)

    if count==0:
        _,Contours4box, Hierarchybox = cv2.findContours(image=copy.deepcopy(BinImg4box),
                                               mode=cv2.RETR_TREE,
                                               method=cv2.CHAIN_APPROX_NONE)
        if Contours4box!=None:
            MaxContour4box, _ = getMaxContour(Contours4box)
            c4box = MaxContour4box
            _, _, angle4box = cv2.fitEllipse(MaxContour4box)

            x_4b, y_4b, w_4b, h_4b = cv2.boundingRect(c4box)

    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(cv2.erode(BinImg,kernel,iterations=20),connectivity=8,ltype=cv2.CV_32S)

    width_start=None

    max_c = 0
    for idx in range(nlabels):
        if not idx == 0:
            if max_c < stats[idx][4]:
                max_c = stats[idx][4]

    if count==1:
        for idx in range(nlabels):
            if stats[idx][4]==max_c:
                if stats[idx][1] < 10 and stats[idx][3]-stats[idx][1]<stats[0][4]-5 :
                    width_start=stats[idx][0]
                    width_end=stats[idx][2]
    if width_start!=None:
        left_w=int(x/2)-width_start
        right_w=width_end-width_start-left_w

    imgf=scipy.ndimage.gaussian_filter(cv2.erode(BinImg,kernel,iterations=60),16)
    labels, num = scipy.ndimage.label(imgf>25)


    #MAKING CONTOUR
    _,Contours, Hierarchy = cv2.findContours(image=copy.deepcopy(BinImg),
                                         mode=cv2.RETR_TREE,
                                         method=cv2.CHAIN_APPROX_NONE)
    img_raw=np.ones(SrcImg.shape)-SrcImg

    MaxContour, _ = getMaxContour(Contours)

    Canvas = np.ones(SrcImg.shape, np.uint8)
    image=cv2.drawContours(image=Canvas, contours=[MaxContour], contourIdx=0, color=(255), thickness=-1)
    #hull=cv2.convexHull(MaxContour)

    mask = (Canvas != 255)
    RoiImg = copy.deepcopy(BinImg)
    RoiImg[mask] = 255
    RoiImg = cv2.morphologyEx(src=RoiImg, op=cv2.MORPH_CLOSE, kernel=np.ones((3,3)), iterations=4)

    c0=MaxContour
    epsilon = 0.01 * cv2.arcLength(c0, True)
    c0 = cv2.approxPolyDP(c0, epsilon, True)
    M=cv2.moments(MaxContour)

    x_c,y_c,w_c,h_c = cv2.boundingRect(c0)

    ellipse=cv2.fitEllipse(MaxContour)
    _,_,angle=cv2.fitEllipse(MaxContour)

    img_e=cv2.ellipse(RoiImg,ellipse,10)

    print("w",w_c,"h",h_c)
    ratio=float(h_c/w_c)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    print("crop_x=",x,",crop_y=",y)
    leftmost = tuple(c0[c0[:, :, 0].argmin()][0])
    rightmost = tuple(c0[c0[:, :, 0].argmax()][0])
    topmost = tuple(c0[c0[:, :, 1].argmin()][0])
    bottommost = tuple(c0[c0[:, :, 1].argmax()][0])
    """
    plt.subplot(1,2,1)
    plt.imshow(img_raw, cmap='bone')
    plt.title("center of contour")
    plt.axis('off')
    plt.scatter([cx], [cy], c="r", s=30)

    plt.subplot(1,2,2)
    plt.imshow(image, cmap='bone')
    plt.plot(
        [x_c, x_c + w_c, x_c + w_c, x_c, x_c],
        [y_c, y_c, y_c + h_c, y_c + h_c, y_c],
        c="r"
    )
    plt.axis("off")
    plt.scatter(
        [leftmost[0], rightmost[0], topmost[0], bottommost[0]],
        [leftmost[1], rightmost[1], topmost[1], bottommost[1]],
        c="b", s=30)
    plt.title("Extream Points")
    """
    print("\ncenter: (",str(cx),",",str(cy),")")
    print("leftmost:",leftmost)
    print("rightmost:",rightmost)
    print("topmost:",topmost)
    print("bottommost:",bottommost)
    print()

    plt.show()


    if count==0:
        Src = Src[int(y_4b - y_4b * (1 / 2)):int(y_4b + h_4b + (y_4b + h_4b) * (1 / 2)),
             int(x_4b - x_4b * (1 / 2)):int(x_4b + w_4b + (1 / 2) * (x_4b + w_4b))]
        #Src = Src[int(y_c - y_c * (1 / 2)):int(y_c + h_c + (y_c + h_c) * (1 / 2)),
         #     int(x_c - x_c * (1 / 2)):int(x_c + w_c + (1 / 2) * (x_c + w_c))]

    #ROTATION PART
    if condition(angle,c0,x,y,set_flag)!=None:
        angle,set_flag=condition(angle,c0,x,y,set_flag)

    print("ellipse angle=",angle)
    if angle>135 and not (angle==180):
        M1 = cv2.getRotationMatrix2D((w / 2, h / 2), 180-angle, 1)
    else:
        M1 = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    img_end = cv2.warpAffine(Src, M1, (w, h))
    #M2=cv2.getRotationMatrix2D((w/2,h/2),180,1)
    #if width_start!=None:
     #   if(left_w>right_w):
      #      Src=cv2.warpAffine(Src,M2,(w,h))

    return img_end, set_flag

li=os.listdir("0724_incomplete/img_test/")
li.sort()
path = '/home/j/python_work/rota/sample'
i=0
for file in li:
    Src=None
    count=0
    set_flag=0
    print(file)
    Src=cv2.imread("0724_incomplete/img_test/"+file, cv2.IMREAD_GRAYSCALE)
    img_end=Process(Src,count,set_flag)[0]
    """
    while(set_flag == 0):
        count+=1
        if Process(Src,count,set_flag)[1]==0:
            Src=Process(Src,count,set_flag)[0]
        if(count==5):
            break
    """
    #img_end=Src
    count+=1
    if Process(Src,count,set_flag)[1]==0:
       img_end=Process(img_end,count,set_flag)[0]

    cv2.imwrite('0805_2/'+file,img_end)

    #plt.title("Rotated Image")
    #plt.imshow(img_end)
    #plt.show()