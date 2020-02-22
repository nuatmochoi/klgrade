import cv2
import numpy as np
from matplotlib import pyplot as plt


img=cv2.imread('triangle.jpeg',cv2.IMREAD_GRAYSCALE)
img2=cv2.imread('triangle2.jpeg',cv2.IMREAD_GRAYSCALE)
img3=cv2.imread('triangle3.jpeg',cv2.IMREAD_GRAYSCALE)
img4=cv2.imread('triangle4.jpeg',cv2.IMREAD_GRAYSCALE)
img_list=[img,img2,img3,img4]

def which_more(image):
    sum1 = 0
    sum2 = 0
    h,w=image.shape

    for j in range(w):
        for i in range(0,int(h/2)+1):
            sum1+=image[i-1][j-1]
        for i in range(int(h / 2), h + 1):
            sum2 += image[i - 1][j - 1]
    if abs(sum1-sum2)<200000: # 위아래 차이가 거의 나지 않을 때
        sum1 = 0
        sum2 = 0
        for j in range(h):
            for i in range(0, int(w / 2) + 1):
                sum1 += image[j - 1][i - 1]
            for i in range(int(w / 2), w + 1):
                sum2 += image[j - 1][i - 1]
            if sum1>sum2: big=90
            else: big=270
    else:
        if sum1>sum2: big=180
        else: big=0
    return big


def Rotate(image,degree):
    h,w =image.shape
    if degree==90:
        M=cv2.getRotationMatrix2D((w/2,h/2),90,1)
    elif degree==180:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 180, 1)
    elif degree==270:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 270, 1)
    dst = cv2.warpAffine(image, M, (w, h))
    return image


plt.imshow(Rotate(img_list[1],which_more(img_list[1])))
plt.show()