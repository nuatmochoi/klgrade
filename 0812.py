# upside bone's bottommost & down side bone's avg height(leftmost[1] and topmost[1])

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
import csv


def extreme_point_top(img_top):
    Contours, _ = cv2.findContours(image=copy.deepcopy(img_top),
                                              mode=cv2.RETR_TREE,
                                              method=cv2.CHAIN_APPROX_NONE)
    MaxContour = max(Contours, key=cv2.contourArea) if Contours else None
    c0 = MaxContour
    x_c, y_c, w_c, h_c = cv2.boundingRect(c0)
    top_w = w_c
    if MaxContour is not None:
        bottommost = tuple(c0[c0[:, :, 1].argmax()][0])
        print(bottommost)
    else:
        bottommost=(100,170)
    return bottommost, top_w


def extreme_point_bottom(img_bottom):
    Contours_b, _ = cv2.findContours(image=copy.deepcopy(img_bottom),
                                                mode=cv2.RETR_TREE,
                                                method=cv2.CHAIN_APPROX_NONE)
    MaxContour_b = max(Contours_b, key=cv2.contourArea) if Contours_b else None
    c_b = MaxContour_b
    x_c, y_c, w_c, h_c = cv2.boundingRect(c_b)
    bot_w = w_c
    if MaxContour_b is not None:
        leftmost = tuple(c_b[c_b[:, :, 0].argmin()][0])
        topmost = tuple(c_b[c_b[:, :, 1].argmin()][0])
    else:
        leftmost=(0,0)
        topmost=(0,0)
    return topmost, leftmost, bot_w


def process(path, csv_writer):
    li = os.listdir(path)
    li.sort()
    sum_ratio = 0
    sum_ratio_r = 0
    #flag = 0
    #flag_r = 0

    for file in li:
        print(file)
        img = cv2.imread(path + file, cv2.IMREAD_GRAYSCALE)
        Srcimg = img.copy()
        rec=Srcimg.copy()
        Srcimg = cv2.bilateralFilter(Srcimg, 9, 75, 75)
        Srcimg = cv2.GaussianBlur(Srcimg, (3, 3), 0)

        # clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        # Srcimg=clahe.apply(Srcimg)

        h, w = Srcimg.shape
        center3 = img[int(h / 2) - 10:int(h / 2) + 40,
                  int(w / 2) - 40:int(w / 2) + 40]

        # 뼈부분 평균 픽셀
        sum_bone = 0
        bone = Srcimg[int(h / 4):int(h / 4) + 20, int(w / 2) - 10:int(w / 2) + 10]
        bone2=Srcimg[int(3*h/4)-20:int(3*h/4),int(w/2)-10:int(w/2)+10]
        cv2.rectangle(rec,(int(w/2)-40,int(h/2)-10),(int(w/2)+40,int(h/2)+40),(255,255,255),1)
        cv2.rectangle(rec,(int(w/2)-10,int(h/4)),(int(w/2)+10,int(h/4)+20),(255,0,0),1)
        cv2.rectangle(rec,(int(w/2)-10,int(3*h/4)-20),(int(w/2)+10,int(3*h/4)),(255,0,0),1)

        for i in range(bone.shape[0]):
            for j in range(bone.shape[1]):
                sum_bone += bone[i][j]
                sum_bone += bone2[i][j]
        avg_bone = sum_bone / (bone.shape[0] * bone.shape[1])*2
        print(avg_bone)

        # 가운데 부분 평균 픽셀
        sum_cen = 0
        center_area=center3.shape[0] * center3.shape[1]
        for i in range(center3.shape[0]):
            for j in range(center3.shape[1]):
                sum_cen += center3[i][j]
        avg = int(sum_cen / center_area)

        avg_init = avg
        turn_flag = 0
        while avg_bone < avg:  # 뼈와 거리가 있는 적정값 찾아가도록
            avg += 10
            if avg > 256:
                avg = avg_init
                turn_flag = 1
            if turn_flag == 1:
                avg -= 10
            if avg < 10:
                break

        sum_lt=0; sum_lb=0; sum_rt=0; sum_rb=0
        mid_line=int(h/2)+22
        img_lt=Srcimg[0:mid_line,0:10]
        img_lb=Srcimg[mid_line:h,0:10]
        img_rt=Srcimg[0:mid_line,w-10:w]
        img_rb=Srcimg[mid_line:h,w-10:w]
        small_h,small_w=img_lt.shape
        small_area=small_w*small_h
        for i in range(small_h):
            for j in range(small_w):
                sum_lt+=img_lt[i][j]
                sum_rt+=img_rt[i][j]
        for i in range(img_lb.shape[0]):
            for j in range(img_lb.shape[1]):
                sum_lb += img_lb[i][j]
                sum_rb += img_rb[i][j]

        avg_lt=int(sum_lt/small_area)
        avg_lb=int(sum_lb/small_area)
        avg_rt=int(sum_rt/small_area)
        avg_rb=int(sum_rb/small_area)
        print(avg_lt, avg_lb, avg_rt, avg_rb, avg)


        # left side cropping
        img_top_ori_l = Srcimg[0:int(h / 2) + 22, 0:int(w / 2) - 50]
        _, img_top_l = cv2.threshold(img_top_ori_l, avg-5, 256, cv2.THRESH_BINARY)
        cv2.imwrite("0812/lt/"+file,img_top_l)
        cv2.rectangle(rec,(0,0),(int(w/2)-50,int(h/2)+22),(0,0,255),1)
        img_bottom_ori_l = Srcimg[int(h / 2) + 22:h, 0:int(w / 2) - 50]
        cv2.rectangle(rec,(0,int(h/2)+22),(int(w/2)-50,h),(0,0,255),1)
        _, img_bottom_l = cv2.threshold(img_bottom_ori_l, avg-5, 256, cv2.THRESH_BINARY)
        cv2.imwrite("0812/lb/"+file, img_bottom_l)

        bottommost, top_w = extreme_point_top(img_top_l)
        topmost, leftmost, bot_w = extreme_point_bottom(img_bottom_l)

        if bottommost[1]!=170:
            dist_l=170-bottommost[1]
        else:
            dist_l=0

        dist_bot_l=(leftmost[1] + topmost[1])/2

        avg_bot_h = dist_bot_l#+dist_l
        longer_w = top_w if top_w > bot_w else bot_w
        if dist_bot_l != 0:
            ratio = avg_bot_h / longer_w
        else:
            ratio = 0

        if ratio >= 1:
            ratio = 0
            sum_ratio += ratio
        else:
            sum_ratio += ratio

        # right side cropping
        img_top_ori_r = Srcimg[0:int(h / 2) + 22, int(w / 2) + 50:w]
        _, img_top_r = cv2.threshold(img_top_ori_r, avg-5, 256, cv2.THRESH_BINARY)
        cv2.imwrite("0812/rt/"+file, img_top_r)
        cv2.rectangle(rec, (int(w/2)+50,0), (w, int(h /2)+22), (0, 0, 255), 1)
        img_bottom_ori_r = Srcimg[int(h / 2) + 22:h, int(w / 2) + 50:w]
        _, img_bottom_r = cv2.threshold(img_bottom_ori_r, avg-5, 256, cv2.THRESH_BINARY)
        cv2.imwrite("0812/rb/"+file, img_bottom_r)

        cv2.rectangle(rec, (int(w / 2) + 50, int(h/2)+22), (w, h), (0, 0, 255), 1)

        #plt.imshow(rec,cmap="bone")
        #plt.title(file)
        #plt.show()

        bottommost_r, top_w_r = extreme_point_top(img_top_r)
        topmost_r, leftmost_r, bot_w_r = extreme_point_bottom(img_bottom_r)

        dist_bot_r = (leftmost_r[1] + topmost_r[1]) / 2
        if bottommost_r[1]!=170:
            dist_r=170-bottommost_r[1]
        else:
            dist_r=0
        avg_bot_h_r = dist_bot_r#+dist_r
        longer_w_r = top_w_r if top_w_r > bot_w_r else bot_w_r
        if dist_bot_l != 0:
            ratio_r = avg_bot_h_r / longer_w_r
        else:
            ratio_r = 0

        if ratio_r >= 1:
            ratio_r = 0

        if ratio==0 or ratio_r==0:
            ratio_whole=0
        else:
            ratio_whole = (ratio + ratio_r) / 2
        sum_ratio_r += ratio_whole

        csv_writer.writerow([file, ratio, ratio_r])

    avg_ratio = sum_ratio / len(li)

    csv_writer.writerow(['\n'])
    csv_writer.writerow(['(Average ratio=)', avg_ratio])
    csv_writer.writerow(['\n'])


with open("result.csv", 'w', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['[FILENAME]', '[LEFT RATIO]', '[RIGHT RATIO]'])
    for i in range(5):

        csv_writer.writerow(['[KL'+str(i)+']'])
        path = 'kl'+str(i)+'/'
        process(path, csv_writer)

