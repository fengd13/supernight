# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 20:40:23 2018

@author: fd
"""
import math
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageEnhance
import time
print(time.clock())
MIN_MATCH_COUNT = 5
STEP=2
imglist=[]  
testname="test7"
def sharp(img):
    #img=cv.imread("final%d.png"%STEP)
    
    # 1
    blur=cv.GaussianBlur(img,(0,0),3)
    image=cv.addWeighted(img,1.5,blur,-0.5,0)
    # 2
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    image = cv.filter2D(img, -1, kernel)
    # 3
    image=cv.bilateralFilter(img,9,75,75)
    # 4
    sigma = 1; threshold = 5; amount = 1
    blurred=cv.GaussianBlur(img,(0,0),1,None,1)
    lowContrastMask = abs(img - blurred) < threshold
    sharpened = img*(1+amount) + blurred*(-amount)
    image=cv.bitwise_or(sharpened.astype(np.uint8),lowContrastMask.astype(np.uint8))
    return image
def sharp2(img):
    image = Image.fromarray(cv.cvtColor(img,cv.COLOR_BGR2RGB)) 
#    #亮度增强
#    enh_bri = ImageEnhance.Brightness(image)
#    brightness = 1.1
#    image_brightened = enh_bri.enhance(brightness)
#    #image_brightened.save('01亮度增强.png')
#    #色度增强
#    enh_col = ImageEnhance.Color(image_brightened)
#    color = 1.1
#    image_colored = enh_col.enhance(color)
#    #image_colored.save('02色度增强.png')
    #对比度增强
#    enh_con = ImageEnhance.Contrast(image)
#    contrast = 1.8
#    image_contrasted = enh_con.enhance(contrast)
    #image_contrasted.save('03对比度增强.png')
    #锐度增强
    enh_sha = ImageEnhance.Sharpness(image)
    sharpness = STEP
    image_sharped = enh_sha.enhance(sharpness)
    #image_sharped.save('04锐度增强.png')
    return cv.cvtColor(np.asarray(image_sharped),cv.COLOR_RGB2BGR) 
def contrast(img):
    image = Image.fromarray(cv.cvtColor(img,cv.COLOR_BGR2RGB)) 
    #对比度增强
    enh_con = ImageEnhance.Contrast(image)
    contrast = 1.8
    image_contrasted = enh_con.enhance(contrast)
    return cv.cvtColor(np.asarray(image_contrasted),cv.COLOR_RGB2BGR) 

cap=cv.VideoCapture(testname+'.mp4')
frame_count = 1
success = True
success, frame = cap.read()
nowimg=np.array(frame,dtype='float64')
x,y,_=nowimg.shape
while(success):
    success, frame = cap.read()
    frame=np.array(frame,dtype='float64')
    if success:
        if frame_count==1:
            sum_img= np.array(frame,dtype='float64')
        elif frame_count%STEP==0:
            #cv.imwrite(str(frame_count)+".png",nowimg/5)
            imglist.append(nowimg/nowimg.max()*255)
            nowimg=np.zeros(shape=nowimg.shape,dtype='float64')
        else:
            nowimg+=frame
    frame_count = frame_count + 1
cap.release()
l=len(imglist)

mid=int(l/2)
midimg=np.array(imglist[mid],dtype="uint8")
re=np.array(imglist[mid],dtype="float32")
for i in range(l):
    if i==mid:
        continue
    img1 = np.array(imglist[i],dtype="uint8")          # queryImage
    img2 = midimg # trainImage
    # Initiate SIFT detector
    sift = cv.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w,d = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)
    res = cv.warpPerspective(img1, M, (y,x))
    res=sharp2(res)

    #cv.imwrite("res"+str(i)+".png",res)
    re+=np.array(res,dtype='float64')
re=np.sqrt(re)
#re=np.log(1+re)
re=re*255/re.max()
#cv.imwrite(testname+"final%d.png"%STEP,re)
cv.imwrite(testname+"final%d.png"%STEP,contrast(np.array(re,dtype="uint8")))
    
#def hisEqulColor(img):
#    ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCR_CB)
#    channels = cv.split(ycrcb)
#    #print len(channels)
#    cv.equalizeHist(channels[0], channels[0])
#    cv.merge(channels, ycrcb)
#    cv.cvtColor(ycrcb, cv.COLOR_YCR_CB2BGR, img)
#    return img
#
#im = cv.imread("final%d.png"%STEP)
#
#eq = hisEqulColor(im)
#cv.imshow('image2',eq )
#cv.imwrite('lena2.png',eq)

print(time.clock())
