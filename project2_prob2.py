import cv2
import numpy as np
from numpy import linalg
import random

def coords(im1,im2,w,rz_ck):
    l=[]
    if rz_ck==True:
        im1=cv2.resize(im1,(int(im1.shape[1]*20/100),int(im1.shape[0]*20/100)))
    gray_img1=cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    s=cv2.SIFT_create()
    kp1,des1=s.detectAndCompute(gray_img1,None)
    if rz_ck==True:
        im2=cv2.resize(im2,(int(im2.shape[1]*20/100),int(im2.shape[0]*20/100)))
    gray_img2=cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
    kp2,des2=s.detectAndCompute(gray_img2,None)
    bf=cv2.BFMatcher()
    match=bf.knnMatch(des1,des2,k=2)
    good=[]
    good_draw=[]
    for m,n in match:
        if m.distance<w*n.distance:
            good.append(m)
            good_draw.append([m])
    for j in good:
        (pts_1_x,pts_1_y)=kp1[j.queryIdx].pt 
        (pts_2_x,pts_2_y)=kp2[j.trainIdx].pt
        l.append([pts_1_x,pts_1_y,pts_2_x,pts_2_y])
    pts=np.array(l)
    
    return pts,im1,im2


def Homography(pos):
    Lst=[[]]
    h_t=[]
    for i in range(pos.shape[0]):
        lst_1=[pos[i][0],pos[i][1],1,0,0,0,-pos[i][2]*pos[i][0],-pos[i][2]*pos[i][1],-pos[i][2]]
        lst_2=[0,0,0,pos[i][0],pos[i][1],1,-pos[i][3]*pos[i][0],-pos[i][3]*pos[i][1],-pos[i][3]]
        Lst.append(lst_1)
        Lst.append(lst_2)
    Lst.pop(0)
    Lst=np.array(Lst)
    Lst_T=Lst.T
    Lst_TLst=np.matmul(Lst_T,Lst)
    print(Lst_TLst)
    x,y=linalg.eig(Lst_TLst)
    for i in range(9):
        if(min(x)==x[i]):
            h_t=y[:,i]
    h_t=np.reshape(h_t,(3,3))
    h_t=(1/h_t[2][2])*h_t
    return h_t


def distance(pts,H):
    print(pts)
    p_1=np.array([pts[0],pts[1],1])
    p_1=p_1.T
    p_2_approx=np.dot(H,p_1)
    p_2_approx=(1/p_2_approx[2])*p_2_approx
    p_2=np.array([pts[0],pts[1],1]).T
    error=p_2-p_2_approx
    return linalg.norm(error)

def ransac(pts,t):
    maxinl=[]
    H_fin=None
    for i in range(500):
        p1=pts[random.randrange(0,len(pts))]
        p2=pts[random.randrange(0,len(pts))]
        four_pts=np.vstack((p1,p2))
        p3=pts[random.randrange(0,len(pts))]
        four_pts=np.vstack((four_pts,p3))
        p4=pts[random.randrange(0,len(pts))]
        four_pts=np.vstack((four_pts,p4))
    H=Homography(four_pts)
    inl=[]
    for i in range(len(pts)):
        d=distance(pts[i],H)
        if d<t:
            inl.append(pts[i])
    if len(inl)>len(maxinl):
        maxinl=inl
        H_fin=H
    return H_fin

def crop(img):
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    thresh=cv2.threshold(img_gray,0,255,cv2.THRESH_BINARY)[1]
    white=np.transpose(np.nonzero(thresh))
    x_min=0
    y_min=0
    y_max=0
    x_max=0
    for i in range(white.shape[0]):
        if white[i][0]==0 and white[i][1]>y_max:
            y_max=white[i][1]
    for i in range(white.shape[0]):
        if white[i][1]==y_max and white[i][0]>x_max:
            x_max=white[i][0]
    img=img[y_min:y_max,x_min:x_max]
    return img
    

Image_1=cv2.imread('image_1.jpg')
Image_2=cv2.imread('image_2.jpg')
pt_s_1,Image_1_rz,Image_2_rz=coords(Image_1,Image_2,0.5,True)
homo1=Homography(pt_s_1)
det1=cv2.warpPerspective(Image_2_rz,linalg.inv(homo1),(Image_1_rz.shape[1]+Image_2_rz.shape[1],Image_1_rz.shape[0]+Image_2_rz.shape[0]))
det1[0:Image_1_rz.shape[0], 0:Image_1_rz.shape[1]] = Image_1_rz
det1=crop(det1)
det1=crop(det1)

Image_3=cv2.imread('image_3.jpg')
Image_4=cv2.imread('image_4.jpg')
pt_s_2,Image_3_rz,Image_4_rz=coords(Image_3,Image_4,0.7,True)
homo2=Homography(pt_s_2)
det2=cv2.warpPerspective(Image_4_rz,linalg.inv(homo2),(Image_3_rz.shape[1]+Image_4_rz.shape[1],Image_3_rz.shape[0]+Image_4_rz.shape[0]))
det2[0:Image_3_rz.shape[0],0:Image_3_rz.shape[1]]=Image_3_rz

pt_s_3,Image_5_rz,Image_6_rz=coords(det1,det2,0.52,False)
homo3=Homography(pt_s_3)
det3=cv2.warpPerspective(Image_6_rz,linalg.inv(homo3),(Image_5_rz.shape[1]+Image_6_rz.shape[1],Image_5_rz.shape[0]+Image_6_rz.shape[0]))
det3[0:Image_5_rz.shape[0],0:Image_5_rz.shape[1]]=Image_5_rz
det3=cv2.resize(det3,(int(det3.shape[1]*60/100),int(det3.shape[0]*60/100)))
cv2.imshow("combined",det3)
cv2.waitKey(0)