import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import math
from itertools import groupby, product
from scipy.spatial.transform import Rotation as Ro


def Abs(tp1, tp2):
    return abs(tp1[0] - tp2[0]) + abs(tp1[1] - tp2[1])

def grouping(pts): 
  mtt_tups = [sorted(sb) for sb in product(pts, repeat = 2)
                                           if Abs(*sb) <= 20]
  r_dt = {el: {el} for el in pts}
  for tp1, tp2 in mtt_tups:
      r_dt[tp1] |= r_dt[tp2]
      r_dt[tp2] = r_dt[tp1]
  ne = [[*next(l_v)] for key, l_v in groupby(
          sorted(r_dt.values(), key = id), id)]
  m = []
  for i in ne: 
    m_x = np.round(np.mean(i, axis=0))
    m.append(m_x)
  m_v = sorted(m, key=lambda x: x[0])
  return m_v

def Homography(pts):
  co_ds = [(0,0), (21.6,0), (0,27.9), (21.6,27.9)]
  Z_n = np.array([[co_ds[0][0], co_ds[0][1], 1, 0, 0, 0, -pts[0][0]*co_ds[0][0], -pts[0][0]*co_ds[0][1], -pts[0][0]],
                [0, 0, 0, co_ds[0][0], co_ds[0][1], 1, -pts[0][1]*co_ds[0][0], -pts[0][1]*co_ds[0][1], -pts[0][1]],
                [co_ds[1][0], co_ds[1][1], 1, 0, 0, 0, -pts[1][0]*co_ds[1][0], -pts[1][0]*co_ds[1][1], -pts[1][0]],
                [0, 0, 0, co_ds[1][0], co_ds[1][1], 1, -pts[1][1]*co_ds[1][0], -pts[1][1]*co_ds[1][1], -pts[1][1]],
                [co_ds[2][0], co_ds[2][1], 1, 0, 0, 0, -pts[2][0]*co_ds[2][0], -pts[2][0]*co_ds[2][1], -pts[2][0]],
                [0, 0, 0, co_ds[2][0], co_ds[2][1], 1, -pts[2][1]*co_ds[2][0], -pts[2][1]*co_ds[2][1], -pts[2][1]],
                [co_ds[3][0], co_ds[3][1], 1, 0, 0, 0, -pts[3][0]*co_ds[3][0], -pts[3][0]*co_ds[3][1], -pts[3][0]],
                [0, 0, 0, co_ds[3][0], co_ds[3][1], 1, -pts[3][1]*co_ds[3][0], -pts[3][1]*co_ds[3][1], -pts[3][1]],
                [0, 0, 0, 0, 0, 0, 0, 0, 1]])
  Z_n = Z_n.astype(int)
  
  x_nn, y_nn = np.linalg.eigh(Z_n.T @ Z_n)
  v_min = y_nn[:,0]
  v_min = v_min.reshape(3,3)

  v_min = (1/v_min[2,2])*v_min

  return v_min

def Rotat_Tran(mat):

  P = np.matrix([[1382.58398,	0,	945.743164], [0,	1383.57251,	527.04834], [0, 0, 1]])
  
  P_H = np.linalg.inv(P)@ mat

  ft_lam = np.linalg.norm(P_H[:,0])
  nd_lamb = np.linalg.norm(P_H[:,1])
  mean_lamb = (ft_lam+nd_lamb)/2 

  rotat = P_H/mean_lamb
  rotat1 = rotat[:, 0]
  rotat2 = rotat[:, 1]
  rotat3 = np.cross(rotat1, rotat2, axis = 0)
  Tran = rotat[:, 2]
  R_t = np.hstack((rotat1,rotat2,rotat3))

  return R_t, Tran 


vid = cv.VideoCapture('proj2.avi')
p_ss = []
t_et = []
p_th = []
l_f = 0
h_f = 0

while(vid.isOpened()):
    ret, frame = vid.read()
    if ret == True:

        #resizing the video
        w = int(frame.shape[1] * 70 / 100)  #reducing the scale of width of the video
        h = int(frame.shape[0] * 70 / 100)  #reducing the scale of height of the video
        new_size = (w, h)                   #new sized video dimensions
        #print(new_size)
        img = cv.resize(frame,new_size)     #resizing the video
        
        gray_conversion = cv.cvtColor(img, cv.COLOR_BGR2GRAY)   #converting the video from BGR to Grayscale
        
        blured_image = cv.GaussianBlur(gray_conversion,(11,11),-1)  #blurring the image using the gaussian blur to reduce the noise and unimportant edges 
        
        detected_edges = cv.Canny(blured_image,150,200)     #Detecting the edges in the image using the canny edge detection
        #print(detected_edges)

        dilation_of_edges = cv.dilate(detected_edges,(3,3),iterations=1) #dilating the detected edges to enhance the detected edges
        #print(dilation_of_edges.shape)
        edge = np.where(dilation_of_edges != 0)
        #print(edge)
        frame_x , frame_y = dilation_of_edges.shape
        #print(frame_y)

        diagonal_d =int(np.sqrt((frame_x * frame_x) + (frame_y * frame_y)))
        #print(int(diagonal_d))
        #cv.imshow("edge",edge)

        teta = np.arange(0,181)
        #print(teta)
        Hmatrix = {}
        # Hmatrix = np.zeros((int(diagonal_d),len(teta)))
        # print(Hmatrix)
        
        Coordinate_height, Coordinate_width = edge
        #print(Coordinate_width)
        # print("***")
        # print(len(Coordinate_height))
        # print("***")
        # print(len(Coordinate_width))
        #radians = teta * ((math.pi)/180)
        # print(len(teta))
        # print(len(radians))
        for k in range (len(Coordinate_height)):
            y = Coordinate_height[k]
            x = Coordinate_width[k]
            for l in teta:
                d = int(x * np.cos(l * (math.pi / 180)) + y * np.sin(l * (math.pi / 180)))
                if (d,l) in Hmatrix:
                    Hmatrix[(d,l)] +=1
                else:
                    Hmatrix[(d,l)] = 1
        # part = np.argpartition(-Hmatrix.ravel(),40)[:40]
        # idx = np.column_stack(np.unravel_index(part, Hmatrix.shape))
        # print(idx)

        sort_arr = sorted(Hmatrix.items(), key=lambda x: x[1])[-20:]
        tehta = list([list(x[0]) for x in sort_arr])
        #sorted_matrix = {k:v for k,v in sorted(Hmatrix.items(),key = lambda item : item[1])}
        #print(tehta)
        c_rds = np.vstack(tehta)
        # for i in sort_arr:
        #     coor,rep = i
        #     coords.append(coor)
        #print(coords)
        #print(sorted_matrix)
        # l = []
        #det_edges = set()
        # for x in list(sorted_matrix)[(len(sorted_matrix) - 20):]:
        #     l.append(x)    
        #print(l)
        #k = []
        # for val in l:
        #     d,theta = val
        #     theta = theta*math.pi/180
        #     k.append((d,theta))
        # print(k)
        i = 0
        pt = []
        for m in c_rds:
            dm,tm = m
            for n in c_rds[i:]:
                dn,tn = n
                if abs(tm - tn) in range(85,95):
                    A = np.array([[np.cos(tm * (math.pi / 180)),np.sin(tm * (math.pi/180))],[np.cos(tn * (math.pi/ 180)),np.sin(tn * (math.pi/180))]])
                    D = np.array([[dm],[dn]])
                    X = np.matmul(np.linalg.inv(A),D)
                    pt.append((math.ceil(X[0]), math.ceil(X[1])))
                    #det_edges.add(pt)
                    #cv.circle(img, (math.ceil(X[0]), math.ceil(X[1])), 5, (0, 0, 255), -1)
            i += 1
        #print(det_edges)
        #coordsx_y = list(det_edges) 
        #print(coordsx_y)
        grp_vals = grouping(pt)
        
        h_f +=1
        if len(grp_vals) == 4:
            l_f += 1
            h_m = Homography(grp_vals)
            r_tt, t_ff = Rotat_Tran(h_m)
            R = Ro.from_matrix(r_tt)
            p_th.append(R.as_euler('zyx', degrees=True)[0])
            t_et.append(R.as_euler('zyx', degrees=True)[1])
            p_ss.append(R.as_euler('zyx', degrees=True)[2])
            
        # for i in clus_values:
        #     cv.circle(frame, (int(i[1]), int(i[0])), 5,  (0, 0, 255), -1)
        for el in grp_vals:
            o,p = el
            cv.circle(img,(int(o),int(p)),5,(0,0,255),-1)
        
        #print(r_tt)
        
        #print(type(clus_values))
        # plt.plot(clus_values)
        # plt.show()
        # for m in range (0,len(l)):
        #     for n in range (0,len(l)):
        #         if l[m][1] - l[n][1] or l[n][1] - l[m][1] in range(89,91):
        #             det_edges.append(l[m])
        # corners = list(set(det_edges))
        # print(corners)
        # cart_corner_coords = []
        # for a in corners:
        #     d,theta = a
        #     x = math.ceil(d * np.cos(theta * (math.pi / 180)))
        #     y = math.ceil(d * np.sin(theta * (math.pi / 180)))
        #     cart_corner_coords.append((x,y))
        # print(cart_corner_coords)

        # circle_img = frame.copy()
        # cv.circle(circle_img,cart_corner_coords[0], 5, (255,0,0), -1)
        # cv.circle(circle_img,cart_corner_coords[1], 5, (255,0,0), -1)
        # cv.circle(circle_img,cart_corner_coords[2], 5, (255,0,0), -1)
        # cv.circle(circle_img,cart_corner_coords[3], 5, (255,0,0), -1)
        
        
        #cv.imshow("frame",img)
        

        
        if cv.waitKey(1) & 0xFF == ord('q'):
          break
    else:
        break
# fr = np.arange(0, 147, 1)
# plt.plot(fr,p_th)
# plt.show()
range = np.arange(1,l_f+1,1)
fig = plt.figure()
print(len(p_th))
plt.plot(range,p_th,c="red")
plt.plot(range,t_et,c="green")
plt.plot(range,p_ss,c="blue")
plt.show()
vid.release()
cv.destroyAllWindows()
