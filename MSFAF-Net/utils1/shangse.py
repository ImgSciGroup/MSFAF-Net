import cv2
import numpy as np


img_gt = cv2.imread(r"D:\ImageProcess\11_MSFF2Net\code\data\whu\res\change_label.tif", 0)
img_result = cv2.imread(r"D:\ImageProcess\11_MSFF2Net\code\data\whu\res\conc_res.png", 0)
height, width = img_gt.shape
result = np.zeros((height, width, 3))

UC=0
CC=0
MD=0
FD=0
color = open('D:\\ImageProcess\\11_MSFF2Net\\code\\data\\whu\\res\\color_conc.txt', 'w')
for i in range(height):
    for j in range(width):
        result[i, j, :] = img_result[i, j]
        if(img_result[i][j]==0) :
            UC = UC+1
        if(img_result[i][j]==255):
            CC = CC+1
        if img_gt[i][j] == 0 and img_result[i][j] != 0:    # FP
            result[i][j][0] = 222
            result[i][j][1] = 162
            result[i][j][2] = 0
            MD = MD+1
        if img_gt[i][j].all() != img_result[i][j].all() and img_gt[i][j].all() == 1:    # FN
            result[i][j][0] = 0
            result[i][j][1] = 0
            result[i][j][2] = 255
            FD = FD+1
cv2.imshow("result", result)
color.write('Unchanged_true_num='+str(UC)+'\n'
            'changed_true_num='+str(CC)+'\n'
            'miss_num='+str(MD)+'\n'
            'false_num='+str(FD)+'\n')

color.close()
#print(UC,CC,MD,FD)
cv2.waitKey(0)
cv2.imwrite("D:\\ImageProcess\\11_MSFF2Net\\code\\data\\whu\\res\\conc_color.png", result)