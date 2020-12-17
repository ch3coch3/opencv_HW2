import cv2
import numpy as np
from matplotlib import pyplot as plt

B = 178
f = 2826
d = 123
def cli_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        font = cv2.FONT_HERSHEY_COMPLEX
        strXY = str(x) + ', ' + str(y)
        dis = disparity[int(x/2),int(y/2)]
        depth = round(f*B/(dis[0]+123),0)
        cv2.rectangle(disparity, (width - 250, height - 100), (width, height), (255, 255, 255), -1)
        cv2.putText(disparity, "disparity pixel:"+str(dis[0]), (width-200,height-70), font, .5,(0,0,0),2)
        cv2.putText(disparity, "Depth:"+str(depth)+"mm", (width-200,height-30), font, .5,(0,0,0),2)
        cv2.imshow('disparity',disparity)

imgL = cv2.imread('Q4_Image\imgL.png',0)
imgR = cv2.imread('Q4_Image\imgR.png',0)
stereo = cv2.StereoBM_create(numDisparities=16*9, blockSize=5)
disparity = stereo.compute(imgL, imgR)

# normalize
disparity = cv2.normalize(disparity, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)

# change range 0>=1 to range 0=>255
disparity = np.array(disparity*255, dtype=np.uint8)

# to draw color point
disparity = cv2.cvtColor(disparity, cv2.COLOR_GRAY2RGB)

# resize
disparity = cv2.resize(disparity, (int(disparity.shape[1]/2),int(disparity.shape[0]/2)) )

width = int(disparity.shape[1]/2)
height = int(disparity.shape[0]/2)

cv2.imshow("disparity", disparity)
cv2.setMouseCallback('disparity', cli_event)
cv2.waitKey(0)
cv2.destroyAllWindows()


