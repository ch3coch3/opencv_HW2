import cv2
import os
import numpy as np
def draw(img, corners, imgpts):
    # 原點
    corner = tuple(corners[0].ravel())
    img = cv2.arrowedLine(img, corner, tuple(imgpts[0].ravel()), (0,155,255), 10)
    img = cv2.arrowedLine(img, corner, tuple(imgpts[1].ravel()), (0,155,255), 10)
    img = cv2.arrowedLine(img, corner, tuple(imgpts[2].ravel()), (0,155,255), 10)
    return img

def drawTetra(img, tetra):
    img = cv2.line(img, tuple(tetra[0].ravel()), tuple(tetra[1].ravel()), (0,0,255), 5)
    img = cv2.line(img, tuple(tetra[0].ravel()), tuple(tetra[2].ravel()), (0,0,255), 5)
    img = cv2.line(img, tuple(tetra[0].ravel()), tuple(tetra[3].ravel()), (0,0,255), 5)

    img = cv2.line(img, tuple(tetra[1].ravel()), tuple(tetra[2].ravel()), (0,0,255), 5)
    img = cv2.line(img, tuple(tetra[2].ravel()), tuple(tetra[3].ravel()), (0,0,255), 5)
    img = cv2.line(img, tuple(tetra[3].ravel()), tuple(tetra[1].ravel()), (0,0,255), 5)
    return img
    

def button_3():
    nx = 11
    ny = 8
    dir_path = './Q3_Image/'
    fname = os.listdir(dir_path)
    fname.sort(key=lambda x: int(x[:-4]))       
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    # 定義座標軸點
    axis = np.float32([[5,0,0], [0,5,0], [0,0,-5]]).reshape(-1,3)
    # 定義tetra點
    tetraPoint = np.float32([[3,3,-3],[1,1,0],[3,5,0],[5,1,0]]).reshape(-1,3)

    object_pt = []
    img_pt = []
    num = 1
    # id = str(self.comboBox.currentText())
    for name in fname:
        img = cv2.imread(dir_path+name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corner = cv2.findChessboardCorners(img, (nx,ny))
        if ret == True:
            object_pt.clear()
            img_pt.clear()
            object_pt.append(objp)
            img_pt.append(corner)

        # calibrate the camera
        img_size = (img.shape[1], img.shape[0])
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_pt, img_pt, img_size,None,None)
        mtx = np.reshape(mtx,(3,3))

        rvecs,_ = cv2.Rodrigues(np.squeeze(rvecs,None))
        tvecs = np.reshape(tvecs,(3,1))
        extrinsic = np.concatenate([rvecs, tvecs],axis=1)

        # 投影目標點
        imgpts, _ = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        tetra, _ = cv2.projectPoints(tetraPoint, rvecs, tvecs, mtx, dist)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # 畫坐標軸
        img = draw(img, corner, imgpts)
        img = drawTetra(img, tetra)
        img = cv2.resize(img, (int(img.shape[1]/2),int(img.shape[0]/2)),interpolation=cv2.INTER_LINEAR)


        cv2.imshow(name, img)
        cv2.waitKey(1000)
        num = num + 1


if __name__=='__main__':
    findIntrinsic()
    cv2.destroyAllWindows()
