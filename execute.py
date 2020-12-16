import cv2
import numpy as np
import os
import PyQt5
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel,QWidget, QPushButton, QVBoxLayout
from PyQt5.QtCore import pyqtSlot
from Ui_GUI import Ui_MainWindow

class myWindow(QtWidgets.QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(myWindow, self).__init__()
        self.setupUi(self)
        self.drawContour.clicked.connect(lambda:self.contour('1'))
        self.count_coins.clicked.connect(lambda:self.contour('2'))
        self.Find_Corner.clicked.connect(lambda:self.corner())
        self.Find_Intrinsic.clicked.connect(lambda:self.findIntrinsic('1'))
        self.Find_Distorsion.clicked.connect(lambda:self.findIntrinsic('3'))
        self.AugmentedReality.clicked.connect(lambda:self.button_3())

        # combobox
        choice =  [str(x) for x in range(1,16)]
        self.comboBox.addItems(choice)
        self.Find_Extrinsic.clicked.connect(lambda:self.findIntrinsic('2'))

    @pyqtSlot()
    def contour(self,st):
        dir_path = './Q1_Image/'
        fname = os.listdir(dir_path)
        j = 1
        for name in fname:
            img = cv2.imread(dir_path + name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     # convert to gray
            img = cv2.GaussianBlur(img, (11,11),0)          # gaussian blur
            _,img = cv2.threshold(img, 120,255,cv2.THRESH_BINARY)
            img = cv2.Canny(img,100,150)                    # canny to find edge
            _,contours,_ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(img, contours, -1, (0,255,0),2)
            if st == '1':
                cv2.imshow(name,img)
            if st == '2':
                if j == 1:
                    self.label.setText("There are "+str(len(contours))+" coins in coin01.jpg")
                    j = 0
                else:
                    self.label_2.setText("There are "+str(len(contours))+" coins in coin01.jpg")
    def corner(self):
        nx = 11
        ny = 8
        dir_path = './Q2_Image/'
        fname = os.listdir(dir_path)
        fname.sort(key=lambda x: int(x[:-4]))
        for name in fname:
            img = cv2.imread(dir_path+name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corner = cv2.findChessboardCorners(img, (nx,ny))


            if ret == True:
                cv2.drawChessboardCorners(img,(nx,ny), corner, ret)
                img = cv2.resize(img, (int(img.shape[1]/2),int(img.shape[0]/2)),interpolation=cv2.INTER_LINEAR)
                cv2.imshow(name, img)
                cv2.waitKey(500)
        cv2.destroyAllWindows()


    def findIntrinsic(self, st):
        nx = 11
        ny = 8
        dir_path = './Q2_Image/'
        fname = os.listdir(dir_path)
        fname.sort(key=lambda x: int(x[:-4]))       
        objp = np.zeros((nx*ny,3), np.float32)
        objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

        object_pt = []
        img_pt = []
        num = 1
        id = str(self.comboBox.currentText())
        for name in fname:
            img = cv2.imread(dir_path+name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corner = cv2.findChessboardCorners(img, (nx,ny))
            if ret == True:
                object_pt.clear()
                img_pt.clear()
                object_pt.append(objp)
                img_pt.append(corner)
                cv2.drawChessboardCorners(img,(nx,ny), corner, ret)

            # calibrate the camera
            img_size = (img.shape[1], img.shape[0])
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_pt, img_pt, img_size,None,None)
            mtx = np.reshape(mtx,(3,3))
            if st == '1':
                print("Intrinsic matrix of",name)
                print(mtx)
            if st == '2' and num == int(id):
                rvecs,_ = cv2.Rodrigues(np.squeeze(rvecs,None))
                tvecs = np.reshape(tvecs,(3,1))
                extrinsic = np.concatenate([rvecs, tvecs],axis=1)
                print("extrinsic matrix of",name)
                print(extrinsic)
                break
            if st == '3':
                print("Distorsion matrix of",name)
                print(dist)

            num = num + 1
      
    def button_3(self):
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
            cv2.waitKey(500)
            num = num + 1
        cv2.destroyAllWindows()

# Q3
def draw(img, corners, imgpts):
    # original point
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


if __name__ == '__main__':
    app = QApplication([])
    window = myWindow()
    window.show()
    app.exec_()
    