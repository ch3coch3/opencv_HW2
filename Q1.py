import cv2
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
    





# def contour(img):
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     # convert to gray
#     img = cv2.GaussianBlur(img, (11,11),0)          # gaussian blur
#     _,img = cv2.threshold(img, 120,255,cv2.THRESH_BINARY)
#     img = cv2.Canny(img,100,150)                    # canny to find edge
#     _,contours,_ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     print(len(contours))
#     img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     cv2.drawContours(img, contours, -1, (0,255,0),2)
#     return img

# img = cv2.imread("coin01.jpg")
# img1 = img.copy()
# img = cv2.imread("coin02.jpg")
# img2 = img.copy()

# img1 = contour(img1)
# img2 = contour(img2)
# cv2.imshow("coin_02", img1)
# # cv2.imshow("coin_02_original", img1)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
if __name__ == '__main__':
    app = QApplication([])
    window = myWindow()
    window.show()
    app.exec_()
    