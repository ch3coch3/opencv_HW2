# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'd:\HW2\GUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1094, 781)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(60, 80, 311, 241))
        self.groupBox.setObjectName("groupBox")
        self.drawContour = QtWidgets.QPushButton(self.groupBox)
        self.drawContour.setGeometry(QtCore.QRect(50, 40, 201, 41))
        self.drawContour.setObjectName("drawContour")
        self.count_coins = QtWidgets.QPushButton(self.groupBox)
        self.count_coins.setGeometry(QtCore.QRect(50, 120, 201, 41))
        self.count_coins.setObjectName("count_coins")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setGeometry(QtCore.QRect(50, 180, 211, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setGeometry(QtCore.QRect(50, 210, 211, 16))
        self.label_2.setObjectName("label_2")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(60, 350, 611, 231))
        self.groupBox_2.setObjectName("groupBox_2")
        self.Find_Corner = QtWidgets.QPushButton(self.groupBox_2)
        self.Find_Corner.setGeometry(QtCore.QRect(50, 40, 201, 23))
        self.Find_Corner.setObjectName("Find_Corner")
        self.Find_Intrinsic = QtWidgets.QPushButton(self.groupBox_2)
        self.Find_Intrinsic.setGeometry(QtCore.QRect(50, 100, 201, 23))
        self.Find_Intrinsic.setObjectName("Find_Intrinsic")
        self.Find_Distorsion = QtWidgets.QPushButton(self.groupBox_2)
        self.Find_Distorsion.setGeometry(QtCore.QRect(50, 160, 201, 23))
        self.Find_Distorsion.setObjectName("Find_Distorsion")
        self.groupBox_3 = QtWidgets.QGroupBox(self.groupBox_2)
        self.groupBox_3.setGeometry(QtCore.QRect(320, 30, 261, 171))
        self.groupBox_3.setObjectName("groupBox_3")
        self.label_3 = QtWidgets.QLabel(self.groupBox_3)
        self.label_3.setGeometry(QtCore.QRect(30, 30, 81, 21))
        self.label_3.setObjectName("label_3")
        self.comboBox = QtWidgets.QComboBox(self.groupBox_3)
        self.comboBox.setGeometry(QtCore.QRect(30, 70, 69, 22))
        self.comboBox.setObjectName("comboBox")
        self.Find_Extrinsic = QtWidgets.QPushButton(self.groupBox_3)
        self.Find_Extrinsic.setGeometry(QtCore.QRect(30, 120, 181, 23))
        self.Find_Extrinsic.setObjectName("Find_Extrinsic")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1094, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "1.Find Contour"))
        self.drawContour.setText(_translate("MainWindow", "1.1 Draw Contour"))
        self.count_coins.setText(_translate("MainWindow", "1.2 Count Coins"))
        self.label.setText(_translate("MainWindow", "There are __ coins in coin01.jpg"))
        self.label_2.setText(_translate("MainWindow", "There are __ coins in coin02.jpg"))
        self.groupBox_2.setTitle(_translate("MainWindow", "2.Calibration"))
        self.Find_Corner.setText(_translate("MainWindow", "2.1 Find Corners"))
        self.Find_Intrinsic.setText(_translate("MainWindow", "2.2 Find Intrinsic"))
        self.Find_Distorsion.setText(_translate("MainWindow", "2.4 Find Distorsion"))
        self.groupBox_3.setTitle(_translate("MainWindow", "2.3 Find Extrinsic"))
        self.label_3.setText(_translate("MainWindow", "Select Image"))
        self.Find_Extrinsic.setText(_translate("MainWindow", "2.3 Find Extrinsic"))
