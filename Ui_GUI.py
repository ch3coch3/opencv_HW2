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
        self.groupBox.setGeometry(QtCore.QRect(100, 130, 311, 241))
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