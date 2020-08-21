from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Container(object):
    def setupUi(self, Container):
        Container.setObjectName("Container")
        Container.resize(1024, 768)
        
        # self.location = QtWidgets.QWidget(Container)
        # self.location.setGeometry(QtCore.QRect(30, 50, 480, 360))
        # self.location.setObjectName("location")

        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei")
        font.setPointSize(15)

        hBox = QtWidgets.QHBoxLayout()

        topLeft = QtWidgets.QFrame()
        topLeft.setFrameShape(QtWidgets.QFrame.StyledPanel)

        topRightTop = QtWidgets.QFrame()
        topRightTop.setFrameShape(QtWidgets.QFrame.StyledPanel)
        topRightTop.setVisible(True)

        topRightBottom = QtWidgets.QFrame()
        topRightBottom.setFrameShape(QtWidgets.QFrame.StyledPanel)

        bottom = QtWidgets.QFrame()
        bottom.setFrameShape(QtWidgets.QFrame.StyledPanel)

        # 右上部分上下窗口分割
        splitterTopRight = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        splitterTopRight.addWidget(topRightTop)
        splitterTopRight.addWidget(topRightBottom)
        splitterTopRight.setSizes([100, 400])
        # splitterTopRight.setDisabled(True)

        # 上部分左右窗口分割
        splitterTop = QtWidgets.QSplitter(QtCore.Qt.Horizontal) # Qt.Vertical 垂直   Qt.Horizontal 水平
        splitterTop.addWidget(topLeft)
        splitterTop.addWidget(splitterTopRight)
        splitterTop.setSizes([700,300])
        # splitterTop.setDisabled(True)
        
        # 整体布局上下窗口分割
        splitterBottom = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        splitterBottom.addWidget(splitterTop)
        splitterBottom.addWidget(bottom)
        splitterBottom.setSizes([500,200])
        # splitterBottom.setDisabled(True)

        # 设置窗口伸缩属性
        splitterTopRight.setStretchFactor(0, 0)
        splitterTopRight.setStretchFactor(1, 0)
        splitterTop.setStretchFactor(0, 0)
        splitterTop.setStretchFactor(1, 1)
        splitterBottom.setStretchFactor(0, 0)
        splitterBottom.setStretchFactor(1, 1)

        # 添加录入按键
        self.btnReg = QtWidgets.QPushButton(topRightTop)
        # self.btnReg.setGeometry(QtCore.QRect(550, 160, 80, 30))
        self.btnReg.setFont(font)
        self.btnReg.setFocusPolicy(QtCore.Qt.NoFocus)
        self.btnReg.setText("录 入")
        self.btnReg.setObjectName("btnReg")
        self.btnReg.setEnabled(True)

        # 添加开始按键
        self.btnStart = QtWidgets.QPushButton(topRightTop)
        # self.btnStart.setGeometry(QtCore.QRect(550, 220, 80, 30))
        self.btnStart.setFont(font)
        self.btnStart.setFocusPolicy(QtCore.Qt.NoFocus)
        self.btnStart.setText("开 始")
        self.btnStart.setObjectName("btnStart")
        # self.btnStart.setEnabled(True)
        self.btnStart.setDisabled = False

        # 添加倒计时标签
        self.timeLabel = QtWidgets.QLabel(topRightTop)
        self.timeLabel.setFixedWidth(200)
        self.timeLabel.setStyleSheet("QLabel{color:rgb(0, 0, 0);font-size:30px;font-weight:bold;font-family:宋体;}" )
        self.timeLabel.setText("00 : 00 : 00")

        # 上右上布局
        vBoxTopRightTop = QtWidgets.QVBoxLayout()
        # vBoxTopRightTop.addStretch(0)
        vBoxTopRightTop.addWidget(self.btnReg)
        vBoxTopRightTop.addWidget(self.btnStart)

        hBoxTopRightTop = QtWidgets.QHBoxLayout()
        # hBoxTopRightTop.addStretch(0)
        # hBoxTopRightTop.setSpacing(5)
        hBoxTopRightTop.addLayout(vBoxTopRightTop)
        hBoxTopRightTop.addWidget(self.timeLabel)
        topRightTop.setLayout(hBoxTopRightTop)
        
        # 添加事件组件标签
        self.eventLabel = QtWidgets.QLabel(topRightBottom)
        self.eventLabel.setFixedWidth(200)
        self.eventLabel.setStyleSheet("QLabel{color:rgb(128, 128, 128);font-size:10px;font-weight:bold;font-family:宋体;}" )
        self.eventLabel.setText("预警事件")

        # 添加事件列表
        self.eventListView = QtWidgets.QListView(topRightBottom)
        self.eventListModel = QtCore.QStringListModel()
        self.eventListView.setStyleSheet("QListView{border-width:0;border-style:outset;color:rgb(0, 0, 0);}")

        # self.list = ["[2020-08-20 10:00]: 空位事件","[2020-08-20 10:03]: 作弊事件", "[2020-08-20 10:10]: 作弊事件", "[2020-08-20 10:15]: 替考事件",
        #                     "[2020-08-20 10:00]: 空位事件","[2020-08-20 10:03]: 作弊事件", "[2020-08-20 10:10]: 作弊事件", "[2020-08-20 10:15]: 替考事件",
        #                     "[2020-08-20 10:00]: 空位事件","[2020-08-20 10:03]: 作弊事件", "[2020-08-20 10:10]: 作弊事件", "[2020-08-20 10:15]: 替考事件",
        #                     "[2020-08-20 10:00]: 空位事件","[2020-08-20 10:03]: 作弊事件", "[2020-08-20 10:10]: 作弊事件", "[2020-08-20 10:15]: 替考事件"]
        # self.eventListModel.setStringList(self.list)
        self.eventListView.setModel(self.eventListModel)
        

        # eventListView.clicked.connect(self.onClickedListView)

        # 上右下布局
        vBoxTopRightBottom = QtWidgets.QVBoxLayout()
        vBoxTopRightBottom.addWidget(self.eventLabel)
        vBoxTopRightBottom.addWidget(self.eventListView)
        topRightBottom.setLayout(vBoxTopRightBottom)

        # 添加日志显示标签
        self.logTextBrowser = QtWidgets.QTextBrowser(bottom)
        # self.logTextBrowser.setText("[OK]: 获取人脸图像\r\n[OK]: 获取人脸图像\r\n[OK]: 获取人脸图像\r\n[OK]: 获取人脸图像\r\n[OK]: 获取人脸图像\r\n")
        self.logTextBrowser.setStyleSheet("QTextBrowser{border-width:0;border-style:outset;color:rgb(0, 0, 0)}")

        # 下布局
        hBoxBottom = QtWidgets.QHBoxLayout()
        hBoxBottom.addWidget(self.logTextBrowser)
        bottom.setLayout(hBoxBottom)

        # # 添加视频显示标签
        self.cameraLabel = QtWidgets.QLabel(topLeft)
        self.cameraLabel.setMinimumSize(QtCore.QSize(700, 500))
        self.cameraLabel.setAlignment(QtCore.Qt.AlignCenter)

        self.cameraLabel.setPixmap(QtGui.QPixmap("./res/image_logo.png").scaled(150, 150))

        hBox.addWidget(splitterBottom)
        Container.setLayout(hBox)


        self.retranslateUi(Container)
        QtCore.QMetaObject.connectSlotsByName(Container)

    def retranslateUi(self, Container):
        _translate = QtCore.QCoreApplication.translate
        Container.setWindowTitle(_translate("Container", "Container"))