'''
@authur: Nil_gox
@file: yolo_detection.py
@time: 2020/5/10 14:23
'''
import PIL
import numpy as np
import cv2 as cv
import time
from PIL import ImageTk
import tkinter as tk
from tkinter import *
from tkinter.filedialog import askopenfilename

class Application(Frame) :
    def __init__ (self, master=None) :
        Frame.__init__(self, master, bg='black')
        self.pack(expand=YES, fill=BOTH)
        self.window_init()
        self.createWidgets()

    def window_init (self) :
        self.master.title('宠物品种识别')
        self.master.bg = 'black'
        width, height = self.master.maxsize()
        self.master.geometry("{}x{}".format(width, height))

    def createWidgets (self) :
        # fm1
        self.fm1 = Frame(self, bg='black')
        self.titleLabel = Label(self.fm1, text="Welcome to use traffic-light detective system", font=('微软雅黑', 24), fg="white", bg='black')
        self.titleLabel.pack()
        self.fm1.pack(side=TOP, expand=0, fill='x', pady=20)

        # fm2
        self.fm2 = Frame(self, bg='black')
        self.fm2_left = Frame(self.fm2, bg='black')

        self.selectEntry = Entry(self.fm2_left, font=('微软雅黑', 24), width='72', fg='#1E90FF')
        self.selectButton = Button(self.fm2_left, text='select', bg='#1E90FF', fg='white',font=('微软雅黑', 36), width='16', command=self.Open_Img)
        self.selectButton.pack(side=LEFT)
        self.selectEntry.pack(side=LEFT, fill='y', padx=20)
        self.fm2_left.pack(side=LEFT, padx=30, pady=20, expand=1, fill='x')

        self.fm2.pack(side=TOP, expand=1, fill="x")

        # fm3
        self.fm3 = Frame(self, bg='black')
        self.fm3.pack(side=TOP, expand=1, fill=BOTH, pady=10)

        # fm4
        self.fm4 = Frame(self, bg='black')
        self.startImg = ImageTk.PhotoImage(file='yunxing.png')
        self.startButton = Button(self.fm4, image=self.startImg, text='start', bg='black', fg='white',
                                  font=('微软雅黑', 36), command=self.yolo_detect)
        self.startButton.pack(expand=1, fill=BOTH)
        self.fm4.pack(side=BOTTOM,expand=0,pady=30)

    def Open_Img (self) :
        global img_path
        img_path = askopenfilename()
        img = PIL.Image.open(img_path)
        (x, y) = img.size
        x_s = 250
        y_s = y * x_s // x
        img = img.resize((x_s, y_s), PIL.Image.ANTIALIAS)
        initIamge = ImageTk.PhotoImage(img)
        self.panel = Label(self.fm3, image=initIamge)
        self.panel.image = initIamge
        self.panel.pack(side=LEFT, padx=30, pady=20, expand=1)

        self.selectEntry.delete(0, END)
        # self.selectEntry.config(text=img_path)
        self.selectEntry.insert(0, img_path)

    def yolo_detect(self):
        weightsPath = "yolov3-voc_5000.weights"  # 权重文件
        configPath = "yolov3-pet.cfg"  # 配置文件
        labelsPath = "pet.names"  # label名称
        imgPath =  img_path  # 测试图像
        CONFIDENCE = 0.85  # 过滤弱检测的最小概率
        THRESHOLD = 0.4  # 非极大值抑制阈值

        # 加载网络、配置权重
        net = cv.dnn.readNetFromDarknet(configPath, weightsPath)

        # 加载图片、转为blob格式、送入网络输入层
        img = cv.imread(imgPath)
        width, height = img.shape[:2][: :-1]
        img = cv.resize(img, (int(width * 1), int(height * 1)), interpolation=cv.INTER_CUBIC)
        blobImg = cv.dnn.blobFromImage(img, 1.0/255.0, (416, 416), None, True, False)
        net.setInput(blobImg)

        # 获取网络输出层信息（所有输出层的名字），设定并前向传播
        outInfo = net.getUnconnectedOutLayersNames()
        start = time.time()
        layerOutputs = net.forward(outInfo)
        end = time.time()

        # 获取图片新的尺寸
        (H, W) = img.shape[:2]

        # 过滤layerOutputs
        boxes = [] # 所有边界框（各层结果放一起）
        confidences = [] # 所有置信度
        classIDs = [] # 所有分类ID

        # 过滤掉置信度低的框
        for out in layerOutputs:
            for detection in out:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # 根据置信度筛查
                if confidence > CONFIDENCE:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # 使用非极大值抑制进一步筛选
        idxs = cv.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)
        # 得到labels列表
        with open(labelsPath, 'rt') as f:
            labels = f.read().rstrip('\n').split('\n')
        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                color = [int(c) for c in COLORS[classIDs[i]]]
                cv.rectangle(img, (x, y), (x+w, y+h), color, 2)
                text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
                cv.putText(img, text, (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv.imshow('detected image', img)
        cv.waitKey(0)

if __name__ == '__main__' :
    app = Application()
    # to do
    app.mainloop()