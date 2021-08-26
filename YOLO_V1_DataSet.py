import random
from torch.utils.data import Dataset
import os
import cv2
import xml.etree.ElementTree as ET
import torch
import torchvision.transforms as transforms
import numpy as np

class VOCDataSet(Dataset):

    def __init__(self, imgs_dir="./VOC2007/Train/JPEGImages", annotations_dir="./VOC2007/Train/Annotations", img_size=448, S=7, B=2, ClassesFile="../VOC2007/Train/class.data", label_smooth_value = 0.05): # 图片路径、注解文件路径、图片尺寸、每个grid cell预测的box数量、类别文件
        img_names = os.listdir(imgs_dir)
        img_names.sort()
        self.transfrom = transforms.Compose([
            transforms.ToTensor(), # height * width * channel -> channel * height * width
            transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)) #归一化后.不容易产生梯度爆炸的问题
        ])
        self.img_path = []
        for img_name in img_names:
            self.img_path.append(os.path.join(imgs_dir,img_name))
        annotation_names = os.listdir(annotations_dir)
        annotation_names.sort() #图片和文件排序后可以按照相同索引对应
        self.annotation_path = []
        for annotation_name in annotation_names:
            self.annotation_path.append(os.path.join(annotations_dir,annotation_name))
        self.img_size = img_size
        self.S = S
        self.B = B
        self.grid_cell_size = self.img_size / self.S
        self.ClassNameToClassIndex = {}
        classIndex = 0
        with open(ClassesFile, 'r') as classNameFile:
            for className in classNameFile:
                className = className.replace('\n','')
                self.ClassNameToClassIndex[className] = classIndex #根据类别名制作索引
                classIndex = classIndex + 1
        print(self.ClassNameToClassIndex)
        self.classNum = classIndex # 一共的类别个数
        self.label_smooth_value = label_smooth_value
        self.getGroundTruth()
        self.data = [list([self.img_path[i], self.ground_truth[i]]) for i in range(len(self.img_path))]

    # PyTorch 无法将长短不一的list合并为一个Tensor
    def getGroundTruth(self):
        self.ground_truth = np.zeros(shape=(len(self.img_path), self.S, self.S, 10 + self.classNum))
        ground_truth_index = 0
        for annotation_file in self.annotation_path:
            ground_truth = [[list() for row in range(self.S)] for col in range(self.S)]
            # 解析xml文件--标注文件
            tree = ET.parse(annotation_file)
            annotation_xml = tree.getroot()
            # 计算 目标尺寸 对于 原图尺寸 width的比例
            width = (int)(annotation_xml.find("size").find("width").text)
            scaleX = self.img_size / width
            # 计算 目标尺寸 对于 原图尺寸 height的比例
            height = (int)(annotation_xml.find("size").find("height").text)
            scaleY = self.img_size / height
            # 因为两次除法的误差可能比较大 这边采用除一次乘一次的方式
            # 一个注解文件可能有多个object标签，一个object标签内部包含一个bnd标签
            objects_xml = annotation_xml.findall("object")
            for object_xml in objects_xml:
                # 获取目标的名字
                class_name = object_xml.find("name").text
                if class_name not in self.ClassNameToClassIndex: # 不属于我们规定的类
                    continue
                bnd_xml = object_xml.find("bndbox")
                # 目标尺度放缩
                xmin = (int)((int)(bnd_xml.find("xmin").text) * scaleX)
                ymin = (int)((int)(bnd_xml.find("ymin").text) * scaleY)
                xmax = (int)((int)(bnd_xml.find("xmax").text) * scaleX)
                ymax = (int)((int)(bnd_xml.find("ymax").text) * scaleY)
                # 目标中心点
                centerX = (xmin + xmax) / 2
                centerY = (ymin + ymax) / 2
                # 当前物体的中心点落于 第indexRow行 第indexCol列的 grid cell内
                indexRow = (int)(centerY / self.grid_cell_size)
                indexCol = (int)(centerX / self.grid_cell_size)
                # 真实物体的list
                ClassIndex = self.ClassNameToClassIndex[class_name]
                # label_smooth技术
                ClassList = [self.label_smooth_value / (self.classNum - 1) for i in range(self.classNum)]
                ClassList[ClassIndex] = 1 - self.label_smooth_value
                ground_box = list([centerX / self.grid_cell_size - indexCol,centerY / self.grid_cell_size - indexRow,(xmax-xmin)/self.img_size,(ymax-ymin)/self.img_size,1,xmin,ymin,xmax,ymax,(xmax-xmin)*(ymax-ymin)])
                #增加上类别
                ground_box.extend(ClassList)
                ground_truth[indexRow][indexCol].append(ground_box)

            #同一个grid cell内的多个groudn_truth，选取面积最大的那个
            for i in range(self.S):
                for j in range(self.S):
                    if len(ground_truth[i][j]) != 0:
                        ground_truth[i][j].sort(key = lambda box: box[9], reverse=True)
                        self.ground_truth[ground_truth_index][i][j] = np.array(ground_truth[i][j][0])

            ground_truth_index = ground_truth_index + 1
        self.ground_truth = torch.Tensor(self.ground_truth)

    def __getitem__(self, item):
        # height * width * channel
        img_data = cv2.imread(self.data[item][0])
        img_data = cv2.resize(img_data, (448, 448), interpolation=cv2.INTER_AREA)
        img_data = self.transfrom(img_data)
        return img_data,self.data[item][1]

    def __len__(self):
        return len(self.img_path)

    def shuffleData(self):
        random.shuffle(self.data)

