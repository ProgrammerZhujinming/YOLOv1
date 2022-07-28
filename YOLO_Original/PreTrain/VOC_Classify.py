from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import torchvision.transforms as transforms
import cv2
import os
import time
import random
from utils import image
import numpy as np
class voc_classify(Dataset):
    def __init__(self,imgs_path = "../DataSet/VOC2007+2012/Train/JPEGImages", annotations_path = "../DataSet/VOC2007+2012/Train/Annotations", classes_file = "../DataSet/VOC2007+2012/class.data", is_train=True, edge_threshold=200, input_size=224):  # input_size:输入图像的尺度
        imgs_path = os.listdir(imgs_path)
        self.is_train = is_train

        class_dict = {}
        class_index = 0
        with open(classes_file, 'r') as file:
            for class_name in file:
                class_name = class_name.replace('\n', '')
                class_dict[class_name] = class_index  # 根据类别名制作索引
                class_index = class_index + 1

        self.transform_common = transforms.Compose([
            transforms.ToTensor(),  # height * width * channel -> channel * height * width
            transforms.Normalize(mean=(0.408, 0.448, 0.471), std=(0.242, 0.239, 0.234))  # 归一化后.不容易产生梯度爆炸的问题
        ])

        self.input_size = input_size

        self.train_data = [] # [img_path,[[coord, class_id]]]
        for img_path in imgs_path:
            annotation_path = os.path.join(annotations_path, img_path.replace(".jpg", ".xml"))
            tree = ET.parse(annotation_path)
            annotation_xml = tree.getroot()
            objects_xml = annotation_xml.findall("object")
            coords = []
            for object_xml in objects_xml:
                class_name = object_xml.find("name").text
                if class_name not in class_dict:  # 不属于我们规定的类
                    continue

                class_index = class_dict[class_name]

                bnd_xml = object_xml.find("bndbox")
                xmin = round((float)(bnd_xml.find("xmin").text))
                ymin = round((float)(bnd_xml.find("ymin").text))
                xmax = round((float)(bnd_xml.find("xmax").text))
                ymax = round((float)(bnd_xml.find("ymax").text))

                if (xmax - xmin) < edge_threshold or (ymax - ymin) < edge_threshold:
                    pass

                coords.append([xmin, ymin, xmax, ymax, class_index])

            if len(coords) != 0:
                self.train_data.append([img_path, coords])

    def __getitem__(self, item):

        img_path, coords = self.train_data[item]
        img = cv2.imread(img_path)
        random.seed(int(time.time()))
        random_index = random.randint(0, len(coords) - 1)
        xmin, ymin, xmax, ymax, class_index = coords[random_index]
        img = img[ymin: ymax, xmin: xmax]

        if self.is_train:
            transform_seed = random.randint(0, 2)

            if transform_seed == 0:  # 原图
                img = image.resize_image_without_annotation(img, self.input_size, self.input_size)

            elif transform_seed == 1:  # 明度调整 YOLO在论文中称曝光度为明度
                img = image.resize_image_without_annotation(img, self.input_size, self.input_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                H, S, V = cv2.split(img)
                cv2.merge([np.uint8(H), np.uint8(S), np.uint8(V * 1.5)], dst=img)
                cv2.cvtColor(src=img, dst=img, code=cv2.COLOR_HSV2BGR)

            else:  # 饱和度调整
                img = image.resize_image_without_annotation(img, self.input_size, self.input_size)
                H, S, V = cv2.split(img)
                cv2.merge([np.uint8(H), np.uint8(S * 1.5), np.uint8(V)], dst=img)
                cv2.cvtColor(src=img, dst=img, code=cv2.COLOR_HSV2BGR)

        else:
            img = image.resize_image_without_annotation(img, self.input_size, self.input_size)

        img = self.transform_common(img)
        return img, class_index

    def __len__(self):
        return len(self.train_data)