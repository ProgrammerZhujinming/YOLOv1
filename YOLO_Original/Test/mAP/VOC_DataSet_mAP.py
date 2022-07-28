from torch.utils.data import Dataset
import os
import cv2
import xml.etree.ElementTree as ET
import torchvision.transforms as transforms
import numpy as np
import random
from utils import image

class VOCDataSet_mAP(Dataset):
    def __init__(self, imgs_path="../DataSet/VOC2007+2012/Train/JPEGImages",
                 annotations_path="../DataSet/VOC2007+2012/Train/Annotations",
                 classes_file="../DataSet/VOC2007+2012/class.data", is_train = True, class_num=20,
                 label_smooth_value=0.05, input_size=448, grid_size=64):  # input_size:输入图像的尺度
        self.label_smooth_value = label_smooth_value
        self.class_num = class_num
        self.imgs_name = os.listdir(imgs_path)
        self.input_size = input_size
        self.grid_size = grid_size
        self.is_train = is_train
        self.transform_common = transforms.Compose([
            transforms.ToTensor(),  # height * width * channel -> channel * height * width
            transforms.Normalize(mean=(0.408, 0.448, 0.471), std=(0.242, 0.239, 0.234))  # 归一化后.不容易产生梯度爆炸的问题
        ])
        self.imgs_path = imgs_path
        self.annotations_path = annotations_path
        self.class_dict = {}
        class_index = 0
        with open(classes_file, 'r') as file:
            for class_name in file:
                class_name = class_name.replace('\n', '')
                self.class_dict[class_name] = class_index  # 根据类别名制作索引
                class_index = class_index + 1

    def __getitem__(self, item):

        img_path = os.path.join(self.imgs_path, self.imgs_name[item])
        annotation_path = os.path.join(self.annotations_path, self.imgs_name[item].replace(".jpg", ".xml"))
        img = cv2.imread(img_path)
        tree = ET.parse(annotation_path)
        annotation_xml = tree.getroot()

        objects_xml = annotation_xml.findall("object")
        coords = []

        for object_xml in objects_xml:
            bnd_xml = object_xml.find("bndbox")
            class_name = object_xml.find("name").text
            if class_name not in self.class_dict:  # 不属于我们规定的类
                continue
            xmin = round((float)(bnd_xml.find("xmin").text))
            ymin = round((float)(bnd_xml.find("ymin").text))
            xmax = round((float)(bnd_xml.find("xmax").text))
            ymax = round((float)(bnd_xml.find("ymax").text))
            class_id = self.class_dict[class_name]
            coords.append([xmin, ymin, xmax, ymax, class_id])

        if self.is_train:

            transform_seed = random.randint(0, 4)

            if transform_seed == 0:  # 原图
                img, coords = image.resize_image_with_coords(img, self.input_size, self.input_size, coords)
                img = self.transform_common(img)

            elif transform_seed == 1:  # 缩放+中心裁剪
                img, coords = image.center_crop_with_coords(img, coords)
                img, coords = image.resize_image_with_coords(img, self.input_size, self.input_size, coords)
                img = self.transform_common(img)

            elif transform_seed == 2:  # 平移
                img, coords = image.transplant_with_coords(img, coords)
                img, coords = image.resize_image_with_coords(img, self.input_size, self.input_size, coords)
                img = self.transform_common(img)

            elif transform_seed == 3:  # 明度调整 YOLO在论文中称曝光度为明度
                img, coords = image.resize_image_with_coords(img, self.input_size, self.input_size, coords)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                H, S, V = cv2.split(img)
                cv2.merge([np.uint8(H), np.uint8(S), np.uint8(V * 1.5)], dst=img)
                cv2.cvtColor(src=img, dst=img, code=cv2.COLOR_HSV2BGR)
                img = self.transform_common(img)

            else:  # 饱和度调整
                img, coords = image.resize_image_with_coords(img, self.input_size, self.input_size, coords)
                H, S, V = cv2.split(img)
                cv2.merge([np.uint8(H), np.uint8(S * 1.5), np.uint8(V)], dst=img)
                cv2.cvtColor(src=img, dst=img, code=cv2.COLOR_HSV2BGR)
                img = self.transform_common(img)

        else:
            img, coords = image.resize_image_with_coords(img, self.input_size, self.input_size, coords)
            img = self.transform_common(img)

        ground_truth = self.getGroundTruth(coords)

        # 通道变化方法: img = img[:, :, ::-1]

        return img, ground_truth, self.imgs_name[item]

    def __len__(self):
        return len(self.imgs_name)

    def getGroundTruth(self, coords):

        feature_size = self.input_size // self.grid_size
        #ground_truth = np.zeros([feature_size, feature_size, 10 + self.class_num])
        ground_truth = np.zeros([feature_size, feature_size, 10 + 1])

        for coord in coords:
            # positive_num = positive_num + 1
            # bounding box归一化
            xmin, ymin, xmax, ymax, class_id = coord

            ground_width = (xmax - xmin)
            ground_height = (ymax - ymin)

            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2

            # 计算当前中心点分别落于3个特征尺度下的哪个grid内
            index_row = (int)(center_y * feature_size)
            index_col = (int)(center_x * feature_size)

            # 分类标签 label_smooth
            '''
            # 转化为one_hot编码，将物体的类别设置为1，其他为0
            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            # 对one_hot编码做平滑处理          
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            # 计算中心点坐标(x,y) = ((x_max, y_max) + (x_min, y_min)) * 0.5
            # 计算宽高(w,h) = (x_max, y_max) - (x_min, y_min)
            # 拼接成一个数组(x, y, w, h)
            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)

            '''
            #class_one_hot = [0 for _ in range(self.class_num)]
            #class_list = [self.label_smooth_value / (self.class_num - 1) for i in range(self.class_num)]
            #class_list[class_index] = 1 - self.label_smooth_value

            # 定位数据预设
            ground_box = [center_x * feature_size - index_col, center_y * feature_size - index_row,
                          ground_width, ground_height, 1,
                          round(xmin * self.input_size), round(ymin * self.input_size),
                          round(xmax * self.input_size), round(ymax * self.input_size),
                          round(ground_width * self.input_size * ground_height * self.input_size)
                          ]
            #ground_box.extend(class_list)
            ground_box.extend([class_id])

            ground_truth[index_row][index_col] = np.array(ground_box)

        return ground_truth

'''
import random
from torch.utils.data import Dataset
import os
import cv2
import xml.etree.ElementTree as ET
import torch
import torchvision.transforms as transforms
import numpy as np
import math

class VOCDataSet(Dataset):

    def __init__(self, imgs_dir="../DataSet/VOC2007+2012/Train/JPEGImages", annotations_dir="../DataSet/VOC2007+2012/Train/Annotations", img_size=448, S=7, B=2, ClassesFile="../DataSet/VOC2007+2012/class.data", label_smooth_value = 0.05): # 图片路径、注解文件路径、图片尺寸、每个grid cell预测的box数量、类别文件
        img_names = os.listdir(imgs_dir)
        img_names.sort()
        self.transform_common = transforms.Compose([
            transforms.ToTensor(),  # height * width * channel -> channel * height * width
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))  # 归一化后.不容易产生梯度爆炸的问题
        ])
        self.transform_bright_value = transforms.ColorJitter(brightness=1.5)
        self.transform_saturation = transforms.ColorJitter(saturation=1.5)
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

        self.class_bbox_nums = [0 for index in range(classIndex)]
        self.class_bboxes = [list() for index in range(classIndex)]

        print(self.ClassNameToClassIndex)
        self.classNum = classIndex # 一共的类别个数
        self.label_smooth_value = label_smooth_value
        self.getGroundTruth()
        self.data = [list([self.img_path[i], self.ground_truth[i]]) for i in range(len(self.img_path))]

    def generateGroundTruth(self, annotation_path):
        final_ground_truth = np.zeros(shape=(self.S, self.S, 10 + 1))
        ground_truth_index = 0
        ground_truth = [[list() for row in range(self.S)] for col in range(self.S)]
        # 解析xml文件--标注文件
        tree = ET.parse(annotation_path)
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
            if class_name not in self.ClassNameToClassIndex:  # 不属于我们规定的类
                continue
            bnd_xml = object_xml.find("bndbox")
            # 目标尺度放缩
            xmin = (int)((float)(bnd_xml.find("xmin").text) * scaleX)
            ymin = (int)((float)(bnd_xml.find("ymin").text) * scaleY)
            xmax = (int)((float)(bnd_xml.find("xmax").text) * scaleX)
            ymax = (int)((float)(bnd_xml.find("ymax").text) * scaleY)
            # 目标中心点
            centerX = (xmin + xmax) / 2
            centerY = (ymin + ymax) / 2
            # 当前物体的中心点落于 第indexRow行 第indexCol列的 grid cell内
            indexRow = (int)(centerY / self.grid_cell_size)
            indexCol = (int)(centerX / self.grid_cell_size)
            # 真实物体的list
            ClassIndex = self.ClassNameToClassIndex[class_name]
            # label_smooth技术
            # ClassList = [self.label_smooth_value / (self.classNum - 1) for i in range(self.classNum)]
            # ClassList[ClassIndex] = 1 - self.label_smooth_value
            ground_box = list([centerX / self.grid_cell_size - indexCol, centerY / self.grid_cell_size - indexRow,
                                   (xmax - xmin) / self.img_size, (ymax - ymin) / self.img_size, 1, xmin, ymin, xmax,
                                   ymax, (xmax - xmin) * (ymax - ymin)])
            # 增加上类别
            # ground_box.extend(ClassList)
            ground_box.append(ClassIndex)
            ground_truth[indexRow][indexCol].append(ground_box)

        # 同一个grid cell内的多个groudn_truth，选取面积最大的那个
        for i in range(self.S):
            for j in range(self.S):
                if len(ground_truth[i][j]) != 0:
                    ground_truth[i][j].sort(key=lambda box: box[9], reverse=True)
                    final_ground_truth[i][j] = np.array(ground_truth[i][j][0])

        return torch.Tensor(final_ground_truth)

    # PyTorch 无法将长短不一的list合并为一个Tensor
    def getGroundTruth(self):
        self.ground_truth = np.zeros(shape=(len(self.img_path), self.S, self.S, 10 + 1))
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
                xmin = (int)((float)(bnd_xml.find("xmin").text) * scaleX)
                ymin = (int)((float)(bnd_xml.find("ymin").text) * scaleY)
                xmax = (int)((float)(bnd_xml.find("xmax").text) * scaleX)
                ymax = (int)((float)(bnd_xml.find("ymax").text) * scaleY)
                # 目标中心点
                centerX = (xmin + xmax) / 2
                centerY = (ymin + ymax) / 2
                # 当前物体的中心点落于 第indexRow行 第indexCol列的 grid cell内
                indexRow = (int)(centerY / self.grid_cell_size)
                indexCol = (int)(centerX / self.grid_cell_size)
                # 真实物体的list
                ClassIndex = self.ClassNameToClassIndex[class_name]
                # label_smooth技术
                #ClassList = [self.label_smooth_value / (self.classNum - 1) for i in range(self.classNum)]
                #ClassList[ClassIndex] = 1 - self.label_smooth_value
                ground_box = list([centerX / self.grid_cell_size - indexCol,centerY / self.grid_cell_size - indexRow,(xmax-xmin)/self.img_size,(ymax-ymin)/self.img_size,1,xmin,ymin,xmax,ymax,(xmax-xmin)*(ymax-ymin)])
                #增加上类别
                #ground_box.extend(ClassList)
                ground_box.append(ClassIndex)
                ground_truth[indexRow][indexCol].append(ground_box)

            #同一个grid cell内的多个groudn_truth，选取面积最大的那个
            for i in range(self.S):
                for j in range(self.S):
                    if len(ground_truth[i][j]) != 0:
                        ground_truth[i][j].sort(key = lambda box: box[9], reverse=True)
                        self.ground_truth[ground_truth_index][i][j] = np.array(ground_truth[i][j][0])

                        ground_index = np.argmax(ground_truth[i][j][0][10:])
                        self.class_bboxes[ground_index].append(ground_truth[i][j][0])
                        self.class_bbox_nums[ground_index] = self.class_bbox_nums[ground_index] + 1

            ground_truth_index = ground_truth_index + 1
        self.ground_truth = torch.Tensor(self.ground_truth)

    def __getitem__(self, item):
        transform_seed = random.randint(0, 4)
        # 为了加速，增强的图像预先生成
        # height * width * channel
        img_bgr = cv2.imread(self.data[item][0])
        img_bgr = cv2.resize(img_bgr, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)
        img_tensor = self.transform_common(img_bgr)
        ground_truth = self.data[item][1]
        
        # 使用随机数选择增强方案
        if transform_seed == 0: #原图
            img_bgr = cv2.imread(self.data[item][0])
            img_bgr = cv2.resize(img_bgr, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)
            img_tensor = self.transform_common(img_bgr)
            ground_truth = self.data[item][1]

        elif transform_seed == 1: #缩放+中心裁剪
            img_path = self.data[item][0]
            img_path = img_path.replace("JPEGImages", "Augment/Scaling/JPEGImages")
            annotation_path = img_path.replace("JPEGImages", "Augment/Scaling/Annotations")
            img_bgr = cv2.imread(self.data[item][0])
            img_bgr = cv2.resize(img_bgr, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
            ground_truth = self.generateGroundTruth(annotation_path=annotation_path)

        elif transform_seed == 2: #平移
            img_path = self.data[item][0]
            img_path = img_path.replace("JPEGImages", "Augment/Translation/JPEGImages")
            annotation_path = img_path.replace("JPEGImages", "Augment/Translation/Annotations")
            img_bgr = cv2.imread(self.data[item][0])
            img_bgr = cv2.resize(img_bgr, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
            ground_truth = self.generateGroundTruth(annotation_path=annotation_path)

        elif transform_seed == 3: # 明度调整 YOLO在论文中称曝光度为明度
            img_path = self.data[item][0]
            img_path = img_path.replace("JPEGImages", "Augment/Exposure/JPEGImages")
            img_bgr = cv2.imread(img_path)
            img_bgr = cv2.resize(img_bgr, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)
            img_tensor = self.transform_saturation(img_bgr)
            img_tensor = self.transform_common(img_tensor)
            ground_truth = self.data[item][1]

        else: # 饱和度调整
            # 转换成HSV空间
            # img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            # H, S, V = cv2.split(img_hsv)
            img_path = self.data[item][0]
            img_path = img_path.replace("JPEGImages", "Augment/Saturation/JPEGImages")
            img_bgr = cv2.imread(img_path)
            img_bgr = cv2.resize(img_bgr, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)
            img_tensor = self.transform_saturation(img_bgr)
            img_tensor = self.transform_common(img_tensor)
            ground_truth = self.data[item][1]
        

        return img_tensor,ground_truth

    def __len__(self):
        return len(self.img_path)

    def shuffleData(self):
        random.shuffle(self.data)

'''