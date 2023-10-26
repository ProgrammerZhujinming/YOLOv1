import os
import cv2
import torch
import random
from utils import image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

voc_class_dict = {
    "aeroplane": 1,
    "bicycle": 2,
    "bird": 3,
    "boat": 4,
    "bottle": 5,
    "bus": 6,
    "car": 7,
    "cat": 8,
    "chair": 9,
    "cow": 10,
    "diningtable": 11,
    "dog": 12,
    "horse": 13,
    "motorbike": 14,
    "person": 15,
    "pottedplant": 16,
    "sheep": 17,
    "sofa": 18,
    "train": 19,
    "tvmonitor": 20
}

class COCO_Detection_Set(Dataset):
    def __init__(self):
        pass

    def getitem(self, item):
        pass
    
    def __len__(self):
        pass

class Detection_Set(Dataset):
    def __init__(self, data_path="../../data/VOC/train/labels/data.pth", is_train = True, class_num=20,
                 label_smooth_value=0.05, input_size=448, grid_size=64, loss_mode="mse"):  # input_size:输入图像的尺度
        self.label_smooth_value = label_smooth_value
        self.class_num = class_num
        self.input_size = input_size
        self.grid_size = grid_size
        self.is_train = is_train
        self.data = torch.load(data_path, torch.device("cpu"))["data"]
        self.transform_common = transforms.Compose([
            transforms.ToTensor(),  # height * width * channel -> channel * height * width
            transforms.Normalize(mean=(0.485, 0.456, 0.406) ,std=(0.229, 0.224, 0.225))  # 归一化后.不容易产生梯度爆炸的问题
        ])
        self.loss_mode = loss_mode

    def getData(self, img_path, coords):
        img = cv2.imread(img_path) 
        #coords.sort(key=lambda coord : (coord[2] - coord[0]) * (coord[3] - coord[1]) )

        if self.is_train:

            transform_seed = random.randint(0, 21)

            '''
            elif transform_seed == 0:  # 中心裁剪
                img, coords = image.center_crop(img, coords)

            elif transform_seed == 1:  # 平移
                img, coords = image.transplant(img, coords)
            '''

            #image.show_image_normal(img, coords, "ori")

            #原图不需要任何操作
            if transform_seed == 0:
                #image.show_image_normal(img, coords, "ori")
                pass

            elif transform_seed == 1:
                img = image.AddSaltPepperNoise(img)
                #image.show_image_normal(img, coords, "AddSaltPepperNoise")
            
            elif transform_seed == 2:
                img = image.AddGaussNoise(img)
                #image.show_image_normal(img, coords, "AddGaussNoise")

            elif transform_seed == 3:
                img = image.MeanBlur(img)
                #image.show_image_normal(img, coords, "MeanBlur")

            elif transform_seed == 4:
                img = image.GaussianBulr(img)
                #image.show_image_normal(img, coords, "GaussianBulr")

            elif transform_seed == 5:
                img = image.MedianBlur(img)
                #image.show_image_normal(img, coords, "MedianBlur")

            elif transform_seed == 6:
                img = image.BilateralBlur(img)
                #image.show_image_normal(img, coords, "BilateralBlur")

            elif transform_seed == 7:
                img, coords = image.RotateImage(img, coords)
                #img = image.SqrtImg(img)
                #image.show_image_normal(img, coords, "SqrtImg")

            elif transform_seed == 8:
                img = image.EqualizeHistImage(img)
                #image.show_image_normal(img, coords, "EqualizeHist")

            elif transform_seed == 9:
                img = image.ClaheImg(img)
                #image.show_image_normal(img, coords, "ClaheImg")

            elif transform_seed == 10:
                img = image.DetailEnhance(img)
                #image.show_image_normal(img, coords, "DetailEnhance")

            elif transform_seed == 11:
                img = image.illuminationChange(img)
                #image.show_image_normal(img, coords, "illuminationChange")

            elif transform_seed == 12:  #水平翻转
                img, coords = image.X_Flip(img, coords)
                #image.show_image_normal(img, coords, "X_Flip")

            elif transform_seed == 13:
                img, coords = image.transplant(img, coords)
                #image.show_image_normal(img, coords, "transplant")

            elif transform_seed == 14:
                img, coords = image.center_crop(img, coords)
                #image.show_image_normal(img, coords, "center_crop")

            elif transform_seed == 15:
                img, coords = image.RotateImage(img, coords)
                #image.show_image_normal(img, coords, "RotateImage")

            elif transform_seed == 16:
                img = image.AugBrightness_HSV(img)
            
            elif transform_seed == 17:
                img = image.AugBrightness_RGB(img)

            elif transform_seed == 18:
                img = image.change_contrast(img)

            elif transform_seed == 19:
                img = image.gamma_transfer(img)

            elif transform_seed == 20:
                img = image.saturation(img)

            elif transform_seed == 21:
                img = image.exposure(img)
            '''
            elif transform_seed == 4:  #亮度变化
                img = Image.fromarray(img)
                img = transforms.ColorJitter(brightness=0.5)(img)
                img = np.array(img)
                #img = image.change_brightness(img, brightness=1.03)

            elif transform_seed == 5:  #对比度变化
                img = Image.fromarray(img)
                img = transforms.ColorJitter(contrast=0.5)(img)
                img = np.array(img)
                #img = image.change_contrast(img, coefficent=1.2)

            elif transform_seed == 6:  #直方图均衡化:
                img = image.equalizeHist(img)

            elif transform_seed == 7:
                img = image.gamma_transfer(img, gamma=0.5)

            elif transform_seed == 8:
                img = image.gaussian_blur(img)

            elif transform_seed == 9:
                img = image.gaussian_noise(img)

            elif transform_seed == 10:
                img = Image.fromarray(img)
                img = transforms.ColorJitter(saturation=0.5)(img)
                img = np.array(img)
                #img = image.exposure(img, gamma=0.5)# 曝光度调整
            
            elif transform_seed == 11:
                img = Image.fromarray(img)
                img = transforms.ColorJitter(hue=0.5)(img)#色调
                img = np.array(img)
                #img = image.brightness(img)
            '''
        
        #from utils.image import show_image
        #show_image(img, coords)

        img, coords = image.resize_image(img, self.input_size, self.input_size, coords)
        img = self.transform_common(img)
        ground_truth, ground_mask_positive = self.getGroundTruth(coords)
        
        return img, ground_truth, ground_mask_positive

        #ground_truth, ground_mask_positive, ground_mask_negative = self.getGroundTruth(coords)
        #return img, ground_truth, ground_mask_positive, ground_mask_negative
        # 通道变化方法: img = img[:, :, ::-1]

    def __getitem__(self, item):
        img_path = self.data[item]['img_path']
        coords = self.data[item]['coords']
        return self.getData(img_path, coords)

    def __len__(self):
        return len(self.data)

    def getGroundTruth(self, coords):

        grid_num = self.input_size // self.grid_size

        ground_mask_positive = np.full(shape=(grid_num, grid_num, 1), fill_value=False, dtype=bool)

        if self.loss_mode == "mse":
            ground_truth = np.zeros([grid_num, grid_num, 10 + self.class_num + 2], dtype=np.float32)
        else:
            ground_truth = np.zeros([grid_num, grid_num, 10 + 1], dtype=np.float32)

        for coord in coords:

            xmin, ymin, xmax, ymax, class_id = coord

            ground_width = (xmax - xmin)
            ground_height = (ymax - ymin)

            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2

            index_row = (int)(center_y * grid_num)
            index_col = (int)(center_x * grid_num)


            # 分类标签 label_smooth
            if self.loss_mode == "mse":
                # 转化为one_hot编码 对one_hot编码做平滑处理
                class_list = np.full(shape=self.class_num, fill_value=1.0, dtype=float)
                class_list = class_list * (self.label_smooth_value / (self.class_num - 1))
                class_list[class_id] = 1.0 - self.label_smooth_value
            elif self.loss_mode == "cross_entropy":
                class_list = [class_id]
            else:
                raise Exception("the loss mode can't be support!")

            # 定位数据预设
            ground_box = [center_x * grid_num - index_col, center_y * grid_num - index_row,
                          ground_width, ground_height, 1,
                          round(xmin * self.input_size), round(ymin * self.input_size),
                          round(xmax * self.input_size), round(ymax * self.input_size),
                          round(ground_width * self.input_size * ground_height * self.input_size)
                          ]
            #print("gt:{}".format(ground_box))
            ground_box.extend(class_list)
            ground_box.extend([index_col, index_row])

            try:
                ground_truth[index_row][index_col] = np.array(ground_box)
                ground_mask_positive[index_row][index_col] = True
            except Exception:
                print("row:{} col:{} shape:{} center_x:{} grid_num:{} xmin:{} xmax:{}".format(index_row, index_col, ground_mask_positive.shape, center_x, grid_num, xmin, xmax))


        return ground_truth, torch.BoolTensor(ground_mask_positive)#, torch.BoolTensor(ground_mask_negative)








'''

from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
class voc_dataloader(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

import torch
class voc_prefetcher():
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_data, self.next_label = next(self.loader)
        except  StopIteration:
            self.next_data = None
            self.next_label = None
            return

        with torch.cuda.stream(self.stream):
            self.next_data = self.next_data.float().to(device=self.device,non_blocking=True)
            self.next_label[0] = self.next_label[0].float().to(device=self.device,non_blocking=True)
            self.next_label[1] = self.next_label[1].to(device=self.device, non_blocking=True)
            self.next_label[2] = self.next_label[2].to(device=self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data
        labels = self.next_label
        if data is not None:
            data.record_stream(torch.cuda.current_stream())
        if labels is not None:
            for label_idx in range(len(self.next_label)):
                self.next_label[label_idx].record_stream(torch.cuda.current_stream())

        self.preload()
        return data, labels
'''
'''
device = torch.device("cuda:0")
dataset = VOC_Detection_Set()
img, label = dataset.test_read_data()
img = img.unsqueeze(0).to(device=device)
label[0] = torch.FloatTensor(label[0]).to(device=device)
label[1] = torch.BoolTensor(label[1]).to(device=device)
label[2] = torch.BoolTensor(label[2]).to(device=device)
print(label[1].int().sum())
print(label[2].int().sum())
from YOLO.Train.YOLOv1_LossFunction import YOLOv1_Loss
yolo_loss = YOLOv1_Loss().to(device=device)
from YOLO.Train.YOLOv1_Model import YOLOv1
yolo = YOLOv1().to(device=device)
loss = yolo_loss(bounding_boxes=yolo(img), ground_labels=label)
print(loss)
'''