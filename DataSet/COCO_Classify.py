from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import os
import time
import random
from utils import image
import numpy as np
class coco_classify(Dataset):
    def __init__(self,imgs_path = "../DataSet/COCO2014/train/JPEGImages", annotations_path = "../DataSet/COCO2014/train/label-train2014", edge_threshold=200, img_size=256, class_num=80):  # input_size:输入图像的尺度
        img_names = os.listdir(imgs_path)

        self.transform_common = transforms.Compose([
            transforms.ToTensor(),  # height * width * channel -> channel * height * width
            transforms.Normalize(mean=(0.408, 0.448, 0.471), std=(0.242, 0.239, 0.234))  # 归一化后.不容易产生梯度爆炸的问题
        ])

        self.img_size = img_size

        self.train_data = [] # [img_path,[[coord, class_id]]]
        for img_name in img_names:
            img_path = os.path.join(imgs_path, img_name)
            annotation_path = os.path.join(annotations_path, img_name.replace(".jpg", ".txt"))
            img = cv2.imread(img_path)

            width, height, _ = img.shape
            coords = []

            with open(annotation_path, 'r') as label_txt:
                for label in label_txt:
                    label = label.replace("\n", "").split(" ")
                    class_id = int(label[0])

                    if class_id >= class_num:
                        continue

                    center_x = round(float(label[1]) * width)
                    center_y = round(float(label[2]) * height)
                    box_width = round(float(label[3]) * width)
                    box_height = round(float(label[4]) * height)

                    xmin = round(min(0, center_x - box_width / 2))
                    ymin = round(min(0, center_y - box_height / 2))
                    xmax = round(max(width - 1, center_x + box_width - box_width / 2))
                    ymax = round(max(height - 1, center_y + box_height - box_height / 2))

                    if (xmax - xmin) < edge_threshold or (ymax - ymin) < edge_threshold:
                        pass

                    coords.append([xmin, ymin, xmax, ymax, class_id])

            if len(coords) != 0:
                self.train_data.append([img_path, coords])

    def __getitem__(self, item):
        transform_seed = random.randint(0, 2)
        img_path, coords = self.train_data[item]
        img = cv2.imread(img_path)

        if transform_seed == 0:  # 原图
            img, coords = image.resize_image_with_coords(img, self.input_size, self.input_size, coords)
            img = self.transform_common(img)

        elif transform_seed == 1:  # 饱和度调整
            img, coords = image.resize_image_with_coords(img, self.input_size, self.input_size, coords)
            img = image.exposure(img, gamma=0.5)
            img = self.transform_common(img)

        else:  # 饱和度调整
            img, coords = image.resize_image_with_coords(img, self.input_size, self.input_size, coords)
            H, S, V = cv2.split(img)
            cv2.merge([np.uint8(H), np.uint8(S * 1.5), np.uint8(V)], dst=img)
            cv2.cvtColor(src=img, dst=img, code=cv2.COLOR_HSV2BGR)
            img = self.transform_common(img)

        random.seed(int(time.time()))
        random_index = random.randint(0, len(coords) - 1)
        xmin, ymin, xmax, ymax, class_index = coords[random_index]
        img = img[ymin : ymax, xmin : xmax]
        return img, class_index

    def __len__(self):
        return len(self.train_data)

    '''
    elif transform_seed == 1:  # 明度调整 可能会引起图片颜色分离，导致RGB上结构信息丢失
        img, coords = image.resize_image_with_coords(img, self.input_size, self.input_size, coords)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(img)
        cv2.merge([np.uint8(H), np.uint8(S), np.uint8(V * 1.5)], dst=img)
        cv2.cvtColor(src=img, dst=img, code=cv2.COLOR_HSV2BGR)
        img = self.transform_common(img)
    '''