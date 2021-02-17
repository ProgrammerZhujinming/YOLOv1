from torch.utils.data import Dataset
import os
import cv2
import xml.etree.ElementTree as ET
import torch
import torchvision.transforms as transforms

class YoloV1DataSet(Dataset):

    def __init__(self, imgs_dir="./Data/Train/JPEGImages", annotations_dir="./Data/Train/Annotations", img_size=448, S=7, B=2, ClassesFile="./Data/class.data"): # 图片路径、注解文件路径、图片尺寸、每个grid cell预测的box数量、类别文件
        img_names = os.listdir(imgs_dir)
        img_names.sort()
        self.transfrom = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
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
        self.ClassNameToInt = {}
        classIndex = 0
        with open(ClassesFile, 'r') as f:
            for line in f:
                line = line.replace('\n','')
                self.ClassNameToInt[line] = classIndex #根据类别名制作索引
                classIndex += 1
        self.Classes = classIndex # 一共的类别个数
        self.getGroundTruth()
        #self.getImgData()

    # PyTorch 无法将长短不一的list合并为一个Tensor
    def getGroundTruth(self):
        self.ground_truth = [[[list() for i in range(self.S)] for j in range(self.S)] for k in
                             range(len(self.img_path))]  # 根据标注文件生成ground_truth
        ground_truth_index = 0
        #self.ground_truth = [list() for k in range(len(self.img_path))] # 根据标注文件生成文件个数个ground_truth
        for annotation_file in self.annotation_path:
            ground_truth = [[list() for i in range(self.S)] for j in range(self.S)]
            #ground_truth = np.zeros(shape=(self.S, self.S, 5 * self.B + self.Classes))
            # 解析xml文件--标注文件
            tree = ET.parse(annotation_file)
            annotation_xml = tree.getroot()
            # 计算 目标尺寸 -> 原图尺寸 self.img_size * self.img_size , x的变化比例
            width = (int)(annotation_xml.find("size").find("width").text)
            scaleX = self.img_size / width
            # 计算 目标尺寸 -> 原图尺寸 self.img_size * self.img_size , y的变化比例
            height = (int)(annotation_xml.find("size").find("height").text)
            scaleY = self.img_size / height
            # 因为两次除法的误差可能比较大 这边采用除一次乘一次的方式
            # 一个注解文件可能有多个object标签，一个object标签内部包含一个bnd标签
            objects_xml = annotation_xml.findall("object")
            for object_xml in objects_xml:
                # 获取目标的名字
                class_name = object_xml.find("name").text
                if class_name not in self.ClassNameToInt: # 不属于我们规定的类
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
                # 当前物体的中心点落于 第indexI行 第indexJ列的 grid cell内
                indexI = (int)(centerY / self.grid_cell_size)
                indexJ = (int)(centerX / self.grid_cell_size)
                # 真实物体的list
                ClassIndex = self.ClassNameToInt[class_name]
                ClassList = [0 for i in range(self.Classes)]
                ClassList[ClassIndex] = 1
                ground_box = list([centerX,centerY,(xmax-xmin)/self.img_size,(ymax-ymin)/self.img_size,1,xmin,ymin,xmax,ymax,(xmax-xmin)*(ymax-ymin)])
                #增加上类别
                ground_box.extend(ClassList)
                ground_truth[indexI][indexJ].append(ground_box)

            #同一个grid cell内的多个groudn_truth，选取面积最大的两个
            for i in range(self.S):
                for j in range(self.S):
                    ground_truth[i][j].sort(key = lambda box: box[9])
                    for k in range(min(2,len(ground_truth[i][j]))):
                        self.ground_truth[ground_truth_index][i][j].append(ground_truth[i][j][k])
                    box_num = len(self.ground_truth[ground_truth_index][i][j])
                    while box_num < 2:
                        self.ground_truth[ground_truth_index][i][j].append([0 for i in range(self.B * 5 + self.Classes)])
                        box_num = box_num + 1

            ground_truth_index = ground_truth_index + 1
        self.ground_truth = torch.Tensor(self.ground_truth).float()

    # 直接预读取图片
    '''
    def getImgData(self):
        self.img_data = torch.Tensor([]).float()
        for img_index in range(len(self.img_path)):
            print("index1:{}  index2:{}".format(img_index,len(self.img_path)))
            img_data = cv2.imread(self.img_path[img_index])
            img_data = cv2.resize(img_data, (448, 448), interpolation=cv2.INTER_AREA)
            img_data = self.transfrom(img_data).resize(448, 448, 3)
            self.img_data = torch.cat((self.img_data, img_data),dim=0)
        self.img_data = torch.Tensor(self.img_data).float()
        print(self.img_data)
    '''

    def __getitem__(self, item):
        img_data = cv2.imread(self.img_path[item])
        img_data = cv2.resize(img_data, (448, 448), interpolation=cv2.INTER_AREA)
        img_data = self.transfrom(img_data)
        return img_data,self.ground_truth[item]


    def __len__(self):
        return len(self.img_path)


#YoloV1DataSet(imgs_dir="./Data/Train/JPEGImages",annotations_dir="./Data/Train/Annotations",img_size=448,S=7,B=2,ClassesFile="./Data/class.data")
