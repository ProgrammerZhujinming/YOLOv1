#------step1: 导入网络------
from YOLO_V1_Model import YOLO_V1
YoloV1 = YOLO_V1().cuda()

#------step:2 读取权重文件------
import torch
weight_file_name = "YOLO_V1_5900.pth"
YoloV1.load_state_dict(torch.load(weight_file_name))
YoloV1.eval()

#------step:3 类别索引与类别名的映射------
class_file_name = "./VOC2007/Train/class.data"
class_index_Name = {}
classIndex = 0
with open(class_file_name, 'r') as f:
    for line in f:
        line = line.replace('\n', '')
        class_index_Name[classIndex] = line  # 根据类别名制作索引
        classIndex = classIndex + 1

#------step:4 NMS算法处理输出结果------

def iou(box_one, box_two):
    LX = max(box_one[0], box_two[0])
    LY = max(box_one[1], box_two[1])
    RX = min(box_one[2], box_two[2])
    RY = min(box_one[3], box_two[3])
    if LX >= RX or LY >= RY:
        return 0
    return (RX - LX) * (RY - LY) / ((box_one[2]-box_one[0]) * (box_one[3] - box_one[1]) + (box_two[2]-box_two[0]) * (box_two[3] - box_two[1]))

import numpy as np
def NMS(bounding_boxes,S=7,B=2,img_size=448,confidence_threshold=0.9,iou_threshold=0.3):
    bounding_boxes = bounding_boxes.cpu().detach().numpy().tolist()
    predict_boxes = []
    nms_boxes = []
    grid_size = img_size / S
    for batch in range(len(bounding_boxes)):
        for i in range(S):
            for j in range(S):
                gridX = grid_size * j
                gridY = grid_size * i
                if bounding_boxes[batch][i][j][4] < bounding_boxes[batch][i][j][9]:
                    bounding_box = bounding_boxes[batch][i][j][5:10]
                else:
                    bounding_box = bounding_boxes[batch][i][j][0:5]
                class_possible = (bounding_boxes[batch][i][j][10:])
                bounding_box.extend(class_possible)
                if bounding_box[4] < confidence_threshold:
                    continue
                centerX = (int)(gridX + bounding_box[0] * grid_size)
                centerY = (int)(gridY + bounding_box[1] * grid_size)
                width = (int)(bounding_box[2] * img_size)
                height = (int)(bounding_box[3] * img_size)
                bounding_box[0] = max(0, (int)(centerX - width / 2))
                bounding_box[1] = max(0, (int)(centerY - height / 2))
                bounding_box[2] = min(img_size - 1, (int)(centerX + width / 2))
                bounding_box[3] = min(img_size - 1, (int)(centerY + height / 2))
                predict_boxes.append(bounding_box)

        while len(predict_boxes) != 0:
            predict_boxes.sort(key=lambda box:box[4])
            assured_box = predict_boxes[0]
            temp = []
            classIndex = np.argmax(assured_box[5:])
            #print("类别索引:{}".format(classIndex))
            assured_box[4] = assured_box[4] * assured_box[5 + classIndex] #修正置信度为 物体分类准确度 × 含有物体的置信度
            assured_box[5] = classIndex
            nms_boxes.append(assured_box)
            i = 1
            while i < len(predict_boxes):
                if iou(assured_box,predict_boxes[i]) <= iou_threshold:
                    temp.append(predict_boxes[i])
                i = i + 1
            predict_boxes = temp

        return nms_boxes

#------step:5 开启摄像头获取图片并识别输出-------
import cv2
import torchvision.transforms as transforms
transform = transforms.Compose([
    transforms.ToTensor(), # height * width * channel -> channel * height * width
    transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
])
video_file_name = "detection.mp4"
capture = cv2.VideoCapture(video_file_name)
while True:
    success, img_data = capture.read()
    img_data = cv2.resize(img_data, (448, 448), interpolation=cv2.INTER_AREA)
    train_data = transform(img_data).cuda()
    train_data = train_data.unsqueeze(0)
    bounding_boxes = YoloV1(train_data)
    NMS_boxes = NMS(bounding_boxes)
    for box in NMS_boxes:
        print(box)
        img_data = cv2.rectangle(img_data, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
        img_data = cv2.putText(img_data, "class:{} confidence:{}".format(class_index_Name[box[5]], box[4]),
                               (box[0], box[1]), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 0, 255), 1)

    cv2.imshow("video_detection", img_data)
    cv2.waitKey(20)

capture.release()
cv2.destroyAllWindows()