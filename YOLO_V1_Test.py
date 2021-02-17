# 网络加载
from YOLO_v1_Model import YOLO_V1
Yolo_V1 = YOLO_V1()
import torch
Yolo_V1.load_state_dict(torch.load('./YOLO_V1_41.pth'))
Yolo_V1 = Yolo_V1.cuda()
# 类别与索引转换
IndexToClassName = {}
with open("./VOC2007/Train/class.data","r") as f:
    index = 0
    for line in f:
        IndexToClassName[index] = line
        index = index + 1

def iou(box_one, box_two):
    LX = max(box_one[0], box_two[0])
    LY = max(box_one[1], box_two[1])
    RX = min(box_one[2], box_two[2])
    RY = min(box_one[3], box_two[3])
    if LX >= RX or LY >= RY:
        return 0
    return (RX - LX) * (RY - LY) / ((box_one[2]-box_one[0]) * (box_one[3] - box_one[1]) + (box_two[2]-box_two[0]) * (box_two[3] - box_two[1]))

# NMS算法
'''
def NMS(bounding_boxes,S=7,B=2,img_size=448,confidence_threshold=0.5,iou_threshold=0.7):
    bounding_boxes = bounding_boxes.cpu().detach().numpy().tolist()
    predict_boxes = []
    nms_boxes = []
    grid_size = img_size / S
    for batch in range(len(bounding_boxes)):
        for i in range(S):
            for j in range(S):
                gridX = grid_size * i
                gridY = grid_size * j
                if bounding_boxes[batch][i][j][4] < bounding_boxes[batch][i][j][9]:
                    bounding_box = bounding_boxes[batch][i][j][5:10]
                else:
                    bounding_box = bounding_boxes[batch][i][j][0:5]
                bounding_box.extend(bounding_boxes[batch][i][j][10:])
                if bounding_box[4] >= confidence_threshold:
                    predict_boxes.append(bounding_box)
                centerX = (int)(gridX + bounding_box[0] * grid_size)
                centerY = (int)(gridY + bounding_box[1] * grid_size)
                width = (int)(bounding_box[2] * grid_size)
                height = (int)(bounding_box[3] * grid_size)
                bounding_box[0] = max(0, (int)(centerX - width / 2))
                bounding_box[1] = max(0, (int)(centerY - height / 2))
                bounding_box[2] = min(img_size - 1, (int)(centerX + width / 2))
                bounding_box[3] = min(img_size - 1, (int)(centerY + height / 2))

        while len(predict_boxes) != 0:
            predict_boxes.sort(key=lambda box:box[4])
            temp = []
            nms_boxes.append(predict_boxes[0])
            i = 1
            while i < len(predict_boxes):
                if iou(predict_boxes[0],predict_boxes[i]) <= iou_threshold:
                    temp.append(predict_boxes[i])
                i = i + 1
            predict_boxes = temp

        return nms_boxes
'''
import numpy as np
def NMS(bounding_boxes,S=7,B=2,img_size=448,confidence_threshold=0.5,iou_threshold=0.7):
    bounding_boxes = bounding_boxes.cpu().detach().numpy().tolist()
    predict_boxes = []
    nms_boxes = []
    grid_size = img_size / S
    for batch in range(len(bounding_boxes)):
        for i in range(S):
            for j in range(S):
                gridX = grid_size * i
                gridY = grid_size * j
                if bounding_boxes[batch][i][j][4] < bounding_boxes[batch][i][j][9]:
                    bounding_box = bounding_boxes[batch][i][j][5:10]
                else:
                    bounding_box = bounding_boxes[batch][i][j][0:5]
                bounding_box.extend(bounding_boxes[batch][i][j][10:])
                if bounding_box[4] >= confidence_threshold:
                    predict_boxes.append(bounding_box)
                centerX = (int)(gridX + bounding_box[0] * grid_size)
                centerY = (int)(gridY + bounding_box[1] * grid_size)
                width = (int)(bounding_box[2] * grid_size)
                height = (int)(bounding_box[3] * grid_size)
                bounding_box[0] = max(0, (int)(centerX - width / 2))
                bounding_box[1] = max(0, (int)(centerY - height / 2))
                bounding_box[2] = min(img_size - 1, (int)(centerX + width / 2))
                bounding_box[3] = min(img_size - 1, (int)(centerY + height / 2))

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


# 读取测试数据
import cv2
import torch
test_dir = "./VOC2007/Train/JPEGImages/000005.jpg"
img_data = cv2.imread(test_dir)
img_data = cv2.resize(img_data,(448,448),interpolation=cv2.INTER_AREA)
train_data = torch.Tensor(img_data).float().cuda()
train_data = train_data.resize(1,448,448,3)
bounding_boxes = Yolo_V1(train_data)
NMS_boxes = NMS(bounding_boxes)

for box in NMS_boxes:
    print(box)
    img_data = cv2.rectangle(img_data, (box[0],box[1]),(box[2],box[3]),(0,255,0),1)
    img_data = cv2.putText(img_data, "class:{} confidence:{}".format(IndexToClassName[box[5]],box[4]),(box[0],box[1]),cv2.FONT_HERSHEY_PLAIN,0.5,(0,0,255),1)

cv2.imshow("local",img_data)
cv2.waitKey()
