# ---------------step1:Dataset 数据集-------------------
import torch
from YOLO_V1_DataSet import YoloV1DataSet
from torch.utils.data import DataLoader

dataSet = YoloV1DataSet(imgs_dir="./VOC2007/Test/JPEGImages", annotations_dir="./VOC2007/Test/Annotations", ClassesFile="./VOC2007/Train/class.data")
dataLoader = DataLoader(dataSet, batch_size=32, num_workers=4)

# ---------------step2:Model 模型-------------------
from YOLO_V1_Model import YOLO_V1

YOLO = YOLO_V1()
YOLO.load_state_dict(torch.load('./YOLO_V1_5900.pth'))
Yolo = YOLO.cuda(device=0)

# ---------------step3:class_data 与类别相关的数据--------------
#类别索引转类别名
IndexToClassName = {}
#类别名转类别索引
ClassNameToIndex = {}
with open("./VOC2007/Train/class.data", "r") as classFile:
    classIndex = 0
    for className in classFile:
        IndexToClassName[classIndex] = className
        ClassNameToIndex[className] = classIndex
        classIndex = classIndex + 1
classNum = classIndex

# --------------step4:help functions 协助函数-----------
#计算box的面积 box数据格式:[xmin, ymin, xmax, ymax]
def boxArea(box):
    return (box[2] - box[0]) * (box[3] - box[1])

#计算两个box的iou值 box数据格式:[xmin, ymin, xmax, ymax]
def iou(box_one, box_two):
    box_one_area = boxArea(box_one)
    box_two_area = boxArea(box_two)
    #交集框
    inter_box = [
                    max(box_one[0], box_two[0]),
                    max(box_one[1], box_two[1]),
                    min(box_one[2], box_two[2]),
                    min(box_one[3], box_two[3])
                ]

    if inter_box[0] > inter_box[2] or inter_box[1] > inter_box[3]:
        return 0

    inter_box_area = boxArea(inter_box)
    return inter_box_area / (box_one_area + box_two_area)

#对每一张图的bounding_box做NMS 处理后的格式为 S * S * [xmin, ymin, xmax, ymax, confidence, class...]
import numpy as np
def nms(bounding_boxes, iou_threshold = 0.8, S = 7, B = 1, ClassNum = 20, grid_size = 64, img_size = 448):
    predict_boxes = []
    final_boxes = []
    bounding_box_offset = 2
    for indexRow in range(S): # 行
        for indexCol in range(S): # 列
            #认为置信度低的那个就是无效框
            bounding_box = list([indexRow, indexCol])
            if bounding_boxes[indexRow][indexCol][4] < bounding_boxes[indexRow][indexCol][9]:
                bounding_box.extend(bounding_boxes[indexRow][indexCol][5:])
            else:
                bounding_box.extend(bounding_boxes[indexRow][indexCol][0:5])
                bounding_box.extend(bounding_boxes[indexRow][indexCol][10:])
            #bounding_box格式:[中心点相对grid左上角的x偏移,中心点相对grid左上角的y偏移,宽度x比例,高度y比例]
            gridX = indexCol * grid_size
            gridY = indexRow * grid_size
            centerX = (int)(gridX + bounding_box[bounding_box_offset + 0] * grid_size)
            centerY = (int)(gridY + bounding_box[bounding_box_offset + 1] * grid_size)
            widthX = (int)(bounding_box[bounding_box_offset + 2] * img_size)
            heightY = (int)(bounding_box[bounding_box_offset + 3] * img_size)
            bounding_box[bounding_box_offset + 0] = max(0, centerX - widthX / 2)
            bounding_box[bounding_box_offset + 1] = max(0, centerY - heightY / 2)
            bounding_box[bounding_box_offset + 2] = min(img_size - 1, centerX + widthX / 2)
            bounding_box[bounding_box_offset + 3] = min(img_size - 1, centerY + heightY / 2)
            predict_boxes.append(bounding_box)
    #按置信度降序排序
    while len(predict_boxes) != 0:
        predict_boxes.sort(key = lambda box:box[bounding_box_offset + 4], reverse=True)
        assured_box = predict_boxes[0]
        final_boxes.append(assured_box)
        predict_boxes.pop(0)
        temp_boxes = []
        #筛选掉和拥有最大值信度的框重叠度高的框
        for predict_box in predict_boxes:
            if iou(predict_box[bounding_box_offset:], assured_box[bounding_box_offset:]) < iou_threshold:
                temp_boxes.append(predict_box)
        predict_boxes = temp_boxes
    #重新生成NMS过后的bounding_boxes
    bounding_boxes = np.zeros(shape=(S, S, B * 5 + ClassNum))
    for final_box in final_boxes:
        indexRow = int(final_box[0] + 0.1)
        indexCol = int(final_box[1] + 0.1)
        print("row:{} col:{} centerX:{} centerY:{} width:{} height:{}".format(indexRow, indexCol, (final_box[2] + final_box[4]) / 2, (final_box[3] + final_box[5]) / 2, final_box[4] - final_box[2], final_box[5] - final_box[3]))
        bounding_boxes[indexRow][indexCol] = np.array(final_box[2:])
    return bounding_boxes

ground_object_num = 0
confidence_object_num = 0
iou_object_num = 0
#对每一张图片累计计算 TP FP TN FN
def deepLearningIndex(bounding_boxes, ground_boxes, confidence_threshold, TP, FP, TN, FN, iou_threshold=0.1, S = 7, B = 2):
    # 深度学习指标统计
    # 目标检测领域，关注的往往多是正样本的预测情况，因此精确率与召回率总是偏向关注正样本
    # 召回率：有多少正样本被预测出来了 = 预测出来的真的正样本 / 总的正样本数量。  TP / (TP + FN)
    # 精确率：预测出来的结果中，有多少是真的正样本。 TP / TP + FP
    # TP FP TN FN ：检测结果正负 + 预测结果
    global ground_object_num
    global confidence_object_num
    global iou_object_num
    #拿到一副图像的bounding_box
    image_bounding_boxes = bounding_boxes.tolist()
    image_ground_boxes = ground_boxes.tolist()
    for rowIndex in range(S):
        for colIndex in range(S):
            # 拿到ground_truth
            ground_box = image_ground_boxes[rowIndex][colIndex][0]
            bounding_boxes = image_bounding_boxes[rowIndex][colIndex]
            # 先取出拥有较高置信度的预测框 ground_truth为负样本时，我们希望最大置信度不要太高  ground_truth为正样本时，我们希望最大置信度越高越好
            if bounding_boxes[4] < bounding_boxes[9]:
                bounding_box = bounding_boxes[5:]
            else:
                bounding_box = bounding_boxes[0:5]
                bounding_box.extend(bounding_boxes[10:])
            #如果此处实际是物体
            if (int)(ground_box[4] + 0.1) == 1:
                #拿到实际物体的类别标签
                ground_object_num = ground_object_num + 1
                classIndex = np.argmax(ground_box[10:])
                #预测是物体 且 预测类别正确
                if bounding_box[4] >= confidence_threshold: #and classIndex == np.argmax(bounding_box[5:]):
                    confidence_object_num = confidence_object_num + 1
                    #iou大于阈值
                    if iou(bounding_box[0:4], ground_box[5:9]) >= iou_threshold:
                        iou_object_num = iou_object_num + 1
                        TP[classIndex] = TP[classIndex] + 1
                    #iou值不够大
                    else:
                        TN[classIndex] = TN[classIndex] + 1
                #若做了负预测
                else:
                    FN[classIndex] = FN[classIndex] + 1
            #如果此处实际是背景
            else:
                # 拿到预测的物体的标签
                classIndex = np.argmax(bounding_box[5:])
                #错把背景分类为物体 实际是背景,预测为物体 即FP 所有的物体在此处都是把背景预测为物体
                if bounding_box[4] >= confidence_threshold:
                    for FP_Index, value in enumerate(FP):
                        FP[FP_Index] = value + 1
                #把背景预测为背景 如何知道值神对谁的TN呢？？？
                else:
                    TN[classIndex] = TN[classIndex] + 1

# ---------------step5:index 深度学习模型性能指标计算-------------------
from tqdm import tqdm
p_r = [list() for i in range(classNum)]
all_bounding_boxes = []
all_ground_boxes = []
with torch.no_grad():
    with tqdm(total=dataLoader.__len__()) as tbar:
        for batch_index, batch_train in enumerate(dataLoader):
            batch_train_data = batch_train[0].float().cuda(device=0)
            batch_ground_boxes = batch_train[1].float().cuda(device=0)
            batch_bounding_boxes = Yolo(batch_train_data)  # batch_size * 7 * 7 * (2 * 5 + 20)

            batch_bounding_boxes = batch_bounding_boxes.cpu().detach().numpy()
            batch_ground_boxes = batch_ground_boxes.cpu().detach().numpy()

            for image_bounding_boxes in batch_bounding_boxes:
                bounding_boxes = nms(image_bounding_boxes)
                all_bounding_boxes.append(bounding_boxes)

            for image_ground_truth in batch_ground_boxes:
                all_ground_boxes.append(image_ground_truth)

            # 可能会除法0
            # tbar.set_description("recall:{} precision:{} accuracy:{}".format(recall, precision, (TP + TN) / (TP + TN + FP + FN)), refresh=True)

            tbar.update(1)
            # recall = TP / (TP + FN)
            # precision = TP / (TP + FP)

for confidence_cycle in range(5, 95, 30):
    # 不同类别的p-r曲线绘制需要的数据
    confidence_threshold = confidence_cycle / 100
    # 预测为正样本，且实际为正样本
    TP = [0 for i in range(classNum)]
    # 预测为负样本，且实际为正样本
    FP = [0 for i in range(classNum)]
    # 预测为负样本，且实际为负样本
    TN = [0 for i in range(classNum)]
    # 预测为正样本，且实际为负样本
    FN = [0 for i in range(classNum)]

    with tqdm(total=len(all_bounding_boxes)) as tbar:

        for sample_index in range(len(all_bounding_boxes)):
            bounding_boxes = all_bounding_boxes[sample_index]
            ground_boxes = all_ground_boxes[sample_index]

            deepLearningIndex(bounding_boxes, ground_boxes, confidence_threshold, TP, FP, TN, FN)

            for classIndex in range(classNum):
                if TP[classIndex] + FN[classIndex] == 0:
                    recall = 0
                else:
                    recall = TP[classIndex] / (TP[classIndex] + FN[classIndex])
                if TP[classIndex] + FP[classIndex] == 0:
                    precision = 0
                else:
                    precision = TP[classIndex] / (TP[classIndex] + FP[classIndex])
                # accuracy = (TP[index] + TN[index]) / (TP[index] + TN[index] + FP[index] + FN[index])
                p_r[classIndex].append([recall, precision])

print(TP)
print(FP)
print(TN)
print(FN)
print(ground_object_num)
print(confidence_object_num)
print(iou_object_num)
# ----------step6:pr曲线绘制与mAP计算 pr曲线 以recall为横 以precision为纵----------
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

x = []
y = []
AP = []
classNameList = []
plt.title('voc class precision-recall')
plt.xlabel('recall')
plt.ylabel('precision')
# 遍历每一个类别
for classIndex in range(len(p_r)):
    x = []
    y = []
    # p-r按照p排序 计算mAP
    p_r[classIndex].sort(key=lambda pr:pr[0])
    classAP = 0
    pre_recall = 0
    pre_precision = 1
    classNameList.append(IndexToClassName[classIndex])
    # 遍历每一个点
    for prPoint in p_r[classIndex]:
        x.append(prPoint[0])
        y.append(prPoint[1])
        classAP = classAP + (prPoint[0] - pre_recall) * (pre_precision + prPoint[1]) / 2
        pre_recall = prPoint[0]
        pre_precision = prPoint[1]
    #x = np.ndarray(x, dtype=np.float64)
    #y = np.ndarray(y, dtype=np.float64)
    # p-r曲线绘制
    #print("x:{} y:{} label:{}".format(x, y, IndexToClassName[classIndex]))
    plt.plot(x, y, linestyle="solid")
    plt.legend([IndexToClassName[classIndex]])
    plt.show()
    AP.append(classAP)
    plt.clf()
#plt.legend(classNameList)


for classIndex in range(0, classNum, 1):
    print("class:{} AP:{}".format(IndexToClassName[classIndex], AP[classIndex]))
mAP = np.mean(AP)
print("voc-mAP:{}".format(mAP))