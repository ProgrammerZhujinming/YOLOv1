# ---------------step1:Dataset 数据集-------------------
import torch
from Train.YOLO_V1_DataSet import VOCDataSet
from torch.utils.data import DataLoader
dataSet = VOCDataSet(imgs_dir="./VOC2007/Train/JPEGImages", annotations_dir="./VOC2007/Train/Annotations", ClassesFile="./VOC2007/Train/class.data")
#dataSet = VOCDataSet(imgs_dir="./VOC2007/Test/JPEGImages", annotations_dir="./VOC2007/Test/Annotations", ClassesFile="./VOC2007/Train/class.data")
dataLoader = DataLoader(dataSet, batch_size=32, num_workers=0)

ground_class_bbox = dataSet.class_bboxes #ground_truth对应的所有box

# ---------------step2:Model 模型-------------------
from Train.YOLO_V1_Model import YOLO_V1
YOLO = YOLO_V1()
YOLO.load_state_dict(torch.load('./YOLO_V1_1100.pth')['model'])
YOLO = YOLO.cuda(device=0)
YOLO.eval()

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
def nms(bounding_boxes, confidence_threshold, predict_class_bbox, iou_threshold = 0.8, S = 7, B = 1, ClassNum = 20, grid_size = 64, img_size = 448):
    predict_boxes = []
    final_boxes = []
    for indexRow in range(S): # 行
        for indexCol in range(S): # 列
            #认为置信度低的那个一定是无效框
            bounding_box = []
            if bounding_boxes[indexRow][indexCol][4] < bounding_boxes[indexRow][indexCol][9]:
                bounding_box.extend(bounding_boxes[indexRow][indexCol][5:])
            else:
                bounding_box.extend(bounding_boxes[indexRow][indexCol][0:5])
                bounding_box.extend(bounding_boxes[indexRow][indexCol][10:])
            if bounding_box[4] < confidence_threshold:#置信度高的那个也不够阈值 当成背景
                continue
            #bounding_box格式:[中心点相对grid左上角的x偏移,中心点相对grid左上角的y偏移,宽度x比例,高度y比例]
            gridX = indexCol * grid_size
            gridY = indexRow * grid_size
            centerX = (int)(gridX + bounding_box[0] * grid_size)
            centerY = (int)(gridY + bounding_box[1] * grid_size)
            widthX = (int)(bounding_box[2] * img_size)
            heightY = (int)(bounding_box[3] * img_size)
            bounding_box[0] = max(0, centerX - widthX / 2)
            bounding_box[1] = max(0, centerY - heightY / 2)
            bounding_box[2] = min(img_size - 1, centerX + widthX / 2)
            bounding_box[3] = min(img_size - 1, centerY + heightY / 2)
            predict_boxes.append(bounding_box)
    #按置信度降序排序
    while len(predict_boxes) != 0:
        predict_boxes.sort(key = lambda box:box[4], reverse=True)
        assured_box = predict_boxes[0]
        final_boxes.append(assured_box)
        predict_boxes.pop(0)
        temp_boxes = []
        #筛选掉和拥有最大值信度的框重叠度高的框
        for predict_box in predict_boxes:
            if iou(predict_box[0:4], assured_box[0:4]) < iou_threshold:
                temp_boxes.append(predict_box)
        predict_boxes = temp_boxes
    #NMS过后做成类别的list
    for final_box in final_boxes:
        predict_class_index = np.argmax(final_box[5:])
        #print("index:{} len:{}".format(predict_class_index,len(predict_class_bbox)))
        predict_class_bbox[predict_class_index].append(final_box)

ground_object_num = 0
confidence_object_num = 0
iou_object_num = 0
#计算 TP FP
import sys
def deepLearningIndex(TP, FP, iou_threshold=0.1, S = 7, B = 2):
    # 深度学习指标统计
    # 目标检测领域，关注的往往多是正样本的预测情况，因此精确率与召回率总是偏向关注正样本
    # 召回率：有多少正样本被预测出来了 = 预测出来的真的正样本 / 总的正样本数量。  TP / (TP + FN)
    # 精确率：预测出来的结果中，有多少是真的正样本。 TP / TP + FP
    # TP FP TN FN ：检测结果正负 + 预测结果
    global ground_object_num
    global confidence_object_num
    global iou_object_num
    # 遍历每一个类------
    for class_index in range(classNum):
        TP_mark = np.zeros(shape=len(predict_class_bbox[class_index]))
        FP_mark = np.zeros(shape=len(predict_class_bbox[class_index]))
        ground_bbox_choose = np.zeros(shape=len(ground_class_bbox[class_index]))

        for bounding_index in range(len(predict_class_bbox[class_index])):
            max_iou = sys.float_info.min
            bounding_box = predict_class_bbox[bounding_index][ground_index]
            for ground_index in range(len(ground_class_bbox[class_index])):
                ground_box = ground_class_bbox[classIndex][ground_index]
                now_iou = iou(bounding_box[0:4], ground_box[5:9])
                if now_iou > max_iou:
                    max_iou = now_iou
                    max_iou_index = ground_index
            if ground_bbox_choose[max_iou_index] == 0:  # 与预测框有用最大iou的ground_truth没有被选择过
                ground_bbox_choose[max_iou_index] = 1
                # 传入进来的box已经满足置信度够阈值了
                if max_iou > iou_threshold and np.argmax(bounding_box[5:]) == np.argmax(ground_box[10:]):
                    TP_mark[bounding_index] = 1
                else:
                    FP_mark[bounding_index] = 1
            else:
                FP_mark[bounding_index] = 1
            TP = TP_mark.sum()
            FP = FP_mark.sum()


# ---------------step5:index 深度学习模型性能指标计算-------------------
from tqdm import tqdm
p_r = [list() for i in range(classNum)]
all_bounding_boxes = []

with torch.no_grad():
    with tqdm(total=dataLoader.__len__()) as tbar:
        for batch_index, batch_train in enumerate(dataLoader):
            batch_train_data = batch_train[0].float().cuda(device=0)
            batch_ground_boxes = batch_train[1].float().cuda(device=0)
            batch_bounding_boxes = YOLO(batch_train_data)  # batch_size * 7 * 7 * (2 * 5 + 20)

            batch_bounding_boxes = batch_bounding_boxes.cpu().detach().numpy()
            batch_ground_boxes = batch_ground_boxes.cpu().detach().numpy()

            for image_bounding_boxes in batch_bounding_boxes:
                #bounding_boxes = nms(image_bounding_boxes)
                #all_bounding_boxes.append(bounding_boxes)
                all_bounding_boxes.append(image_bounding_boxes)

            #for image_ground_truth in batch_ground_boxes:
                #all_ground_boxes.append(image_ground_truth)

            tbar.update(1)

iou_threshold = 0.45
for confidence_cycle in range(50, 95, 5):
    # 不同类别的p-r曲线绘制需要的数据
    confidence_threshold = confidence_cycle / 100
    # 预测为正样本，且实际为正样本
    TP = [0 for i in range(classNum)]
    # 预测为负样本，且实际为正样本
    FP = [0 for i in range(classNum)]
    # 初始化预测框
    predict_class_bbox = [list() for index in range(classNum)]

    with tqdm(total=len(all_bounding_boxes)) as tbar:
        # 使用不同的confidence阈值筛选bounding box
        for sample_index in range(len(all_bounding_boxes)):
            bounding_boxes = all_bounding_boxes[sample_index]
            nms(bounding_boxes, confidence_threshold, predict_class_bbox)
            tbar.update(1)

        #deepLearningIndex(bounding_boxes, confidence_threshold, TP, FP)
        # 遍历每一个类------
        for class_index in range(classNum):
            TP_mark = np.zeros(shape=len(predict_class_bbox[class_index]))
            FP_mark = np.zeros(shape=len(predict_class_bbox[class_index]))
            ground_bbox_choose = np.zeros(shape=len(ground_class_bbox[class_index]))

            for bounding_index in range(len(predict_class_bbox[class_index])):
                max_iou = sys.float_info.min
                bounding_box = predict_class_bbox[class_index][bounding_index]
                for ground_index in range(len(ground_class_bbox[class_index])):
                    ground_box = ground_class_bbox[class_index][ground_index]
                    now_iou = iou(bounding_box[0:4], ground_box[5:9])
                    if now_iou > max_iou:
                        max_iou = now_iou
                        max_iou_index = ground_index
                if ground_bbox_choose[max_iou_index] == 0:  # 与预测框拥有最大iou的ground_truth没有被选择过
                    # 传入进来的box已经满足置信度够阈值了
                    if max_iou > iou_threshold and np.argmax(bounding_box[5:]) == np.argmax(ground_box[10:]):
                        TP_mark[bounding_index] = 1
                        ground_bbox_choose[max_iou_index] = 1
                    else:
                        FP_mark[bounding_index] = 1
                else:
                    FP_mark[bounding_index] = 1
            TP = TP_mark.sum()
            FP = FP_mark.sum()
            precision = TP / (TP + FP)
            recall = TP / len(ground_class_bbox[class_index])
            print("class:{} TP:{} FP:{} sum:{} precision:{} recall:{}".format(IndexToClassName[class_index], TP, FP, len(ground_class_bbox[class_index]), precision, recall))
            p_r[class_index].append([recall, precision])


print("ground_object_num:{}".format(ground_object_num))
print("confidence_object_num:{}".format(confidence_object_num))
print("iou_object_num:{}".format(iou_object_num))
# ----------step6:pr曲线绘制与mAP计算 pr曲线 以recall为横 以precision为纵----------
import numpy as np
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
    p_r[classIndex].append([1, 0])
    p_r[classIndex].append([0, 1])
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