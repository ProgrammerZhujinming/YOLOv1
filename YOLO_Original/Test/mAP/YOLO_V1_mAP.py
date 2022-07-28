#--------------- step0: common defination -------------------------
import os
import torch
import xml.etree.ElementTree as ET
import torchvision.transforms as transforms

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

transform_common = transforms.Compose([
    transforms.ToTensor(),  # height * width * channel -> channel * height * width
    transforms.Normalize(mean=(0.408, 0.448, 0.471), std=(0.242, 0.239, 0.234))  # 归一化后.不容易产生梯度爆炸的问题
])

class_num = 20
batch_size = 32
input_size = 448
#yolo_weight_file = "./YOLO_V1_400.pth"
#yolo_param = torch.load(yolo_weight_file, map_location=torch.device("cpu"))
test_images_folder = "../../../DataSet/VOC2007+2012/Test/JPEGImages"
test_annotations_folder = "../../../DataSet/VOC2007+2012/Test/Annotations"
# ----------2. help function -----------

def getGTBoxes(GTFolder):
    files = os.listdir(GTFolder)
    files.sort()

    classes = []
    num_pos = {}
    gt_boxes = {}
    for f in files:
        nameOfImage = f.replace(".txt", "")
        fh1 = open(os.path.join(GTFolder, f), "r")

        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")

            cls = (splitLine[0])  # class
            left = float(splitLine[1])
            top = float(splitLine[2])
            right = float(splitLine[3])
            bottom = float(splitLine[4])
            one_box = [left, top, right, bottom, 0]

            if cls not in classes:
                classes.append(cls)
                gt_boxes[cls] = {}
                num_pos[cls] = 0

            num_pos[cls] += 1

            if nameOfImage not in gt_boxes[cls]:
                gt_boxes[cls][nameOfImage] = []
            gt_boxes[cls][nameOfImage].append(one_box) # gt_boxes[cls][nameOfImage]: box list

        fh1.close()
    return gt_boxes, classes, num_pos

def getDetBoxes(DetFolder):
    files = os.listdir(DetFolder)
    files.sort()

    det_boxes = {}
    for f in files:
        nameOfImage = f.replace(".txt", "")
        fh1 = open(os.path.join(DetFolder, f), "r")

        for line in fh1:
            line = line.replace("\n", "")
            if line.replace(' ', '') == '':
                continue
            splitLine = line.split(" ")

            cls = (splitLine[0])  # class
            left = float(splitLine[1])
            top = float(splitLine[2])
            right = float(splitLine[3])
            bottom = float(splitLine[4])
            score = float(splitLine[5])
            one_box = [left, top, right, bottom, score, nameOfImage]

            if cls not in det_boxes:
                det_boxes[cls] = []
            det_boxes[cls].append(one_box)

        fh1.close()
    return det_boxes

def detections(cfg, gtFolder, detFolder, savePath, show_process=True):
    gt_boxes, classes, num_pos = getGTBoxes(cfg, gtFolder)
    det_boxes = getDetBoxes(cfg, detFolder)

    evaluator = Evaluator()

    return evaluator.GetPascalVOCMetrics(cfg, classes, gt_boxes, num_pos, det_boxes)

def plot_save_result(cfg, results, classes, savePath):
    plt.rcParams['savefig.dpi'] = 80
    plt.rcParams['figure.dpi'] = 130

    acc_AP = 0
    validClasses = 0
    fig_index = 0

    for cls_index, result in enumerate(results):
        if result is None:
            raise IOError('Error: Class %d could not be found.' % classId)

        cls = result['class']
        precision = result['precision']
        recall = result['recall']
        average_precision = result['AP']
        acc_AP = acc_AP + average_precision
        mpre = result['interpolated precision']
        mrec = result['interpolated recall']
        npos = result['total positives']
        total_tp = result['total TP']
        total_fp = result['total FP']

        fig_index += 1
        plt.figure(fig_index)
        plt.plot(recall, precision, cfg['colors'][cls_index], label='Precision')
        # plt.plot(mrec, mpre, cfg['colors'][cls_index], label='Precision')
        plt.xlabel('recall')
        plt.ylabel('precision')
        ap_str = "{0:.2f}%".format(average_precision * 100)
        plt.title('Precision x Recall curve \nClass: %s, AP: %s' % (str(cls), ap_str))
        plt.legend(shadow=True)
        plt.grid()
        plt.savefig(os.path.join(savePath, cls + '.png'))
        plt.show()
        plt.pause(0.05)

    mAP = acc_AP / fig_index
    mAP_str = "{0:.2f}%".format(mAP * 100)
    print('mAP: %s' % mAP_str)

# ---------------step1:Dataset 数据集-------------------
import torch
from YOLO_Original.Test.mAP.VOC_DataSet_mAP import VOCDataSet_mAP
dataSet = VOCDataSet_mAP(imgs_path=test_images_folder, annotations_path=test_annotations_folder, classes_file="../../../DataSet/VOC2007+2012/class.data", is_train=False, class_num=class_num)

# ---------------step2:Model 模型-------------------
from YOLO_Original.Train.YOLO_V1_Model import YOLO_V1
YOLO = YOLO_V1()
#YOLO.load_state_dict(yolo_param['model'])
YOLO = YOLO.to(device=device)
YOLO.eval()

# ---------------step3:class_data 与类别相关的数据--------------
#类别索引转类别名
index_to_classname = {}
#类别名转类别索引
classname_to_index = {}
with open("../../../DataSet/VOC2007+2012/class.data", "r") as class_file:
    class_index = 0
    for class_name in class_file:
        index_to_classname[class_index] = class_name
        classname_to_index[class_name] = class_index
        class_index = class_index + 1
class_num = class_index

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
def nms(bounding_boxes, iou_threshold = 0.5, S = 7, class_num = 20, grid_size = 64, img_size = 448):
    predict_boxes = []
    final_boxes = []
    bounding_box_offset = 2
    for indexRow in range(S): # 行
        for indexCol in range(S): # 列
            #认为置信度低的那个就是无效框
            bounding_box = list([indexRow, indexCol])
            class_id = np.argmax(bounding_boxes[indexRow][indexCol][10:])
            if bounding_boxes[indexRow][indexCol][4] < bounding_boxes[indexRow][indexCol][9]:
                bounding_box.extend(bounding_boxes[indexRow][indexCol][5:10])
            else:
                bounding_box.extend(bounding_boxes[indexRow][indexCol][0:5])
            bounding_box.append(class_id)
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
    bounding_boxes = np.zeros(shape=(S, S, 5 + 1))
    for final_box in final_boxes:
        indexRow = int(final_box[0] + 0.1)
        indexCol = int(final_box[1] + 0.1)
        #print("row:{} col:{} centerX:{} centerY:{} width:{} height:{}".format(indexRow, indexCol, (final_box[2] + final_box[4]) / 2, (final_box[3] + final_box[5]) / 2, final_box[4] - final_box[2], final_box[5] - final_box[3]))
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
            ground_box = image_ground_boxes[rowIndex][colIndex]
            bounding_boxes = image_bounding_boxes[rowIndex][colIndex]
            # 先取出拥有较高置信度的预测框 ground_truth为负样本时，我们希望最大置信度不要太高  ground_truth为正样本时，我们希望最大置信度越高越好
            if bounding_boxes[4] < bounding_boxes[9]:
                bounding_box = bounding_boxes[5:]
            else:
                bounding_box = bounding_boxes[0:5]
                bounding_box.extend(bounding_boxes[10:])
            #如果此处实际是物体
            if round(ground_box[4]) == 1:
                #拿到实际物体的类别标签
                ground_object_num = ground_object_num + 1
                classIndex = np.argmax(ground_box[10:])
                #预测是物体 且 预测类别正确
                if bounding_box[4] >= confidence_threshold and classIndex == np.argmax(bounding_box[5:]):
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

def read_xml(annotation_path, class_dict, name, gt_boxes, num_boxes):
    tree = ET.parse(annotation_path)
    annotation_xml = tree.getroot()

    width = round((float)(annotation_xml.find("width").text))
    w_factor = input_size / width
    height = round((float)(annotation_xml.find("height").text))
    h_factor = input_size / height

    objects_xml = annotation_xml.findall("object")

    for object_xml in objects_xml:
        bnd_xml = object_xml.find("bndbox")
        class_name = object_xml.find("name").text
        if class_name not in class_dict:  # 不属于我们规定的类
            continue
        xmin = round((float)(bnd_xml.find("xmin").text) * w_factor)
        ymin = round((float)(bnd_xml.find("ymin").text) * h_factor)
        xmax = round((float)(bnd_xml.find("xmax").text) * w_factor)
        ymax = round((float)(bnd_xml.find("ymax").text) * h_factor)
        class_id = class_dict[class_name]

        if class_id not in gt_boxes:
            gt_boxes[class_id][name] = []
            num_boxes[class_id] = 0

        num_boxes[class_id] = num_boxes[class_id] + 1
        gt_boxes[class_id][name].append([xmin, ymin, xmax, ymax])

# ---------------step5:index 深度学习模型性能指标计算-------------------

if __name__ == '__main__':
    from tqdm import tqdm
    from torch.utils.data import DataLoader

    gt_boxes = {}  # [cls][nameOfImage]
    num_boxes = {} # [cls] 某个类别下拥有的gt数
    dataLoader = DataLoader(dataSet, batch_size=batch_size, num_workers=4)

    with tqdm(total=dataLoader.__len__()) as tbar:
        for batch_index, batch_train in enumerate(dataLoader):
            batch_train_data = batch_train[0].float().to(device=device)
            imgs_path = batch_train[2]
            batch_bounding_boxes = YOLO(batch_train_data)  # batch_size * 7 * 7 * (2 * 5 + 20)

            batch_bounding_boxes = batch_bounding_boxes.cpu().detach().numpy()

            for sample_index in range(len(batch_size)):
                img_path = imgs_path[sample_index]
                img_name = img_path.replace(".jpg", "")
                annotation_path = os.path.join(test_annotations_folder, img_name, ".xml")
                read_xml(annotation_path, classname_to_index, img_name, gt_boxes, num_boxes)
                bounding_boxes = batch_bounding_boxes[sample_index]
                bounding_boxes = nms(bounding_boxes)


            # 可能会除法0
            # tbar.set_description("recall:{} precision:{} accuracy:{}".format(recall, precision, (TP + TN) / (TP + TN + FP + FN)), refresh=True)

            tbar.update(1)
            # recall = TP / (TP + FN)
            # precision = TP / (TP + FP)



# generate data

imgs_path = os.listdir(test_images_folder)



'''
for img_path in imgs_path:
    img_name = img_path.replace(".jpg", "")
    annotation_path = os.path.join(test_annotations_folder, img_name, ".xml")
    read_xml(annotation_path, classname_to_index, img_name, gt_boxes, num_boxes)
    img_
'''


p_r = [list() for i in range(class_num)]
all_bounding_boxes = []
all_ground_boxes = []
with torch.no_grad():
    with tqdm(total=dataLoader.__len__()) as tbar:
        for batch_index, batch_train in enumerate(dataLoader):
            batch_train_data = batch_train[0].float().cuda(device=0)
            batch_ground_boxes = batch_train[1].float().cuda(device=0)
            batch_bounding_boxes = YOLO(batch_train_data)  # batch_size * 7 * 7 * (2 * 5 + 20)

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