def iou(self, box1, box2):  # 计算两个box的IoU值
    # box: lx-左上x ly-左上y rx-右下x ry-右下y 图像向右为y 向下为x
    # 1. 获取交集的矩形左上和右下坐标
    interLX = max(box1[0],box2[0])
    interLY = max(box1[1],box2[1])
    interRX = min(box1[2],box2[2])
    interRY = min(box1[3],box2[3])

    # 2. 计算两个矩形各自的面积
    Area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    Area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 3. 不存在交集
    if interRX < interLX or interRY < interLY:
        return 0

    # 4. 计算IOU
    interSection = (interRX - interLX) * (interRY - interLY)
    return interSection / (Area1 + Area2 - interSection)

import numpy as np
# 这边要求的bounding_boxes为处理后的实际的样子
def NMS(bounding_boxes,confidence_threshold,iou_threshold):
    # boxRow : x y dx dy c
    # 1. 初步筛选,先把grid cell预测的两个bounding box取出置信度较高的那个
    boxes = []
    for boxRow in bounding_boxes:
        # grid cell预测出的两个box,含有物体的置信度没有达到阈值
        if boxRow[4] < confidence_threshold or boxRow[9] < confidence_threshold:
            continue
        # 获取物体的预测概率
        classes = boxRow[10:-1]
        class_probality_index = np.argmax(classes,axis=1)
        class_probality = classes[class_probality_index]
        # 选择拥有更大置信度的box
        if boxRow[4] > boxRow[9]:
            box = boxRow[0:4]
        else:
            box = boxRow[5:9]
        # box : x y dx dy class_probality_index class_probality
        box.append(class_probality_index)
        box.append(class_probality)
        boxes.append(box)

    # 2. 循环直到待筛选的box集合为空
    predicted_boxes = []
    while len(boxes) != 0:
        # 对box集合按照置信度从大到小排序
        boxes = sorted(boxes, key=(lambda x : [x[4]]), reverse=True)
        # 确定含有最大值信度的box会被选中
        choiced_box = boxes[0]
        predicted_boxes.append(choiced_box)
        for index in len(boxes):
            # 如果冲突的box的iou值已经大于阈值 需要丢弃
            if iou(boxes[index],choiced_box) > iou_threshold:
                boxes.pop(index)

    return predicted_boxes