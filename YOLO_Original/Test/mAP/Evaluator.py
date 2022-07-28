import os
import sys
from collections import Counter
import time

import matplotlib.pyplot as plt
import numpy as np

from IPython.display import display

class Evaluator:
    def GetPascalVOCMetrics(self,
                            cfg,
                            classes, 
                            gt_boxes,
                            num_pos,
                            det_boxes):

        ret = []
        groundTruths = []
        detections = []
        
        for c in classes:
            dects = det_boxes[c]#与类别对应的 [det_box] det_box:[left, top, right, bottom, score, nameOfImage]
            gt_class = gt_boxes[c]#{nameOfImage:[ground_box]}:[left, top, right, bottom, 0]
            npos = num_pos[c]#类别c的ground_box 总数
            dects = sorted(dects, key=lambda conf: conf[4], reverse=True)#按照置信度排序
            TP = np.zeros(len(dects))
            FP = np.zeros(len(dects))
                
            for d in range(len(dects)):

                iouMax = sys.float_info.min
                if dects[d][-1] in gt_class:#det预测在 dects[d][-1](图片名) 中存在类别为c的物体
                    for j in range(len(gt_class[dects[d][-1]])):#遍历该图片拥有的ground_boxes
                        iou = Evaluator.iou(dects[d][:4], gt_class[dects[d][-1]][j][:4])
                        if iou > iouMax:
                            iouMax = iou
                            jmax = j

                    if iouMax >= cfg['iouThreshold']:#最大iou大于阈值 
                        if gt_class[dects[d][-1]][jmax][4] == 0:#如果没有被使用
                            TP[d] = 1
                            gt_class[dects[d][-1]][jmax][4] == 1

                        else:#如果已经被使用
                            FP[d] = 1
                    else:#如果最大iou值不满足阈值
                        FP[d] = 1
                else:#如果该图片名不存在c类别的ground_box，属于错误预测为c类 是FP
                    FP[d] = 1
            
            acc_FP = np.cumsum(FP)
            acc_TP = np.cumsum(TP)
            rec = acc_TP / npos
            prec = np.divide(acc_TP, (acc_FP + acc_TP))
            print("class:{} FP:{} TP:{} recall:{} pre:{}".format(c, acc_FP, acc_TP, rec, prec))
            [ap, mpre, mrec, ii] = Evaluator.CalculateAveragePrecision(rec, prec)
            print("class:{} ap:{} mpre:{} mrec:{} ii:{}".format(c, ap, mpre, mrec, ii))
            r = {
                'class': c,
                'precision': prec,
                'recall': rec,
                'AP': ap,
                'interpolated precision': mpre,
                'interpolated recall': mrec,
                'total positives': npos,
                'total TP': np.sum(TP),
                'total FP': np.sum(FP),
            }
            ret.append(r)
        return ret, classes

    @staticmethod
    def CalculateAveragePrecision(rec, prec):
        mrec = []
        mrec.append(0)
        [mrec.append(e) for e in rec]
        mrec.append(1)# recall保持单增 后接1
        mpre = []
        mpre.append(0)
        [mpre.append(e) for e in prec]
        mpre.append(0)# precision力求递减 后接0

        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])#修正 保证precision单调不增
        ii = []
        print("mpre:{}".format(mpre))
        for i in range(len(mrec) - 1):
            if mrec[i+1] != mrec[i]:
                ii.append(i + 1)#按照recall的值对数据点去重
        ap = 0
        for i in ii:
            ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
        return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]

    @staticmethod
    def iou(boxA, boxB):
        # if boxes dont intersect
        if Evaluator._boxesIntersect(boxA, boxB) is False:#判断两个Box是否有交集
            return 0
        interArea = Evaluator._getIntersectionArea(boxA, boxB)#交集区域的面积
        union = Evaluator._getUnionAreas(boxA, boxB, interArea=interArea)#拿到并集区域的面积
        # intersection over union
        iou = interArea / union
        if iou < 0:#异常值处理 可能是因为精度误差导致负数的出现
            import pdb
            pdb.set_trace()
        assert iou >= 0
        return iou

    # boxA = (Ax1,Ay1,Ax2,Ay2)
    # boxB = (Bx1,By1,Bx2,By2)
    @staticmethod
    def _boxesIntersect(boxA, boxB):
        if boxA[0] > boxB[2]:
            return False  # boxA is right of boxB
        if boxB[0] > boxA[2]:
            return False  # boxA is left of boxB
        if boxA[3] < boxB[1]:
            return False  # boxA is above boxB
        if boxA[1] > boxB[3]:
            return False  # boxA is below boxB
        return True

    @staticmethod
    def _getIntersectionArea(boxA, boxB):

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # intersection area
        return (xB - xA + 1) * (yB - yA + 1)

    @staticmethod
    def _getUnionAreas(boxA, boxB, interArea=None):
        area_A = Evaluator._getArea(boxA)
        area_B = Evaluator._getArea(boxB)
        if interArea is None:
            interArea = Evaluator._getIntersectionArea(boxA, boxB)
        return float(area_A + area_B - interArea)

    @staticmethod
    def _getArea(box):
        return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
