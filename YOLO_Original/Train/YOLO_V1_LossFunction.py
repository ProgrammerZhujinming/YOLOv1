import math
import torch
import torch.nn as nn

class YOLO_V1_Loss(nn.Module):

    def __init__(self, S=7, B=2, Classes=20, l_coord=5, l_noobj=0.5, epcoh_threshold = 400):
        # 有物体的box损失权重设为l_coord,没有物体的box损失权重设置为l_noobj
        super(YOLO_V1_Loss, self).__init__()
        self.S = S
        self.B = B
        self.Classes = Classes
        self.l_coord = l_coord
        self.l_noobj = l_noobj
        self.epcoh_threshold = epcoh_threshold

    def iou(self, bounding_box, ground_box, gridX, gridY, img_size=448, grid_size=64):  # 计算两个box的IoU值
        # predict_box: [centerX, centerY, width, height]
        # ground_box : [centerX / self.grid_cell_size - indexJ,centerY / self.grid_cell_size - indexI,(xmax-xmin)/self.img_size,(ymax-ymin)/self.img_size,1,xmin,ymin,xmax,ymax,(xmax-xmin)*(ymax-ymin)
        # 1.  预处理 predict_box  变为  左上X,Y  右下X,Y  两个边界点的坐标 避免浮点误差 先还原成整数
        # 不要共用引用
        # [xmin,ymin,xmax,ymax]
        predict_box = list([0, 0, 0, 0])
        predict_box[0] = (int)(gridX + bounding_box[0] * grid_size)
        predict_box[1] = (int)(gridY + bounding_box[1] * grid_size)
        predict_box[2] = (int)(bounding_box[2] * img_size)
        predict_box[3] = (int)(bounding_box[3] * img_size)

        predict_coord = list([max(0, predict_box[0] - predict_box[2] / 2),
                              max(0, predict_box[1] - predict_box[3] / 2),
                              min(img_size - 1, predict_box[0] + predict_box[2] / 2),
                              min(img_size - 1, predict_box[1] + predict_box[3] / 2)])
        predict_Area = (predict_coord[2] - predict_coord[0]) * (predict_coord[3] - predict_coord[1])

        ground_coord = list([ground_box[5].item(), ground_box[6].item(), ground_box[7].item(), ground_box[8].item()])
        ground_Area = (ground_coord[2] - ground_coord[0]) * (ground_coord[3] - ground_coord[1])

        # 存储格式 xmin ymin xmax ymax

        # 2.计算交集的面积 左边的大者 右边的小者 上边的大者 下边的小者
        CrossLX = max(predict_coord[0], ground_coord[0])
        CrossRX = min(predict_coord[2], ground_coord[2])
        CrossUY = max(predict_coord[1], ground_coord[1])
        CrossDY = min(predict_coord[3], ground_coord[3])

        if CrossRX < CrossLX or CrossDY < CrossUY:  # 没有交集
            return 0

        interSection = (CrossRX - CrossLX) * (CrossDY - CrossUY)

        return interSection / (predict_Area + ground_Area - interSection)

    def forward(self, bounding_boxes, ground_truth, batch_size=32,grid_size=64, img_size=448):  # 输入是 S * S * ( 2 * B + Classes)
        # 定义三个计算损失的变量 正样本定位损失 样本置信度损失 样本类别损失
        loss = 0
        loss_coord = 0
        loss_confidence = 0
        loss_classes = 0
        iou_sum = 0
        object_num = 0

        mseLoss = nn.MSELoss()
        for batch in range(len(bounding_boxes)):
            for indexRow in range(self.S):  # 先行 - Y
                for indexCol in range(self.S):  # 后列 - X
                    bounding_box = bounding_boxes[batch][indexRow][indexCol]
                    predict_box_one = bounding_box [0:5]
                    predict_box_two = bounding_box [5:10]
                    ground_box = ground_truth[batch][indexRow][indexCol]
                    # 1.如果此处ground_truth不存在 即只有背景 那么两个框均为负样本
                    if round(ground_box[9].item()) == 0:  # 面积为0的grount_truth 表明此处只有背景
                        loss = loss + self.l_noobj * torch.pow(predict_box_one[4], 2) + torch.pow(predict_box_two[4], 2)
                        loss_confidence += self.l_noobj * math.pow(predict_box_one[4].item(), 2) + math.pow(predict_box_two[4].item(), 2)
                    else:
                        object_num = object_num + 1
                        predict_iou_one =  self.iou(predict_box_one, ground_box, indexCol * 64, indexRow * 64)
                        predict_iou_two = self.iou(predict_box_two, ground_box, indexCol * 64, indexRow * 64)
                        # 改进：让两个预测的box与ground box拥有更大iou的框进行拟合 让iou低的作为负样本
                        if predict_iou_one >  predict_iou_two: # 框1为正样本  框2为负样本
                            predict_box = predict_box_one
                            iou = predict_iou_one
                            no_predict_box = predict_box_two
                        else:
                            predict_box = predict_box_two
                            iou = predict_iou_two
                            no_predict_box = predict_box_one
                        # 正样本：
                        # 定位
                        loss = loss + self.l_coord * (torch.pow((ground_box[0] - predict_box[0]), 2) + torch.pow((ground_box[1] - predict_box[1]), 2) + torch.pow(torch.sqrt(ground_box[2] + 1e-8) - torch.sqrt(predict_box[2] + 1e-8), 2) + torch.pow(torch.sqrt(ground_box[3] + 1e-8) - torch.sqrt(predict_box[3] + 1e-8), 2))
                        loss_coord += self.l_coord * (math.pow((ground_box[0] - predict_box[0].item()), 2) + math.pow((ground_box[1] - predict_box[1].item()), 2) + math.pow(math.sqrt(ground_box[2] + 1e-8) - math.sqrt(predict_box[2].item() + 1e-8),2) + math.pow(math.sqrt(ground_box[3] + 1e-8) - math.sqrt(predict_box[3].item() + 1e-8), 2))
                        # 置信度
                        loss = loss + torch.pow(predict_box[4] - iou, 2)
                        loss_confidence += math.pow(predict_box[4].item() - iou, 2)
                        iou_sum = iou_sum + iou
                        # 分类
                        ground_class = ground_box[10:]
                        predict_class = bounding_box [self.B * 5:]
                        loss = loss + mseLoss(ground_class, predict_class)
                        loss_classes += mseLoss(ground_class, predict_class).item()
                        # 负样本 置信度：
                        loss = loss + self.l_noobj * torch.pow(no_predict_box[4] - 0, 2)
                        loss_confidence += math.pow(no_predict_box[4].item() - 0, 2)
        return loss / batch_size, loss_coord / batch_size, loss_confidence  / batch_size, loss_classes / batch_size, iou_sum, object_num