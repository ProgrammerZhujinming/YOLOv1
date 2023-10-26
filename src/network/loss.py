import torch
import torch.nn as nn
from utils.util import compute_acc
from utils.image import show_image

#grid_wh_idx_base = torch.ones(size=(7, 7, 2)) #[w,h]

#for h in range(7):
    #for w in range(7):
        #grid_wh_idx_base[h][w][0] = w
        #grid_wh_idx_base[h][w][1] = h

#grid_xy_idx_base.repeat((3, 1, 1, 1))

class YOLO_LOSS(nn.Module):
    

    def __init__(self, class_num=20, B=2, downsample_res=64, img_size=448, coord_loss_mode="default"):
       super(YOLO_LOSS, self).__init__()
       self.class_num = class_num
       self.img_size = img_size
       self.B = B
       self.lambda_coord = 5
       self.lambda_noonj = 0.5
       self.downsample_res = downsample_res
       self.tensor_zero = torch.Tensor([0]).to(device="cuda:0" if torch.cuda.is_available() else "cpu")
       self.tensor_imgsize = torch.Tensor([img_size - 1]).to(device="cuda:0" if torch.cuda.is_available() else "cpu")
       self.coord_loss_mode = coord_loss_mode


    def Inter_Box(self, predict_boxes, truth_boxes):
       return torch.cat([torch.max(predict_boxes[:,0], truth_boxes[:,0]).unsqueeze(1),
                     torch.max(predict_boxes[:,1], truth_boxes[:,1]).unsqueeze(1),
                     torch.min(predict_boxes[:,2], truth_boxes[:,2]).unsqueeze(1),
                     torch.min(predict_boxes[:,3], truth_boxes[:,3]).unsqueeze(1)], dim=1)


    def Circumscribed_Box(self, predict_boxes, truth_boxes):
       return torch.cat([torch.min(predict_boxes[:,0], truth_boxes[:,0]).unsqueeze(1), 
                     torch.min(predict_boxes[:,1], truth_boxes[:,1]).unsqueeze(1),
                     torch.max(predict_boxes[:,2], truth_boxes[:,2]).unsqueeze(1),
                     torch.max(predict_boxes[:,3], truth_boxes[:,3]).unsqueeze(1)], dim=1)


    def GIOU_Loss(self, predict_boxes, truth_boxes, iou, inter_areas):
       circumscribed_boxes = self.Circumscribed_Box(predict_boxes, truth_boxes)
       circumscribed_areas = (circumscribed_boxes[:,2] - circumscribed_boxes[:,0]) * (circumscribed_boxes[:,3] - circumscribed_boxes[:,1])
       
       '''
       inter_boxes = self.Inter_Box(predict_boxes, truth_boxes)
       inter_areas = torch.where((inter_boxes[0] < inter_boxes[2]) & (inter_boxes[1] < inter_boxes[3]) ,(inter_boxes[2] - inter_boxes[0]) * (inter_boxes[3] - inter_boxes[1]), 0)
       iou = inter_areas / union_areas
       '''

       union_areas = predict_boxes[:,4] + truth_boxes[:,4] - inter_areas
       giou_loss = 1 - (iou - (circumscribed_areas - union_areas) / circumscribed_areas)
       return giou_loss


    def DIOU_Loss(self, predict_boxes, truth_boxes, iou, esp=1e-6):
       '''
       inter_boxes = self.Inter_Box(predict_boxes, truth_boxes)
       inter_areas = torch.where((inter_boxes[:,0] < inter_boxes[:,2]) & (inter_boxes[:,1] < inter_boxes[:,3]) ,(inter_boxes[:,2] - inter_boxes[:,0]) * (inter_boxes[:,3] - inter_boxes[:,1]), 0)
       union_areas = predict_boxes[:,4] + truth_boxes[:,4] - inter_areas
       iou = inter_areas / union_areas
       '''

       predict_boxes_center = torch.cat([(predict_boxes[:,0] + predict_boxes[:, 2]).unsqueeze(1) / 2, (predict_boxes[:,1] + predict_boxes[:, 3]).unsqueeze(1) / 2], dim=1)
       truth_boxes_center = torch.cat([(truth_boxes[:,0] + truth_boxes[:, 2]).unsqueeze(1) / 2, (truth_boxes[:,1] + truth_boxes[:, 3]).unsqueeze(1) / 2], dim=1)
       boxes_center_distance = torch.pow(predict_boxes_center[:,0] - truth_boxes_center[:,0], 2) + torch.pow(predict_boxes_center[:,1] - truth_boxes_center[:,1], 2)

       circumscribed_boxes = self.Circumscribed_Box(predict_boxes, truth_boxes)
       circumscribed_diagonal = torch.pow(circumscribed_boxes[:,2] - circumscribed_boxes[:, 0], 2) + torch.pow(circumscribed_boxes[:,3] - circumscribed_boxes[:, 1], 2) + esp

       diou_loss = 1 - iou + (boxes_center_distance / circumscribed_diagonal)
       return diou_loss


    def CIOU_Loss(self, predict_boxes, truth_boxes, iou, esp=1e-7):
       '''
       inter_boxes = self.Inter_Box(predict_boxes, truth_boxes)
       inter_areas = torch.where((inter_boxes[:,0] < inter_boxes[:,2]) & (inter_boxes[:,1] < inter_boxes[:,3]) ,(inter_boxes[:,2] - inter_boxes[:,0]) * (inter_boxes[:,3] - inter_boxes[:,1]), 0)
       union_areas = predict_boxes[:,4] + truth_boxes[:,4] - inter_areas
       iou = inter_areas / union_areas
       '''

       predict_boxes_center = torch.cat([(predict_boxes[:,0] + predict_boxes[:, 2]).unsqueeze(1) / 2, (predict_boxes[:,1] + predict_boxes[:, 3]).unsqueeze(1) / 2], dim=1)
       truth_boxes_center = torch.cat([(truth_boxes[:,0] + truth_boxes[:, 2]).unsqueeze(1) / 2, (truth_boxes[:,1] + truth_boxes[:, 3]).unsqueeze(1) / 2], dim=1)
       boxes_center_distance = torch.pow(predict_boxes_center[:,0] - truth_boxes_center[:,0], 2) + torch.pow(predict_boxes_center[:,1] - truth_boxes_center[:,1], 2)

       circumscribed_boxes = self.Circumscribed_Box(predict_boxes, truth_boxes)
       circumscribed_diagonal = torch.pow(circumscribed_boxes[:,2] - circumscribed_boxes[:, 0], 2) + torch.pow(circumscribed_boxes[:,3] - circumscribed_boxes[:, 1], 2) + esp
       
       predict_width = predict_boxes[:, 2] - predict_boxes[:, 0]
       predict_height = predict_boxes[:, 3] - predict_boxes[:, 1]
       truth_width = truth_boxes[:, 2] - truth_boxes[:, 0]
       truth_height = truth_boxes[:, 3] - truth_boxes[:, 1]

       v = (4 / (torch.pi ** 2)) * ((torch.atan(truth_width / truth_height) - torch.atan(predict_width / predict_height)) ** 2)
       with torch.no_grad():
          alpha = v / (1 - iou + v + esp)

       ciou_loss = 1 - iou + (boxes_center_distance / circumscribed_diagonal) + (alpha * v)
       return ciou_loss


    def IoU_Loss(self, predict_boxes, truth_boxes, iou):
       '''
       LX = torch.max(first_boxes[:,0], second_boxes[:,0])
       LY = torch.max(first_boxes[:,1], second_boxes[:,1])
       RX = torch.min(first_boxes[:,2], second_boxes[:,2])
       RY = torch.min(first_boxes[:,3], second_boxes[:,3])

       inter_area = torch.where((LX < RX) & (LY < RY) ,(RX - LX) * (RY - LY), 0)
       return inter_area / (first_boxes[:,4] + second_boxes[:,4] - inter_area)
       
       inter_boxes = self.Inter_Box(predict_boxes, truth_boxes)
       inter_areas = torch.where((inter_boxes[:,0] < inter_boxes[:,2]) & (inter_boxes[:,1] < inter_boxes[:,3]) ,(inter_boxes[:,2] - inter_boxes[:,0]) * (inter_boxes[:,3] - inter_boxes[:,1]), 0)
       union_areas = predict_boxes[:,4] + truth_boxes[:,4] - inter_areas
       iou = inter_areas / union_areas
       '''
       return 1 - iou

    def IoU(self, predict_boxes, truth_boxes):
       inter_boxes = self.Inter_Box(predict_boxes, truth_boxes)
       inter_areas = torch.where((inter_boxes[:,0] < inter_boxes[:,2]) & (inter_boxes[:,1] < inter_boxes[:,3]) ,(inter_boxes[:,2] - inter_boxes[:,0]) * (inter_boxes[:,3] - inter_boxes[:,1]), 0)
       union_areas = predict_boxes[:,4] + truth_boxes[:,4] - inter_areas
       iou = inter_areas / union_areas
       return iou, inter_areas  #, union_areas

    def forward(self, predict, ground_truth, gt_mask): # predict:bs * s * s * (b * 5 + class_num) gt:x,y,w,h,c,[lx,ly,rx,ry,area],...class...
       bs, width, height, channels_num =  predict.shape

       #获取负责物体的gird对应的预测结果和gt中cell里的数据
       positive_cell_predict = torch.masked_select(predict, gt_mask).view(-1, 5 * self.B + self.class_num)
       positive_cell_gt = torch.masked_select(ground_truth, gt_mask).view(-1, 5 * self.B + self.class_num + 2)

       #获取预测的数据
       positive_cell_first_boxes = positive_cell_predict[:,0:5]
       positive_cell_second_boxes = positive_cell_predict[:,5:10]
       positive_class_prob_predict = positive_cell_predict[:,10:]

       #groundtruth 归一化的目标预测值
       gt_box_normal = positive_cell_gt[:,0:5]
       #未归一化的gt值 用于与predict计算iou 确定哪一个预测是正样本
       gt_box = positive_cell_gt[:,5:10]

       #groundtruth对应的类别概率向量
       positive_class_prob_gt = positive_cell_gt[:,10:-2]

       #拿到grid的顶点坐标        
       gird_offset_x = positive_cell_gt[:,30]
       gird_offset_y = positive_cell_gt[:,31]

       #副样本的cell
       negative_cell_predict = torch.masked_select(predict, ~gt_mask).view(-1, 5 * self.B + self.class_num)  # cells * (b * 5 + class_num)

       #计算分类正确性
       top1_acc, top5_acc = compute_acc(positive_class_prob_predict, torch.argmax(positive_class_prob_gt, dim=1))

       #1.Iou -> postive predict\negitive predict

       #1.1 把预测结果转到img_size尺度下
       positive_cell_first_boxes_center = torch.cat([(positive_cell_first_boxes[:,0] + gird_offset_x).unsqueeze(1) * self.downsample_res, (positive_cell_first_boxes[:,1] + gird_offset_y).unsqueeze(1) * self.downsample_res], dim=1)

       positive_cell_second_boxes_center = torch.cat([(positive_cell_second_boxes[:,0] + gird_offset_x).unsqueeze(1) * self.downsample_res, (positive_cell_second_boxes[:,1] + gird_offset_y).unsqueeze(1) * self.downsample_res], dim=1)

       

       positive_cell_first_boxes_wh = torch.cat([(positive_cell_first_boxes[:, 2] * self.img_size).unsqueeze(1), (positive_cell_first_boxes[:, 3] * self.img_size).unsqueeze(1)], dim=1)

       positive_cell_second_boxes_wh = torch.cat([(positive_cell_second_boxes[:, 2] * self.img_size).unsqueeze(1), (positive_cell_second_boxes[:, 3] * self.img_size).unsqueeze(1)], dim=1)


       positive_cell_first_boxes_imgsize = torch.cat([
                                                 torch.max(self.tensor_zero,(positive_cell_first_boxes_center[:,0] - positive_cell_first_boxes_wh[:,0] / 2)).unsqueeze(1),
                                                 torch.max(self.tensor_zero, (positive_cell_first_boxes_center[:,1] - positive_cell_first_boxes_wh[:,1] / 2)).unsqueeze(1),
                                                 torch.min(self.tensor_imgsize, (positive_cell_first_boxes_center[:,0] + positive_cell_first_boxes_wh[:,0] / 2)).unsqueeze(1),
                                                 torch.min(self.tensor_imgsize, (positive_cell_first_boxes_center[:,1] + positive_cell_first_boxes_wh[:,1] / 2)).unsqueeze(1),
                                                 (positive_cell_first_boxes_wh[:,0] * positive_cell_first_boxes_wh[:,1]).unsqueeze(1)], dim=1)

       positive_cell_second_boxes_imgsize = torch.cat([
                                                 torch.max(self.tensor_zero,(positive_cell_second_boxes_center[:,0] - positive_cell_second_boxes_wh[:,0] / 2)).unsqueeze(1),
                                                 torch.max(self.tensor_zero, (positive_cell_second_boxes_center[:,1] - positive_cell_second_boxes_wh[:,1] / 2)).unsqueeze(1),
                                                 torch.min(self.tensor_imgsize, (positive_cell_second_boxes_center[:,0] + positive_cell_second_boxes_wh[:,0] / 2)).unsqueeze(1),
                                                 torch.min(self.tensor_imgsize, (positive_cell_second_boxes_center[:,1] + positive_cell_second_boxes_wh[:,1] / 2)).unsqueeze(1),
                                                 (positive_cell_second_boxes_wh[:,0] * positive_cell_second_boxes_wh[:,1]).unsqueeze(1)], dim=1)

       positive_cell_first_boxes_IoU, positive_cell_first_boxes_Inter_Areas = self.IoU(positive_cell_first_boxes_imgsize, gt_box) #, positive_cell_first_boxes_Union_Areas
       positive_cell_second_boxes_IoU, positive_cell_second_boxes_Inter_Areas = self.IoU(positive_cell_second_boxes_imgsize, gt_box) #, positive_cell_second_boxes_Union_Areas

       #選兩個box中與gt有更高iou的作為正樣本
       first_box_bounding_boxes_mask = torch.where(positive_cell_first_boxes_IoU > positive_cell_second_boxes_IoU, True, False)


       #计算平均IoU
       ########### 注意 这里需要按照顺序来 原来的方式不能保证与gt顺序对应  举例 first的取的位置是 0 2 4， second的取值位置是 1 3 5，那么原方式得到的是 0 2 4 1 3 5,有问题
       positive_boxes_IoU = torch.where(first_box_bounding_boxes_mask, positive_cell_first_boxes_IoU, positive_cell_second_boxes_IoU)
       positive_boxes_Inter_Areas = torch.where(first_box_bounding_boxes_mask, positive_cell_first_boxes_Inter_Areas, positive_cell_second_boxes_Inter_Areas)
       #positive_boxes_Union_Areas = torch.where(first_box_bounpositive_boxes
       #print("ious:{} m_iou:{}".format(positive_boxes_IoU, Avg_IoU))

       #根据IoU值选择box
       positive_boxes = torch.where(first_box_bounding_boxes_mask.unsqueeze(1), positive_cell_first_boxes, positive_cell_second_boxes)
       positive_boxes_imgsize = torch.where(first_box_bounding_boxes_mask.unsqueeze(1), positive_cell_first_boxes_imgsize, positive_cell_second_boxes_imgsize)
       #positive_boxes = torch.cat([positive_first_boxes, positive_second_boxes], dim=0)
       #positive_first_boxes = torch.masked_select(positive_cell_first_boxes, first_box_bounding_boxes_mask.unsqueeze(1)).view(-1, 5)
       #positive_second_boxes = torch.masked_select(positive_cell_second_boxes, second_box_bounding_boxes_mask.unsqueeze(1)).view(-1, 5)


       #第二個box作為正樣本對應着第一個box作為負樣本 反之亦然positive_boxes_Inter_Areas
       #negative_first_boxes = torch.masked_select(positive_cell_first_boxes, ~first_box_bounding_boxes_mask.unsqueeze(1)).view(-1, 5)
       #negative_second_boxes = torch.masked_select(positive_cell_second_boxes, ~second_box_bounding_boxes_mask.unsqueeze(1)).view(-1, 5)
       negative_boxes = torch.where(~first_box_bounding_boxes_mask.unsqueeze(1), positive_cell_first_boxes, positive_cell_second_boxes)
       
       
       #负样本可以不管顺序 因为不需要回归坐标
       #negative_boxes = negative_cell_predict[:,0:10].contiguous().view(-1, 5)#副样本暂时不考虑 正样本grid内未被分配的box
       negative_boxes = torch.cat([negative_boxes, negative_cell_predict[:,0:10].contiguous().view(-1, 5)], dim=0)

       #coord_loss = mse_loss(positive_boxes[:,0:2], gt_box_normal[:,0:2]) + mse_loss(torch.sqrt(positive_boxes[:,2:4] + 1e-8), torch.sqrt(gt_box_normal[:,2:4] + 1e-8))
       #coord_loss = (torch.pow(positive_boxes[:,0:2] - gt_box_normal[:,0:2], 2).sum() + torch.pow(torch.sqrt(positive_boxes[:,2:4] + 1e-8) - torch.sqrt(gt_box_normal[:,2:4] + 1e-8), 2).sum()) / bs
       
       
       if self.coord_loss_mode == "default":
          coord_loss = (torch.pow(positive_boxes[:,0] - gt_box_normal[:,0], 2).sum() + torch.pow(positive_boxes[:,1] - gt_box_normal[:,1], 2).sum() + (torch.pow(torch.sqrt(positive_boxes[:,2] + 1e-8) - torch.sqrt(gt_box_normal[:,2] + 1e-8), 2).sum()) + (torch.pow(torch.sqrt(positive_boxes[:,3] + 1e-8) - torch.sqrt(gt_box_normal[:,3] + 1e-8), 2).sum())) / bs
       elif self.coord_loss_mode == "iou":
          coord_loss = self.IoU_Loss(positive_boxes_imgsize, gt_box, positive_boxes_IoU).sum() / bs
       elif self.coord_loss_mode == "giou":
          coord_loss = self.GIOU_Loss(positive_boxes_imgsize, gt_box, positive_boxes_IoU, positive_boxes_Inter_Areas).sum() / bs
       elif self.coord_loss_mode == "diou":
          coord_loss = self.DIOU_Loss(positive_boxes_imgsize, gt_box, positive_boxes_IoU).sum() / bs
       elif self.coord_loss_mode == "ciou":
          coord_loss = self.CIOU_Loss(positive_boxes_imgsize, gt_box, positive_boxes_IoU).sum() / bs
       else:
          raise Exception("no support the coord loss mode:{}".format(self.coord_loss_mode))

       #positive_conf_loss = torch.pow(positive_boxes[:,4] - positive_boxes_IoU, 2).sum() / bs
       positive_conf_loss = torch.pow(positive_boxes[:,4] - 1, 2).sum() / bs

       #print("p-iou:{}".format(positive_boxes[:,4].shape))

       negative_conf_loss = torch.pow(negative_boxes[:,4], 2).sum() / bs

       positive_class_loss = torch.pow(positive_class_prob_predict - positive_class_prob_gt, 2).sum() / self.class_num / bs
       '''

       coord_loss = mse_loss(positive_boxes[:,0], gt_box_normal[:,0]) + mse_loss(positive_boxes[:,1], gt_box_normal[:,1]) + mse_loss(torch.sqrt(positive_boxes[:,2] + 1e-8), torch.sqrt(gt_box_normal[:,2] + 1e-8)) + mse_loss(torch.sqrt(positive_boxes[:,3] + 1e-8), torch.sqrt(gt_box_normal[:,3] + 1e-8))

       positive_conf_loss = mse_loss(positive_boxes[:,4], positive_boxes_IoU)

       negative_conf_loss = mse_loss(negative_boxes[:,4], torch.zeros_like(negative_boxes[:,4]).to(device="cuda:0" if torch.cuda.is_available() else "cpu"))

       positive_class_loss = mse_loss(positive_class_prob_predpositive_boxes_imgsizeict, positive_class_prob_gt)
       '''

       loss = self.lambda_coord * coord_loss + positive_conf_loss + self.lambda_noonj * negative_conf_loss + positive_class_loss

       return loss, self.lambda_coord * coord_loss, positive_conf_loss, self.lambda_noonj * negative_conf_loss, positive_class_loss, top1_acc, top5_acc, positive_boxes_IoU.mean()