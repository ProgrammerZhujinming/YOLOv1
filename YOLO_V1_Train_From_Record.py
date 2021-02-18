import torch
from YOLO_V1_DataSet import YoloV1DataSet
dataSet = YoloV1DataSet()
from torch.utils.data import DataLoader
dataLoader = DataLoader(dataSet,batch_size=32,shuffle=True,num_workers=4)

from YOLO_v1_Model import YOLO_V1
Yolo = YOLO_V1().cuda(device=1)

#接續訓練的文件名
train_file = "YOLO_V1_100.pth"
Yolo.load_state_dict(torch.load(train_file))
Yolo.cuda()

from YOLO_V1_LossFunction import  Yolov1_Loss
loss_function = Yolov1_Loss().cuda()

import torch.optim as optim

optimizer = optim.SGD(Yolo.parameters(),lr=3e-4)

from tensorboardX import SummaryWriter
import torchvision.utils as vutils
writer = SummaryWriter('log')

import torch.nn as nn

def feature_map_visualize(img_data, writer):
    img_data = img_data.unsqueeze(0)
    img_grid = vutils.make_grid(img_data, normalize=True, scale_each=True)
    for i,m in enumerate(Yolo.modules()):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or \
                isinstance(m, nn.ReLU) or isinstance(m, nn.MaxPool2d) or isinstance(m, nn.AdaptiveAvgPool2d):
            img_data = m(img_data)
            img_data = img_data.permute(1,0,2,3)
            img_grid = vutils.make_grid(img_data, normalize=True, scale_each=True)
            img_data = img_data.permute(1, 0, 2, 3)
            writer.add_image('feature_map', img_grid)

loss_sum = 0
epoch = int(train_file.split('_')[2].split('.')[0]) + 1
while epoch <= (2000*dataSet.Classes):
    loss_sum = 0
    loss_coord = 0
    loss_confidence = 0
    loss_classes = 0
    epoch_iou = 0
    epoch_object_num = 0
    for batch_index, batch_train in enumerate(dataLoader):
        train_data = torch.Tensor(batch_train[0]).float().cuda()
        train_data.requires_grad = True
        label_data = torch.Tensor(batch_train[1]).float().cuda()
        loss = loss_function(bounding_boxes=Yolo(train_data),ground_truth=label_data)
        loss_coord += loss[0].item()
        loss_confidence += loss[1].item()
        loss_classes += loss[2].item()
        batch_loss = loss[0] + loss[1] + loss[2]
        epoch_iou += loss[3]
        epoch_object_num += loss[4]
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        loss_sum += batch_loss.item()
        # feature_map_visualize(train_data[0], writer)
        print("batch_index : {} ; batch_loss : {}".format(batch_index,batch_loss.item()))
    epoch += 1
    if epoch != 0 and epoch % 100 == 0:
        torch.save(Yolo.state_dict(), './YOLO_V1_' + str(epoch) + '.pth')
    print("epoch : {} ; loss : {}".format(epoch,{loss_sum}))
    writer.add_scalar('Train/Loss_sum', loss_sum, epoch)
    writer.add_scalar('Train/Loss_coord', loss_coord, epoch)
    writer.add_scalar('Train/Loss_confidenct', loss_confidence, epoch)
    writer.add_scalar('Train/Loss_classes', loss_classes, epoch)
    writer.add_scalar('Train/Epoch_iou', epoch_iou / epoch_object_num, epoch)
writer.close()
