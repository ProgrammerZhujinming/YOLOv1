#---------------step1:Dataset-------------------
import torch
from YOLO_V1_DataSet import YoloV1DataSet
dataSet = YoloV1DataSet(imgs_dir="./VOC2007/Train/JPEGImages",annotations_dir="./VOC2007/Train/Annotations",ClassesFile="./VOC2007/Train/class.data")
from torch.utils.data import DataLoader
dataLoader = DataLoader(dataSet,batch_size=32,shuffle=True,num_workers=4)

#---------------step2:Model-------------------
from YOLO_V1_Model import YOLO_V1
Yolo = YOLO_V1().cuda(device=1)
Yolo.initialize_weights()

#---------------step3:LossFunction-------------------
from YOLO_V1_LossFunction import  Yolov1_Loss
loss_function = Yolov1_Loss().cuda(device=1)

#---------------step4:Optimizer-------------------
import torch.optim as optim
optimizer = optim.SGD(Yolo.parameters(),lr=3e-3,momentum=0.9,weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,20000,30000,40000],gamma=0.9)

#--------------step5:Tensorboard Feature Map------------
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import torch.nn as nn
writer = SummaryWriter('log')

def feature_map_visualize(img_data, writer):
    img_data = img_data.unsqueeze(0)
    img_grid = vutils.make_grid(img_data, normalize=True, scale_each=True)
    for i,m in enumerate(Yolo.modules()):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or \
                isinstance(m, nn.ReLU) or isinstance(m, nn.MaxPool2d) or isinstance(m, nn.AdaptiveAvgPool2d):
            img_data = m(img_data)
            img_data = img_data.permute(1, 0, 2, 3)
            img_grid = vutils.make_grid(img_data, normalize=True, scale_each=True)
            img_data = img_data.permute(1, 0, 2, 3)
            writer.add_image('feature_map', img_grid)

#---------------step6:Train-------------------
epoch = 0
while epoch <= 2000*dataSet.Classes:

    loss_sum = 0
    loss_coord = 0
    loss_confidence = 0
    loss_classes = 0
    epoch_iou = 0
    epoch_object_num = 0
    scheduler.step()
    loss_function.setLossWeight(epoch)
    
    for batch_index, batch_train in enumerate(dataLoader):
        optimizer.zero_grad()
        train_data = batch_train[0].float().cuda(device=1)
        train_data.requires_grad = True
        label_data = batch_train[1].float().cuda(device=1)
        loss = loss_function(bounding_boxes=Yolo(train_data),ground_truth=label_data)
        batch_loss = loss[0]
        loss_coord = loss_coord + loss[1]
        loss_confidence = loss_confidence + loss[2]
        loss_classes = loss_classes + loss[3]
        epoch_iou = epoch_iou + loss[4]
        epoch_object_num = epoch_object_num + loss[5]
        batch_loss.backward()
        optimizer.step()
        batch_loss = batch_loss.item()
        loss_sum = loss_sum + batch_loss
        print("batch_index : {} ; batch_loss : {}".format(batch_index, batch_loss))
    epoch = epoch + 1
    if epoch % 100 == 0:
        torch.save(Yolo.state_dict(), './YOLO_V1_' + str(epoch) + '.pth')
        writer.close()
        writer = SummaryWriter(logdir='log',filename_suffix=str(epoch) + '~' + str(epoch + 100))
    print("epoch : {} ; loss : {}".format(epoch,{loss_sum}))
    for name, layer in Yolo.named_parameters():
        writer.add_histogram(name + '_grad', layer.grad.cpu().data.numpy(), epoch)
        writer.add_histogram(name + '_data', layer.cpu().data.numpy(), epoch)
    #feature_map_visualize(batch_train[0], writer)
    writer.add_scalar('Train/Loss_sum', loss_sum, epoch)
    writer.add_scalar('Train/Loss_coord', loss_coord, epoch)
    writer.add_scalar('Train/Loss_confidenct', loss_confidence, epoch)
    writer.add_scalar('Train/Loss_classes', loss_classes, epoch)
    writer.add_scalar('Train/Epoch_iou', epoch_iou / epoch_object_num, epoch)

writer.close()
