#---------------step1:Dataset-------------------
import torch
from YOLO_V1_DataSet import YoloV1DataSet
dataSet = YoloV1DataSet(imgs_dir="./VOC2007/Train/JPEGImages",annotations_dir="./VOC2007/Train/Annotations",ClassesFile="./VOC2007/Train/class.data")
from torch.utils.data import DataLoader
#dataLoader = DataLoader(dataSet,batch_size=8,shuffle=True,num_workers=4)

#---------------step2:Model-------------------
from YOLO_V1_Model import YOLO_V1
Yolo = YOLO_V1().cuda(device=0)
train_file = "YOLO_V1_5900.pth"
Yolo.load_state_dict(torch.load(train_file))

#---------------step3:LossFunction-------------------
from YOLO_V1_LossFunction import  Yolov1_Loss
loss_function = Yolov1_Loss().cuda(device=0)

#---------------step4:Optimizer-------------------
import torch.optim as optim
optimizer = optim.SGD(Yolo.parameters(),lr=3e-4,momentum=0.9,weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=100,gamma=0.95)

#--------------step5:Tensorboard Feature Map------------
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import torch.nn as nn
writer = SummaryWriter('log')

def feature_map_visualize(img_data, writer):
    img_data = img_data.unsqueeze(0)
    img_grid = vutils.make_grid(img_data, normalize=True, scale_each=True)
    for i,m in enumerate(Yolo.modules()):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.ReLU) or isinstance(m, nn.MaxPool2d) or isinstance(m, nn.AdaptiveAvgPool2d):
            img_data = m(img_data)
            img_data = img_data.permute(1, 0, 2, 3)
            img_grid = vutils.make_grid(img_data, normalize=True, scale_each=True)
            img_data = img_data.permute(1, 0, 2, 3)
            writer.add_image('feature_map', img_grid)


from tensorboardX import SummaryWriter
import torchvision.utils as vutils
epoch = int(train_file.split('.')[0].split('_')[2])
writer = SummaryWriter(logdir='log', filename_suffix=' [' + str(epoch) + '~' + str(epoch + 100) + ']')

from tqdm import tqdm
from progressbar import ProgressBar, Percentage, Bar, Timer, ETA, FileTransferSpeed

epoch = epoch + 1
while epoch <= 2000 * dataSet.Classes:

    scheduler.step()
    loss_function.setWeight(epoch)

    train_sum = int(dataSet.__len__() + 0.5)
    train_len = int(train_sum * 0.9)
    val_len = train_sum - train_len

    dataSet.shuffleData()
    train_dataSet, val_dataSet = torch.utils.data.random_split(dataSet, [train_len, val_len])
    train_loader = DataLoader(train_dataSet, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataSet, batch_size=4, shuffle=True, num_workers=0)

    train_len = train_loader.__len__()
    val_len = val_loader.__len__()
    epoch_train_loss = 0
    epoch_val_loss = 0
    epoch_train_iou = 0
    epoch_val_iou = 0
    epoch_train_object_num = 0
    epoch_val_object_num = 0
    epoch_train_loss_coord = 0
    epoch_val_loss_coord = 0
    epoch_train_loss_confidence = 0
    epoch_val_loss_confidence = 0
    epoch_train_loss_classes = 0
    epoch_val_loss_classes = 0

    with tqdm(total=train_len) as tbar:

        for batch_index, batch_train in enumerate(train_loader):
            optimizer.zero_grad()
            train_data = batch_train[0].float().cuda(device=0)
            train_data.requires_grad = True
            label_data = batch_train[1].float().cuda(device=0)
            loss = loss_function(bounding_boxes=Yolo(train_data),ground_truth=label_data)
            batch_loss = loss[0]
            epoch_train_loss_coord = epoch_train_loss_coord + loss[1]
            epoch_train_loss_confidence = epoch_train_loss_confidence + loss[2]
            epoch_train_loss_classes = epoch_train_loss_classes + loss[3]
            epoch_train_iou = epoch_train_iou + loss[4]
            epoch_train_object_num = epoch_train_object_num + loss[5]
            batch_loss.backward()
            optimizer.step()
            batch_loss = batch_loss.item()
            epoch_train_loss = epoch_train_loss + batch_loss

            tbar.set_description("train: coord_loss:{} confidence_loss:{} class_loss:{} avg_iou:{}".format(round(loss[1], 4), round(loss[2], 4), round(loss[3], 4), round(loss[4].item() / loss[5], 4)), refresh=True)
            tbar.update(1)

            #feature_map_visualize(train_data[0][0], writer)
            #print("batch_index : {} ; batch_loss : {}".format(batch_index, batch_loss))
        print("train-batch-mean loss:{} coord_loss:{} confidence_loss:{} class_loss:{} iou:{}".format(round(epoch_train_loss / train_len, 4), round(epoch_train_loss_coord / train_len, 4), round(epoch_train_loss_confidence / train_len, 4), round(epoch_train_loss_classes / train_len, 4), round(epoch_train_iou.item() / epoch_train_object_num, 4)))

    with tqdm(total=val_len) as tbar:

        for batch_index, batch_train in enumerate(val_loader):
            #optimizer.zero_grad()
            train_data = batch_train[0].float().cuda(device=0)
            #train_data.requires_grad = True  验证时计算loss 不需要依附上梯度
            label_data = batch_train[1].float().cuda(device=0)
            loss = loss_function(bounding_boxes=Yolo(train_data), ground_truth=label_data)
            batch_loss = loss[0]
            epoch_val_loss_coord = epoch_val_loss_coord + loss[1]
            epoch_val_loss_confidence = epoch_val_loss_confidence + loss[2]
            epoch_val_loss_classes = epoch_val_loss_classes + loss[3]
            epoch_val_iou = epoch_val_iou + loss[4]
            epoch_val_object_num = epoch_val_object_num + loss[5]
            #batch_loss.backward()
            #optimizer.step()
            batch_loss = batch_loss.item()
            epoch_val_loss = epoch_val_loss + batch_loss

            tbar.set_description("val: coord_loss:{} confidence_loss:{} class_loss:{} iou:{}".format(round(loss[1], 4), round(loss[2], 4), round(loss[3], 4), round(loss[4].item() / loss[5], 4)), refresh=True)
            tbar.update(1)

            # feature_map_visualize(train_data[0][0], writer)
            # print("batch_index : {} ; batch_loss : {}".format(batch_index, batch_loss))
        print("val-batch-mean loss:{} coord_loss:{} confidence_loss:{} class_loss:{} iou:{}".format(round(epoch_val_loss / val_len, 4), round(epoch_val_loss_coord / val_len, 4), round(epoch_val_loss_confidence / val_len, 4), round(epoch_val_loss_classes / val_len, 4), round(epoch_val_iou.item() / epoch_val_object_num, 4)))


    epoch = epoch + 1
    '''
    if (epoch < 1000 and epoch % 100 == 0) or epoch % 1000 == 0:
        torch.save(Yolo.state_dict(), './YOLO_V1_' + str(epoch) + '.pth')
        writer.close()
        writer = SummaryWriter(logdir='log', filename_suffix=' [' + str(epoch) + '~' + str(epoch + 1000) + ']')
    '''
    if epoch % 100 == 0:
        torch.save(Yolo.state_dict(), './YOLO_V1_' + str(epoch) + '.pth')
        writer.close()
        writer = SummaryWriter(logdir='log',filename_suffix=str(epoch) + '~' + str(epoch + 100))
    print("epoch : {} ; loss : {}".format(epoch,{epoch_train_loss}))
    for name, layer in Yolo.named_parameters():
        writer.add_histogram(name + '_grad', layer.grad.cpu().data.numpy(), epoch)
        writer.add_histogram(name + '_data', layer.cpu().data.numpy(), epoch)

    writer.add_scalar('Train/Loss_sum', epoch_train_loss, epoch)
    writer.add_scalar('Train/Loss_coord', epoch_train_loss_coord, epoch)
    writer.add_scalar('Train/Loss_confidenct', epoch_train_loss_confidence, epoch)
    writer.add_scalar('Train/Loss_classes', epoch_train_loss_classes, epoch)
    writer.add_scalar('Train/Epoch_iou', epoch_train_iou / epoch_train_object_num, epoch)

writer.close()