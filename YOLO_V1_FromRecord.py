#---------------step1:Dataset-------------------
import torch
from YOLO_V1_DataSet import VOCDataSet
dataSet = VOCDataSet(imgs_dir="./VOC2007/Train/JPEGImages",annotations_dir="./VOC2007/Train/Annotations",ClassesFile="./VOC2007/Train/class.data")
from torch.utils.data import DataLoader

#---------------step2:Model-------------------
from YOLO_V1_Model import YOLO_V1
YOLO = YOLO_V1().cuda(device=0)
weight_file = "./YOLO_V1_100.pth"
param_dict = torch.load(weight_file)
YOLO.load_state_dict(param_dict['model'])

#---------------step3:LossFunction-------------------
from YOLO_V1_LossFunction import YOLO_V1_Loss
loss_function = YOLO_V1_Loss().cuda(device=0)

#---------------step4:Optimizer-------------------
import torch.optim as optim
#optimizer_Adam = optim.Adam(Yolo.parameters(),lr=3e-4,weight_decay=0.0005)
optimizer_Adam = param_dict['optim']
#使用余弦退火动态调整学习率
#lr_reduce_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer_Adam , T_max=10, eta_min=1e-4, last_epoch=-1)

#--------------step5:Tensorboard Feature Map------------
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import torch.nn as nn

def feature_map_visualize(img_data, writer):
    img_data = img_data.unsqueeze(0)
    img_grid = vutils.make_grid(img_data, normalize=True, scale_each=True)
    for i,m in enumerate(YOLO.modules()):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or \
                isinstance(m, nn.ReLU) or isinstance(m, nn.MaxPool2d) or isinstance(m, nn.AdaptiveAvgPool2d):
            img_data = m(img_data)
            x1 = img_data.transpose(0,1)
            img_grid = vutils.make_grid(x1, normalize=True, scale_each=True)
            writer.add_image('feature_map_' + str(i), img_grid)

#---------------step6:Train-------------------
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
epoch = param_dict['epoch']
writer = SummaryWriter(logdir='./log', filename_suffix=' [' + str(epoch) + '~' + str(epoch + 100) + ']')

from tqdm import tqdm

while epoch <= 200 * dataSet.classNum:

    #loss_function.setWeight(epoch)

    train_sum = int(dataSet.__len__() + 0.5)
    train_len = int(train_sum * 0.9)
    val_len = train_sum - train_len

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

    #dataSet.shuffleData()
    train_dataSet, val_dataSet = torch.utils.data.random_split(dataSet, [train_len, val_len])

    train_loader = DataLoader(train_dataSet, batch_size=64, shuffle=True, num_workers=0)
    train_len = train_loader.__len__()
    YOLO.train()
    with tqdm(total=train_len) as tbar:

        for batch_index, batch_train in enumerate(train_loader):

            train_data = batch_train[0].float().cuda(device=0)
            #train_data.requires_grad = True
            label_data = batch_train[1].float().cuda(device=0)
            loss = loss_function(bounding_boxes=YOLO(train_data),ground_truth=label_data)
            batch_loss = loss[0]
            epoch_train_loss_coord = epoch_train_loss_coord + loss[1]
            epoch_train_loss_confidence = epoch_train_loss_confidence + loss[2]
            epoch_train_loss_classes = epoch_train_loss_classes + loss[3]
            epoch_train_iou = epoch_train_iou + loss[4]
            epoch_train_object_num = epoch_train_object_num + loss[5]
            batch_loss.backward()
            optimizer_Adam.step()
            optimizer_Adam.zero_grad()
            batch_loss = batch_loss.item()
            epoch_train_loss = epoch_train_loss + batch_loss

            tbar.set_description("train: coord_loss:{} confidence_loss:{} class_loss:{} avg_iou:{}".format(round(loss[1], 4), round(loss[2], 4), round(loss[3], 4), round(loss[4] / loss[5], 4)), refresh=True)
            tbar.update(1)

            #feature_map_visualize(train_data[0][0], writer)
            #print("batch_index : {} ; batch_loss : {}".format(batch_index, batch_loss))
        print("train-batch-mean loss:{} coord_loss:{} confidence_loss:{} class_loss:{} iou:{}".format(round(epoch_train_loss / train_len, 4), round(epoch_train_loss_coord / train_len, 4), round(epoch_train_loss_confidence / train_len, 4), round(epoch_train_loss_classes / train_len, 4), round(epoch_train_iou / epoch_train_object_num, 4)))

    #lr_reduce_scheduler.step(epoch_train_loss)

    val_loader = DataLoader(val_dataSet, batch_size=64, shuffle=True, num_workers=0)
    val_len = val_loader.__len__()
    YOLO.eval()
    with torch.no_grad():
        with tqdm(total=val_len) as tbar:

            for batch_index, batch_train in enumerate(val_loader):
                #optimizer.zero_grad()
                train_data = batch_train[0].float().cuda(device=0)
                #train_data.requires_grad = True  验证时计算loss 不需要依附上梯度
                label_data = batch_train[1].float().cuda(device=0)
                loss = loss_function(bounding_boxes=YOLO(train_data), ground_truth=label_data)
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

                tbar.set_description("val: coord_loss:{} confidence_loss:{} class_loss:{} iou:{}".format(round(loss[1], 4), round(loss[2], 4), round(loss[3], 4), round(loss[4] / loss[5], 4)), refresh=True)
                tbar.update(1)

                # feature_map_visualize(train_data[0][0], writer)
                # print("batch_index : {} ; batch_loss : {}".format(batch_index, batch_loss))
            print("val-batch-mean loss:{} coord_loss:{} confidence_loss:{} class_loss:{} iou:{}".format(round(epoch_val_loss / val_len, 4), round(epoch_val_loss_coord / val_len, 4), round(epoch_val_loss_confidence / val_len, 4), round(epoch_val_loss_classes / val_len, 4), round(epoch_val_iou / epoch_val_object_num, 4)))

    epoch = epoch + 1
    '''
    if (epoch < 1000 and epoch % 100 == 0) or epoch % 1000 == 0:
        torch.save(Yolo.state_dict(), './YOLO_V1_' + str(epoch) + '.pth')
        writer.close()
        writer = SummaryWriter(logdir='log', filename_suffix=' [' + str(epoch) + '~' + str(epoch + 1000) + ']')
    '''
    if epoch % 100 == 0:
        state = {}
        state["model"] = YOLO.state_dict()
        state["optim"] = optimizer_Adam
        state["epoch"] = epoch
        torch.save(state, './YOLO_V1_' + str(epoch) + '.pth')
        writer.close()
        writer = SummaryWriter(logdir='log',filename_suffix='[' + str(epoch) + '~' + str(epoch + 100)+']')
    print("epoch : {} ; loss : {}".format(epoch,{epoch_train_loss}))
    for name, layer in YOLO.named_parameters():
        writer.add_histogram(name + '_grad', layer.grad.cpu().data.numpy(), epoch)
        writer.add_histogram(name + '_data', layer.cpu().data.numpy(), epoch)

    writer.add_scalar('Train/Loss_sum', epoch_train_loss, epoch)
    writer.add_scalar('Train/Loss_coord', epoch_train_loss_coord, epoch)
    writer.add_scalar('Train/Loss_confidenct', epoch_train_loss_confidence, epoch)
    writer.add_scalar('Train/Loss_classes', epoch_train_loss_classes, epoch)
    writer.add_scalar('Train/Epoch_iou', epoch_train_iou / epoch_train_object_num, epoch)

writer.close()