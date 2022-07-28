#---------------step0:Common Definition-----------------
import time
import torch

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

B = 2
class_num = 20
batch_size = 16
lr = 0.001
lr_mul_factor_epoch_1 = 1.04
lr_epoch_2 = 0.01
lr_epoch_77 = 0.001
lr_epoch_107 = 0.0001

weight_decay = 0.0005
momentum = 0.9
pre_weight_file = "../PreTrain/weights/resnet18.pth"
backbone_name = "resnet18"
pre_weight_dict = torch.load(pre_weight_file, map_location=torch.device("cpu"))

epoch_val_loss_min = 1000000000
epoch_interval = 10
epoch_unfreeze = 10
epochs_num = 135

#---------------step1:Dataset-------------------
from DataSet.VOC_DataSet import VOCDataSet
train_dataSet = VOCDataSet(imgs_path="../../DataSet/VOC2007+2012/Train/JPEGImages",annotations_path="../../DataSet/VOC2007+2012/Train/Annotations",classes_file="../../DataSet/VOC2007+2012/class.data", is_train=True)
val_dataSet = VOCDataSet(imgs_path="../../DataSet/VOC2007+2012/Val/JPEGImages",annotations_path="../../DataSet/VOC2007+2012/Val/Annotations",classes_file="../../DataSet/VOC2007+2012/class.data", is_train=False)
from torch.utils.data import DataLoader

#---------------step2:Model-------------------
from YOLO_ResNet.Train.YOLO_Model import YOLO
YOLO = YOLO(B, class_num)
YOLO.initialize_weights(pre_weight_dict=pre_weight_dict, backbone_name=backbone_name)
YOLO = YOLO.to(device=device, non_blocking=True)
from utils import model
model.set_freeze_by_idxs(YOLO,[0])

#---------------step3:LossFunction-------------------
from YOLO_ResNet.Train.YOLO_LossFunction import YOLO_V1_Loss
loss_function = YOLO_V1_Loss().to(device=device, non_blocking=True)

#---------------step4:Optimizer-------------------
import torch.optim as optim
optimizer_SGD = optim.SGD(YOLO.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

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
from tqdm import tqdm

if __name__ == "__main__":
    epoch = 0
    writer = SummaryWriter(logdir='./log', filename_suffix=' [' + str(0) + '~' + str(epoch_interval) + ']')

    param_dict = {}

    while epoch <= epochs_num:

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

        train_loader = DataLoader(train_dataSet, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        train_len = train_loader.__len__()
        YOLO.train()
        with tqdm(total=train_len) as tbar:

            for batch_index, batch_train in enumerate(train_loader):

                train_data = batch_train[0].float().to(device=device, non_blocking=True)
                label_data = batch_train[1].float().to(device=device, non_blocking=True)
                loss = loss_function(bounding_boxes=YOLO(train_data),ground_truth=label_data)
                batch_loss = loss[0]
                epoch_train_loss_coord = epoch_train_loss_coord + loss[1]
                epoch_train_loss_confidence = epoch_train_loss_confidence + loss[2]
                epoch_train_loss_classes = epoch_train_loss_classes + loss[3]
                epoch_train_iou = epoch_train_iou + loss[4]
                epoch_train_object_num = epoch_train_object_num + loss[5]
                batch_loss.backward()
                optimizer_SGD.step()
                optimizer_SGD.zero_grad()
                batch_loss = batch_loss.item() * batch_size
                epoch_train_loss = epoch_train_loss + batch_loss

                tbar.set_description("train: coord_loss:{} confidence_loss:{} class_loss:{} avg_iou:{}".format(round(loss[1], 4), round(loss[2], 4), round(loss[3], 4), round(loss[4] / loss[5], 4)), refresh=True)
                tbar.update(1)

                if epoch == 1:
                    lr = min(lr * lr_mul_factor_epoch_1, lr_epoch_2)

            print("train-batch-mean loss:{} coord_loss:{} confidence_loss:{} class_loss:{} iou:{}".format(round(epoch_train_loss / train_len, 4), round(epoch_train_loss_coord / train_len, 4), round(epoch_train_loss_confidence / train_len, 4), round(epoch_train_loss_classes / train_len, 4), round(epoch_train_iou / epoch_train_object_num, 4)))


        val_loader = DataLoader(val_dataSet, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_len = val_loader.__len__()
        YOLO.eval()
        with tqdm(total=val_len) as tbar:
            with torch.no_grad():
                for batch_index, batch_train in enumerate(val_loader):
                    train_data = batch_train[0].float().to(device=device, non_blocking=True)
                    label_data = batch_train[1].float().to(device=device, non_blocking=True)
                    loss = loss_function(bounding_boxes=YOLO(train_data), ground_truth=label_data)
                    batch_loss = loss[0]
                    epoch_val_loss_coord = epoch_val_loss_coord + loss[1]
                    epoch_val_loss_confidence = epoch_val_loss_confidence + loss[2]
                    epoch_val_loss_classes = epoch_val_loss_classes + loss[3]
                    epoch_val_iou = epoch_val_iou + loss[4]
                    epoch_val_object_num = epoch_val_object_num + loss[5]
                    batch_loss = batch_loss.item() * batch_size
                    epoch_val_loss = epoch_val_loss + batch_loss

                    tbar.set_description("val: coord_loss:{} confidence_loss:{} class_loss:{} iou:{}".format(round(loss[1], 4), round(loss[2], 4), round(loss[3], 4), round(loss[4] / loss[5], 4)), refresh=True)
                    tbar.update(1)

                # feature_map_visualize(train_data[0][0], writer)
                # print("batch_index : {} ; batch_loss : {}".format(batch_index, batch_loss))
            print("val-batch-mean loss:{} coord_loss:{} confidence_loss:{} class_loss:{} iou:{}".format(round(epoch_val_loss / val_len, 4), round(epoch_val_loss_coord / val_len, 4), round(epoch_val_loss_confidence / val_len, 4), round(epoch_val_loss_classes / val_len, 4), round(epoch_val_iou / epoch_val_object_num, 4)))

        epoch = epoch + 1

        if epoch == 2:
            lr = lr_epoch_2
        elif epoch == 77:
            lr = lr_epoch_77
        elif epoch == 107:
            lr = lr_epoch_107

        if epoch == epoch_unfreeze:
            model.set_freeze_by_idxs(YOLO,[0])

        for param_group in optimizer_SGD.param_groups:
            param_group["lr"] = lr

        if epoch_val_loss < epoch_val_loss_min:
            epoch_val_loss_min = epoch_val_loss
            optimal_dict = YOLO.state_dict()

        if epoch % epoch_interval == 0:
            param_dict['model'] = YOLO.state_dict()
            param_dict['optim'] = optimizer_SGD
            param_dict['epoch'] = epoch
            param_dict['optimal'] = optimal_dict
            param_dict['epoch_val_loss_min']  = epoch_val_loss_min
            torch.save(param_dict, './weights/YOLO_V1_' + str(epoch) + '.pth')
            writer.close()
            writer = SummaryWriter(logdir='log',filename_suffix='[' + str(epoch) + '~' + str(epoch + epoch_interval) + ']')
        print("epoch : {} ; loss : {}".format(epoch,{epoch_train_loss}))
        '''
        for name, layer in YOLO.named_parameters():
            writer.add_histogram(name + '_grad', layer.grad.cpu().data.numpy(), epoch)
            writer.add_histogram(name + '_data', layer.cpu().data.numpy(), epoch)
        '''
        writer.add_scalar('Train/Loss_sum', epoch_train_loss, epoch)
        writer.add_scalar('Train/Loss_coord', epoch_train_loss_coord, epoch)
        writer.add_scalar('Train/Loss_confidenct', epoch_train_loss_confidence, epoch)
        writer.add_scalar('Train/Loss_classes', epoch_train_loss_classes, epoch)
        writer.add_scalar('Train/Epoch_iou', epoch_train_iou / epoch_train_object_num, epoch)

        writer.add_scalar('Val/Loss_sum', epoch_val_loss, epoch)
        writer.add_scalar('Val/Loss_coord', epoch_val_loss_coord, epoch)
        writer.add_scalar('Val/Loss_confidenct', epoch_val_loss_confidence, epoch)
        writer.add_scalar('Val/Loss_classes', epoch_val_loss_classes, epoch)
        writer.add_scalar('Val/Epoch_iou', epoch_val_iou / epoch_val_object_num, epoch)


    writer.close()
