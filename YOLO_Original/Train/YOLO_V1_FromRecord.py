#---------------step0:Common Definition-----------------
import torch
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

# train hype parameter
B = 2
class_num = 20
lr = 1e-3
lr_epoch_2 = 1e-2
lr_epoch_77 = 1e-3
lr_epoch_107 = 1e-4
lr_mul_factor_epoch_1 = 1.04

batch_size = 16
weight_decay = 5e-4
momentum = 0.9
weight_file = "./weights/YOLO_V1_1.pth"
param_dict = torch.load(weight_file, map_location=torch.device("cpu"))

epoch = param_dict['epoch']
epoch_val_loss_min = param_dict['epoch_val_loss_min']
epoch_interval = 10
epoch_unfreeeze = 10

#---------------step1:Dataset-------------------
from torch.utils.data import DataLoader
from DataSet.VOC_DataSet import VOCDataSet
train_dataSet = VOCDataSet(imgs_path="../../DataSet/VOC2007+2012/Train/JPEGImages",annotations_path="../../DataSet/VOC2007+2012/Train/Annotations",classes_file="../../DataSet/VOC2007+2012/class.data",class_num=20, is_train=True)
val_dataSet = VOCDataSet(imgs_path="../../DataSet/VOC2007+2012/Val/JPEGImages",annotations_path="../../DataSet/VOC2007+2012/Val/Annotations",classes_file="../../DataSet/VOC2007+2012/class.data",class_num=20, is_train=False)

#---------------step2:Model-------------------#
from YOLO_Original.Train.YOLO_V1_Model import YOLO_V1
YOLO = YOLO_V1().to(device=device, non_blocking=True)
YOLO.load_state_dict(param_dict['model'])

#---------------step3:LossFunction-------------------
from YOLO_Original.Train.YOLO_V1_LossFunction import YOLO_V1_Loss
loss_function = YOLO_V1_Loss().to(device=device, non_blocking=True)

#---------------step4:Optimizer-------------------
optimizer_SGD= param_dict['optim']

#---------------step5:Train-------------------
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from tqdm import tqdm
from utils import model
if __name__ == "__main__":
    epoch = param_dict['epoch']
    writer = SummaryWriter(logdir='./log', filename_suffix=' [' + str(epoch) + '~' + str(epoch + epoch_interval) + ']')

    if epoch < epoch_unfreeeze:
        model.set_freeze_by_idxs(YOLO, [0, 1])

    train_loader = DataLoader(train_dataSet, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataSet, batch_size=batch_size, shuffle=True, num_workers=4)

    while epoch <= 160:

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

        train_len = train_loader.__len__()
        YOLO.train()
        with tqdm(total=train_len) as tbar:

            for batch_index, batch_train in enumerate(train_loader):

                train_data = batch_train[0].float().to(device=device, non_blocking=True)
                label_data = batch_train[1].float().to(device=device, non_blocking=True)
                loss = loss_function(bounding_boxes=YOLO(train_data),ground_truth=label_data)
                sample_loss = loss[0]
                epoch_train_loss_coord = epoch_train_loss_coord + loss[1]
                epoch_train_loss_confidence = epoch_train_loss_confidence + loss[2]
                epoch_train_loss_classes = epoch_train_loss_classes + loss[3]
                epoch_train_iou = epoch_train_iou + loss[4]
                epoch_train_object_num = epoch_train_object_num + loss[5]
                sample_loss.backward()
                optimizer_SGD.step()
                optimizer_SGD.zero_grad()
                batch_loss = sample_loss.item() * batch_size
                epoch_train_loss = epoch_train_loss + batch_loss

                tbar.set_description("train: coord_loss:{} confidence_loss:{} class_loss:{} avg_iou:{}".format(round(loss[1], 4), round(loss[2], 4), round(loss[3], 4), round(loss[4] / loss[5], 4)), refresh=True)
                tbar.update(1)

            print("train-batch-mean loss:{} coord_loss:{} confidence_loss:{} class_loss:{} iou:{}".format(round(epoch_train_loss / train_len, 4), round(epoch_train_loss_coord / train_len, 4), round(epoch_train_loss_confidence / train_len, 4), round(epoch_train_loss_classes / train_len, 4), round(epoch_train_iou / epoch_train_object_num, 4)))

        val_len = val_loader.__len__()
        YOLO.eval()
        with torch.no_grad():
            with tqdm(total=val_len) as tbar:

                for batch_index, batch_train in enumerate(val_loader):
                    train_data = batch_train[0].float().cuda(device=0)
                    label_data = batch_train[1].float().cuda(device=0)
                    loss = loss_function(bounding_boxes=YOLO(train_data), ground_truth=label_data)
                    sample_loss = loss[0]
                    epoch_val_loss_coord = epoch_val_loss_coord + loss[1]
                    epoch_val_loss_confidence = epoch_val_loss_confidence + loss[2]
                    epoch_val_loss_classes = epoch_val_loss_classes + loss[3]
                    epoch_val_iou = epoch_val_iou + loss[4]
                    epoch_val_object_num = epoch_val_object_num + loss[5]
                    batch_loss = sample_loss.item() * batch_size
                    epoch_val_loss = epoch_val_loss + batch_loss

                    tbar.set_description("val: coord_loss:{} confidence_loss:{} class_loss:{} iou:{}".format(round(loss[1], 4), round(loss[2], 4), round(loss[3], 4), round(loss[4] / loss[5], 4)), refresh=True)
                    tbar.update(1)

                    # feature_map_visualize(train_data[0][0], writer)
                    # print("batch_index : {} ; batch_loss : {}".format(batch_index, batch_loss))
                print("val-batch-mean loss:{} coord_loss:{} confidence_loss:{} class_loss:{} iou:{}".format(round(epoch_val_loss / val_len, 4), round(epoch_val_loss_coord / val_len, 4), round(epoch_val_loss_confidence / val_len, 4), round(epoch_val_loss_classes / val_len, 4), round(epoch_val_iou / epoch_val_object_num, 4)))

        epoch = epoch + 1

        if epoch == epoch_unfreeeze:
            model.unfreeze_by_idxs(YOLO, [0, 1])

        if epoch == 1:
            lr = lr_epoch_2
        elif epoch == 76:
            lr = lr_epoch_77
        elif epoch == 106:
            lr = lr_epoch_107

        for param_group in optimizer_SGD.param_groups:
            param_group["lr"] = lr

        if epoch_val_loss < epoch_val_loss_min:
            epoch_val_loss_min = epoch_val_loss
            optimal_dict = YOLO.state_dict()

        if epoch % epoch_interval == 0:
            param_dict["model"] = YOLO.state_dict()
            param_dict["optim"] = optimizer_SGD
            param_dict["epoch"] = epoch
            torch.save(param_dict, './weights/YOLO_V1_' + str(epoch) + '.pth')
            writer.close()
            writer = SummaryWriter(logdir='log',filename_suffix='[' + str(epoch) + '~' + str(epoch +00)+']')
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

    writer.close()