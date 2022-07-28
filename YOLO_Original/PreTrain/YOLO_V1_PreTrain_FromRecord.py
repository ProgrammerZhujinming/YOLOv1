#------0.common variable definition------
import torch
from utils.model import accuracy
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

param_pth = "./weights/YOLO_V1_PreTrain_1.pth"
param_dict = torch.load(param_pth, map_location=torch.device("cpu"))
batch_size = 32
num_workers = 4
weight_decay = 0.0005
epoch_num = 500
epoch_interval = 10
class_num = 80
epoch = param_dict['epoch']
min_val_loss = param_dict['min_val_loss']

#------1.voc classify DataSet------
from torch.utils.data import DataLoader
from YOLO_Original.PreTrain.COCO_Classify import coco_classify
train_dataSet = coco_classify(imgs_path="../../DataSet/COCO2014/Train/JPEGImages", txts_path="../../DataSet/COCO2014/Train/Labels", is_train=True, edge_threshold = 200)
val_dataSet = coco_classify(imgs_path="../../DataSet/COCO2014/Val/JPEGImages", txts_path="../../DataSet/COCO2014/Val/Labels", is_train=False, edge_threshold = 200)

#VOC数据集分类
#train_dataSet = coco_classify(imgs_path = "../../DataSet/VOC2007+2012/Train/JPEGImages", annotations_path = "../../DataSet/VOC2007+2012/Train/Annotations", classes_file = "../../DataSet/VOC2007+2012/class.data", is_train=True, edge_threshold = 200, input_size = 224)
#val_dataSet = coco_classify(imgs_path = "../../DataSet/VOC2007+2012/Val/JPEGImages", annotations_path = "../../DataSet/VOC2007+2012/Val/Annotations", classes_file = "../../DataSet/VOC2007+2012/class.data", is_train=False, edge_threshold = 200, input_size = 224)

#------2.Net------
from YOLO_Original.PreTrain.YOLO_V1_PreTrain_Model import YOLO_Feature
yolo_feature = YOLO_Feature(classes_num=class_num,test=False)
yolo_feature.load_state_dict(param_dict['model'])
yolo_feature.to(device=device, non_blocking=True)

#------3.Loss------
import torch.nn as nn
loss_function = nn.CrossEntropyLoss().to(device=device)

#------4.Optimizer------
optimizer = param_dict['optim']

#------ entrance --------
from tqdm import tqdm
from tensorboardX import SummaryWriter

if __name__ == "__main__":

    writer = SummaryWriter(logdir='./log', filename_suffix=' [' + str(epoch) + '~' + str(epoch + epoch_interval) + ']')

    while epoch < epoch_num:

        epoch_train_loss = 0
        epoch_val_loss = 0
        epoch_train_top1_acc = 0
        epoch_train_top5_acc = 0
        epoch_val_top1_acc = 0
        epoch_val_top5_acc = 0

        train_loader = DataLoader(dataset=train_dataSet, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  pin_memory=True)
        train_len = train_loader.__len__()
        yolo_feature.train()
        with tqdm(total=train_len) as tbar:

            for batch_index, batch_train in enumerate(train_loader):
                train_data = batch_train[0].float().to(device=device, non_blocking=True)
                label_data = batch_train[1].long().to(device=device, non_blocking=True)
                net_out = yolo_feature(train_data)
                loss = loss_function(net_out, label_data)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                batch_loss = loss.item() * batch_size
                epoch_train_loss = epoch_train_loss + batch_loss

                # 计算准确率
                net_out = net_out.detach()
                [top1_acc, top5_acc] = accuracy(net_out, label_data)
                top1_acc = top1_acc.item()
                top5_acc = top5_acc.item()

                epoch_train_top1_acc = epoch_train_top1_acc + top1_acc
                epoch_train_top5_acc = epoch_train_top5_acc + top5_acc

                tbar.set_description(
                    "train: class_loss:{} top1-acc:{} top5-acc:{}".format(loss.item(), round(top1_acc, 4),
                                                                          round(top5_acc, 4), refresh=True))
                tbar.update(1)

                # feature_map_visualize(train_data[0][0], writer, yolo_feature)
                # print("batch_index : {} ; batch_loss : {}".format(batch_index, batch_loss))
            print(
                "train-mean: batch_loss:{} batch_top1_acc:{} batch_top5_acc:{}".format(round(epoch_train_loss / train_loader.__len__(), 4), round(
                    epoch_train_top1_acc / train_loader.__len__(), 4), round(
                    epoch_train_top5_acc / train_loader.__len__(), 4)))

        # lr_reduce_scheduler.step()

        val_loader = DataLoader(dataset=val_dataSet, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                pin_memory=True)
        val_len = val_loader.__len__()
        yolo_feature.eval()
        with tqdm(total=val_len) as tbar:
            with torch.no_grad():
                for batch_index, batch_train in enumerate(val_loader):
                    train_data = batch_train[0].float().to(device=device, non_blocking=True)
                    label_data = batch_train[1].long().to(device=device, non_blocking=True)
                    net_out = yolo_feature(train_data)
                    loss = loss_function(net_out, label_data)
                    batch_loss = loss.item() * batch_size
                    epoch_val_loss = epoch_val_loss + batch_loss

                    # 计算准确率
                    net_out = net_out.detach()
                    [top1_acc, top5_acc] = accuracy(net_out, label_data)
                    top1_acc = top1_acc.item()
                    top5_acc = top5_acc.item()

                    epoch_val_top1_acc = epoch_val_top1_acc + top1_acc
                    epoch_val_top5_acc = epoch_val_top5_acc + top5_acc

                    tbar.set_description(
                        "val: class_loss:{} top1-acc:{} top5-acc:{}".format(loss.item(), round(top1_acc, 4),
                                                                            round(top5_acc, 4), refresh=True))
                    tbar.update(1)

                # feature_map_visualize(train_data[0][0], writer, yolo_feature)
                # print("batch_index : {} ; batch_loss : {}".format(batch_index, batch_loss))
            print(
                "train-mean: batch_loss:{} batch_top1_acc:{} batch_top5_acc:{}".format(round(epoch_val_loss / val_loader.__len__(), 4), round(
                    epoch_val_top1_acc / val_loader.__len__(), 4), round(
                    epoch_val_top5_acc / val_loader.__len__(), 4)))
        epoch = epoch + 1

        if min_val_loss > epoch_val_loss:
            min_val_loss = epoch_val_loss
            param_dict['min_val_loss'] = min_val_loss
            param_dict['min_loss_model'] = yolo_feature.state_dict()

        if epoch % epoch_interval == 0:
            param_dict['model'] = yolo_feature.state_dict()
            param_dict['optim'] = optimizer
            param_dict['epoch'] = epoch
            torch.save(param_dict, './weights/YOLO_V1_BackBone_' + str(epoch) + '.pth')
            writer.close()
            writer = SummaryWriter(logdir='log', filename_suffix='[' + str(epoch) + '~' + str(epoch + epoch_interval) + ']')
        print("epoch : {} ; loss : {}".format(epoch, {epoch_train_loss}))

        for i, (name, layer) in enumerate(yolo_feature.named_parameters()):
            if 'bn' not in name:
                writer.add_histogram(name + '_grad', layer, epoch)

        writer.add_scalar('Train/Loss_sum', epoch_train_loss, epoch)
        writer.add_scalar('Val/Loss_sum', epoch_val_loss, epoch)
    writer.close()