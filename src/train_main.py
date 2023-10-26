import time
import torch
import argparse
import torch.nn as nn
import os
import random
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from network.model import YOLOv1, Convolution
from network.loss import YOLO_LOSS
from datasets.detection_dataset import Detection_Set
from utils.model import set_freeze_by_names, unfreeze_by_names

def file_filter(file_name):
    return file_name[:-4] == '.pth'

def set_global_seed(seed=2023):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def init_weights(m):
    if type(m) == nn.Conv2d:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)
    elif type(m) == nn.BatchNorm2d:
        m.weight.data.fill_(1)
        m.bias.data.zero_()

if __name__ == '__main__':
    # 1.training parameters
    parser = argparse.ArgumentParser(description="YOLO train config")
    parser.add_argument('--img_channels', type=int, help="channel num of the input image", default=3)
    parser.add_argument('--batch_size', type=int, help="YOLO train batch_size", default=16)
    parser.add_argument('--num_workers', type=int, help="YOLO train num_worker num", default=4)
    parser.add_argument('--lr', type=float, help="lr", default=1e-3)
    parser.add_argument('--epoch_num', type=int, help="YOLO train epoch num", default=1000)
    parser.add_argument('--save_freq', type=int, help="save YOLO frequency", default=50)
    parser.add_argument('--class_num', type=int, help="YOLO train class_num", default=20)
    parser.add_argument('--detection_dataset_path', type=str, help="path of detection dataset", default="../data/VOC")
    parser.add_argument('--restart', type=bool, help="YOLO train from zero", default=True)
    parser.add_argument('--pre_ckpt_path', type=str, help="YOLO pretrain weight path", default="../checkpoints/pretrain/YOLO_Classify_2023-08-24-150427/weights/YOLO_Classify_Best.pth")
    parser.add_argument('--ckpt_path', type=str, help="YOLO weight path", default="")
    parser.add_argument('--save_path', type=str, help="YOLO weight files root path", default="../checkpoints/train/YOLO_" + time.strftime('%Y-%m-%d-%H%M%S', time.localtime(time.time())))
    #parser.add_argument('--save_path', type=str, help="YOLO weight files root path", default="../checkpoints/train/YOLO_" + time.strftime('%Y-%m-%d-%H%M%S', time.localtime(time.time())))
    parser.add_argument('--seed', type=int, default=-1, help="the global seed, when you set this value which is not equal to -1, you will get reproducible results")
    parser.add_argument('--grad_visualize', type=bool, default=False, help="whether to visualize grad of network")
    parser.add_argument('--coord_loss_mode', type=str, default="default", help="which coord loss you want to choose")
    parser.add_argument('--unfreezed_epoch', type=int, default=20, help="choose which epoch to unfreeze the backbone weight")
    args = parser.parse_args()

    lr = 0.001
    lr_mul_factor_epoch_1 = 1.04
    lr_epoch_2 = 0.01
    lr_epoch_77 = 0.001
    lr_epoch_107 = 0.0001

    #2.common setting
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        if args.seed == -1:
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        else:
            set_global_seed(args.seed)
            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        device = torch.device('cpu')

    #3.param defining and initializing
    yolo = YOLOv1(B=2, class_num=args.class_num)

    if args.restart == False:
        if not os.path.exists(args.ckpt_path):
            raise Exception("please check the ckpt_path path:{}".format(args.ckpt_path))
        args.save_path = args.ckpt_path
        model_files_list = list(filter(file_filter, os.listdir(args.ckpt_path)))
        model_files_list.sort()
        params_dict = torch.load(model_files_list[-1], map_location=torch.device("cpu"))
        max_avg_iou = params_dict['max_avg_iou']
        epoch_start = int(model_files_list[-1].split(".")[0].split("_")[-1])
        yolo.set_pretrain_weight(params_dict['model_weight'])
        optimizer = params_dict['optimizer']

    else:
        if args.pre_ckpt_path == "":
            raise Exception("No pretrain weight, check the pre_ckpt_path path:{}".format(args.pre_ckpt_path))
        params_dict = {}
        max_avg_iou = 0
        epoch_start = 0

        pretrain_net_weight = torch.load(args.pre_ckpt_path, map_location=torch.device("cpu"))
        miss, unexcept = yolo.load_state_dict(pretrain_net_weight, strict=False)

        set_freeze_by_names(yolo, "backbone")

        #print("miss:{} unexcept:{}".format(miss, unexcept))

        #optimizer = optim.SGD(params=yolo.parameters(), lr=args.lr, weight_decay=5e-3, momentum=0.9)
        optimizer = optim.SGD(params=yolo.parameters(), lr=args.lr, weight_decay=5e-3)
        
        logs_dir_path = os.path.join(args.save_path, 'logs')
        weights_dir_path = os.path.join(args.save_path, 'weights')

        try:
            original_umask = os.umask(0)
            if not os.path.exists(logs_dir_path):
                os.makedirs(logs_dir_path, mode=0o777)
            if not os.path.exists(weights_dir_path):
                os.makedirs(weights_dir_path, mode=0o777)
        finally:
            os.umask(original_umask)

    yolo.to(device=device, non_blocking=True)
    loss_yolo = YOLO_LOSS(class_num=args.class_num, coord_loss_mode=args.coord_loss_mode).to(device=device)

    #2.dataset
    #先不用增强测试一下
    train_dataset = Detection_Set(os.path.join(args.detection_dataset_path, 'train' , 'labels', 'data.pth'), is_train=False, class_num=args.class_num)
    val_dataset = Detection_Set(os.path.join(args.detection_dataset_path, 'val', 'labels', 'data.pth'), is_train=False, class_num=args.class_num)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    train_batch_num = train_loader.__len__()
    val_batch_num = val_loader.__len__()

    #3.train and validate
    writer = SummaryWriter(logdir=logs_dir_path,filename_suffix='[' + str(0) + '~' + str(args.save_freq) + ']')
    for epoch_id in range(epoch_start + 1, args.epoch_num + 1, 1):

        epoch_train_loss = 0
        epoch_val_loss = 0

        epoch_train_top1_acc = 0
        epoch_val_top1_acc = 0
        epoch_train_top5_acc = 0
        epoch_val_top5_acc = 0

        epoch_train_loss = 0
        epoch_train_coord_loss = 0
        epoch_train_positive_conf_loss = 0
        epoch_train_negative_conf_loss = 0
        epoch_train_positive_class_loss = 0
        epoch_train_Avg_IoU = 0

        epoch_val_loss = 0
        epoch_val_coord_loss = 0
        epoch_val_positive_conf_loss = 0
        epoch_val_negative_conf_loss = 0
        epoch_val_positive_class_loss = 0
        epoch_val_Avg_IoU = 0

        with tqdm(total=train_batch_num, colour="green") as tbar:
            yolo.train()
            for batch_id, train_data in enumerate(train_loader):
                imgs_data, ground_truths, ground_mask_positives = train_data
                imgs_data = imgs_data.float().to(device=device, non_blocking=True)
                ground_truths = ground_truths.to(device=device, non_blocking=True)
                ground_mask_positives = ground_mask_positives.to(device=device, non_blocking=True)
                
                predict = yolo(imgs_data)
                optimizer.zero_grad()
                loss, coord_loss, positive_conf_loss, negative_conf_loss, positive_class_loss, top1_acc, top5_acc, Avg_IoU = loss_yolo(predict, ground_truths, ground_mask_positives)
                loss.backward()
                optimizer.step()

                loss = loss.item()
                coord_loss = coord_loss.item()
                positive_conf_loss = positive_conf_loss.item()
                negative_conf_loss = negative_conf_loss.item()
                positive_class_loss = positive_class_loss.item()
                Avg_IoU = Avg_IoU.item()

                epoch_train_loss = epoch_train_loss + loss
                epoch_train_coord_loss = epoch_train_coord_loss + coord_loss
                epoch_train_positive_conf_loss = epoch_train_positive_conf_loss + positive_conf_loss
                epoch_train_negative_conf_loss = epoch_train_negative_conf_loss + negative_conf_loss
                epoch_train_positive_class_loss = epoch_train_positive_class_loss + positive_class_loss
                epoch_train_top1_acc = epoch_train_top1_acc + top1_acc
                epoch_train_top5_acc = epoch_train_top5_acc + top5_acc
                epoch_train_Avg_IoU = epoch_train_Avg_IoU + Avg_IoU

                '''
                if epoch_id == epoch_start + 1:
                    lr = min(lr * lr_mul_factor_epoch_1, lr_epoch_2)
                '''    

                tbar.set_description("loss:{} coord:{} pos_conf:{} neg_conf:{} cls:{} top1-acc:{} top5-acc:{} IoU:{}".format(round(loss, 4), round(coord_loss,4), round(positive_conf_loss,4), round(negative_conf_loss,4), round(positive_class_loss,4), round(top1_acc, 4), round(top5_acc, 4), round(Avg_IoU, 4), refresh=True))
                tbar.update(1)

        avg_train_sample_loss = epoch_train_loss / train_batch_num
        avg_train_sample_coord_loss = epoch_train_coord_loss / train_batch_num
        avg_train_sample_positive_conf_loss = epoch_train_positive_conf_loss / train_batch_num
        avg_train_sample_negative_conf_loss = epoch_train_negative_conf_loss / train_batch_num
        avg_train_sample_positive_class_loss = epoch_train_positive_class_loss / train_batch_num
        avg_train_sample_top1_acc = epoch_train_top1_acc / train_batch_num
        avg_train_sample_top5_acc = epoch_train_top5_acc / train_batch_num
        avg_train_sample_Avg_IoU = epoch_train_Avg_IoU / train_batch_num

        print("train-epoch:{} loss:{} coord_loss:{} pos_conf_loss:{} neg_conf_loss:{} pos_cls_loss:{} top1-acc:{} top5-acc:{} Avg_IoU:{}".format(
            epoch_id,
            round(avg_train_sample_loss, 4),
            round(avg_train_sample_coord_loss, 4),
            round(avg_train_sample_positive_conf_loss, 4),
            round(avg_train_sample_negative_conf_loss, 4),
            round(avg_train_sample_positive_class_loss, 4),
            round(avg_train_sample_top1_acc, 4),
            round(avg_train_sample_top5_acc, 4),
            round(avg_train_sample_Avg_IoU, 4)))

        with tqdm(total=val_batch_num, colour="blue") as tbar:
            yolo.eval()
            with torch.no_grad():
                for batch_id, val_data in enumerate(val_loader):
                    imgs_data, ground_truths, ground_mask_positives = val_data
                    imgs_data = imgs_data.float().to(device=device, non_blocking=True)
                    ground_truths = ground_truths.to(device=device, non_blocking=True)
                    ground_mask_positives = ground_mask_positives.to(device=device, non_blocking=True)
                    
                    predict = yolo(imgs_data)
                    loss, coord_loss, positive_conf_loss, negative_conf_loss, positive_class_loss, top1_acc, top5_acc, Avg_IoU = loss_yolo(predict, ground_truths, ground_mask_positives)

                    loss = loss.item()
                    coord_loss = coord_loss.item()
                    positive_conf_loss = positive_conf_loss.item()
                    negative_conf_loss = negative_conf_loss.item()
                    positive_class_loss = positive_class_loss.item()
                    Avg_IoU  = Avg_IoU.item()

                    epoch_val_loss = epoch_val_loss + loss
                    epoch_val_coord_loss = epoch_val_coord_loss + coord_loss
                    epoch_val_positive_conf_loss = epoch_val_positive_conf_loss + positive_conf_loss
                    epoch_val_negative_conf_loss = epoch_val_negative_conf_loss + negative_conf_loss
                    epoch_val_positive_class_loss = epoch_val_positive_class_loss + positive_class_loss
                    epoch_val_top1_acc = epoch_val_top1_acc + top1_acc
                    epoch_val_top5_acc = epoch_val_top5_acc + top5_acc
                    epoch_val_Avg_IoU = epoch_val_Avg_IoU + Avg_IoU

                    tbar.set_description("loss:{} coord:{} pos_conf:{} neg_conf:{} cls:{} top1-acc:{} top5-acc:{} IoU:{}".format(round(loss, 4), round(coord_loss,4), round(positive_conf_loss,4), round(negative_conf_loss,4), round(positive_class_loss,4), round(top1_acc, 4), round(top5_acc, 4), round(Avg_IoU,4), refresh=True))
                    tbar.update(1)

        avg_val_sample_loss = epoch_val_loss / val_batch_num
        avg_val_sample_coord_loss = epoch_val_coord_loss / val_batch_num
        avg_val_sample_positive_conf_loss = epoch_val_positive_conf_loss / val_batch_num
        avg_val_sample_negative_conf_loss = epoch_val_negative_conf_loss / val_batch_num
        avg_val_sample_positive_class_loss = epoch_val_positive_class_loss / val_batch_num
        avg_val_sample_top1_acc = epoch_val_top1_acc / val_batch_num
        avg_val_sample_top5_acc = epoch_val_top5_acc / val_batch_num
        avg_val_sample_Avg_IoU = epoch_val_Avg_IoU / val_batch_num

        print("val-epoch:{} loss:{} coord_loss:{} pos_conf_loss:{} neg_conf_loss:{} cls_loss:{} top1-acc:{} top5-acc:{} Avg_IoU:{}".format(
            epoch_id,
            round(avg_val_sample_loss, 4),
            round(avg_val_sample_coord_loss, 4),
            round(avg_val_sample_positive_conf_loss, 4),
            round(avg_val_sample_negative_conf_loss, 4),
            round(avg_val_sample_positive_class_loss, 4),
            round(avg_val_sample_top1_acc, 4),
            round(avg_val_sample_top5_acc, 4),
            round(avg_val_sample_Avg_IoU,4)))
        
        '''
        if epoch_id == 2:
            lr = lr_epoch_2
        elif epoch_id == 77:
            lr = lr_epoch_77
        elif epoch_id == 107:
            lr = lr_epoch_107

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        '''
            
        if epoch_id > args.unfreezed_epoch:
            unfreeze_by_names(yolo, "backbone")


        if avg_val_sample_Avg_IoU > max_avg_iou:# choice max iou as best result
            max_avg_iou = avg_val_sample_Avg_IoU
            model_weight = yolo.state_dict()
            torch.save(model_weight, os.path.join(weights_dir_path,'YOLO_Best.pth'))

        if epoch_id % args.save_freq == 0:
            params_dict['optimizer'] = optimizer
            params_dict['model_weight'] = yolo.state_dict()
            params_dict['max_avg_iou'] = avg_val_sample_Avg_IoU
            torch.save(params_dict, os.path.join(weights_dir_path,'YOLO_' + str(epoch_id) + '.pth'))
            params_dict = {}
            writer.close()
            writer = SummaryWriter(logdir=logs_dir_path, filename_suffix='[' + str(epoch_id) + '~' + str(epoch_id + args.save_freq) + ']')

        '''
        if args.grad_visualize:
            for i, (name, layer) in enumerate(yolo.named_parameters()):##some bug
                if 'bn' not in name:
                    if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                        writer.add_histogram(name + '_weight', layer, epoch_id)
                    else:
                        layer.weight_visualize(writer, epoch_id)
        '''   

        writer.add_scalar('Train/Loss_sample', avg_train_sample_loss, epoch_id)
        writer.add_scalar('Train/Batch_Coord_Loss', round(avg_train_sample_coord_loss, 4), epoch_id)
        writer.add_scalar('Train/Batch_Positive_Loss', round(avg_train_sample_positive_conf_loss, 4), epoch_id)
        writer.add_scalar('Train/Batch_Negative_Loss', round(avg_train_sample_negative_conf_loss, 4), epoch_id)
        writer.add_scalar('Train/Batch_Class_Loss', round(avg_train_sample_positive_class_loss, 4), epoch_id)
        writer.add_scalar('Train/Batch_Acc_Top1', round(avg_train_sample_top1_acc, 4), epoch_id)
        writer.add_scalar('Train/Batch_Acc_Top5', round(avg_train_sample_top5_acc, 4), epoch_id)
        writer.add_scalar('Train/Batch_Avg_IoU', round(avg_train_sample_Avg_IoU, 4), epoch_id)


        writer.add_scalar('Val/Loss_sample', avg_val_sample_loss, epoch_id)
        writer.add_scalar('Val/Batch_Coord_Loss', round(avg_val_sample_coord_loss, 4), epoch_id)
        writer.add_scalar('Val/Batch_Positive_Loss', round(avg_val_sample_positive_conf_loss, 4), epoch_id)
        writer.add_scalar('Val/Batch_Negative_Loss', round(avg_val_sample_negative_conf_loss, 4), epoch_id)
        writer.add_scalar('Val/Batch_Class_Loss', round(avg_val_sample_positive_class_loss, 4), epoch_id)
        writer.add_scalar('Val/Batch_Acc_Top1', round(avg_val_sample_top1_acc, 4), epoch_id)
        writer.add_scalar('Val/Batch_Acc_Top5', round(avg_val_sample_top5_acc, 4), epoch_id)
        writer.add_scalar('Val/Batch_Avg_IoU', round(avg_val_sample_Avg_IoU, 4), epoch_id)

    writer.close()
    