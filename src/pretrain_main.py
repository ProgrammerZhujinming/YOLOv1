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
from network.model import YOLO_CLASSIFY

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

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

def compute_acc(output, target, topk=(1, 5)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True).item()
        res.append((correct_k / batch_size))
    return res

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
    parser = argparse.ArgumentParser(description="YOLO_Classify train config")
    parser.add_argument('--img_channels', type=int, help="channel num of the input image", default=3)
    parser.add_argument('--batch_size', type=int, help="YOLO_Classify train batch_size", default=32)
    parser.add_argument('--num_workers', type=int, help="YOLO_Classify train num_worker num", default=4)
    parser.add_argument('--lr', type=float, help="lr", default=1e-4)
    parser.add_argument('--epoch_num', type=int, help="YOLO_Classify train epoch num", default=1000)
    parser.add_argument('--save_freq', type=int, help="save YOLO_Classify frequency", default=50)
    parser.add_argument('--class_num', type=int, help="YOLO_Classify train class_num", default=256)
    parser.add_argument('--classify_dataset_path', type=str, help="path of classify dataset", default="")
    parser.add_argument('--restart', type=bool, help="YOLO_Classify train from zero", default=True)
    parser.add_argument('--ckpt_path', type=str, help="YOLO_Classify weight path", default="")
    parser.add_argument('--save_path', type=str, help="YOLO_Classify weight files root path", default="../checkpoints/pretrain/YOLO_Classify_" + time.strftime('%Y-%m-%d-%H%M%S', time.localtime(time.time())))
    parser.add_argument('--seed', type=int, default=-1, help="the global seed, when you set this value which is not equal to -1, you will get reproducible results")
    parser.add_argument('--grad_visualize', type=bool, default=False, help="whether to visualize grad of network")
    args = parser.parse_args()

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
    yolo_classify = YOLO_CLASSIFY(args.img_channels, args.class_num)

    if args.restart == False:
        if not os.path.exists(args.ckpt_path):
            raise Exception("please check the ckpt_path path:{}".format(args.ckpt_path))
        args.save_path = args.ckpt_path
        model_files_list = list(filter(file_filter, os.listdir(args.ckpt_path)))
        model_files_list.sort()
        params_dict = torch.load(model_files_list[-1], map_location=torch.device("cpu"))
        max_acc = params_dict['max_acc']
        epoch_start = int(model_files_list[-1].split(".")[0].split("_")[-1])
        yolo_classify.set_pretrain_weight(params_dict['model_weight'])
        optimizer = params_dict['optimizer']

    else:
        params_dict = {}
        max_acc = 0
        epoch_start = 0
        yolo_classify.apply(init_weights)
        optimizer = optim.Adam(params=yolo_classify.parameters(), lr=args.lr, weight_decay=5e-3)
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

    yolo_classify.to(device=device, non_blocking=True)
    loss_classify = nn.CrossEntropyLoss().to(device=device)
    #loss_classify = nn.BCELoss().to(device=device)

    #2.dataset
    from torchvision.datasets import ImageFolder
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  
        transforms.Normalize(mean=(0.55201384, 0.53359979, 0.50502834),std=(0.24185446, 0.24112613, 0.24363275))
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),  
        transforms.Normalize(mean=(0.55201384, 0.53359979, 0.50502834),std=(0.24185446, 0.24112613, 0.24363275))
    ])
                     
    #dataset = ImageFolder(args.classify_dataset_path, transform=transform)
    #data_num = len(dataset)
    #train_num = int(data_num * 0.9)
    #val_num = data_num - train_num

    #train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_num, val_num])
    train_dataset = ImageFolder(os.path.join(args.classify_dataset_path, 'train'), transform=train_transform)
    val_dataset = ImageFolder(os.path.join(args.classify_dataset_path, 'val'), transform=val_transform)
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

        with tqdm(total=train_batch_num) as tbar:
            yolo_classify.train()
            for batch_id, train_data in enumerate(train_loader):
                imgs_data = train_data[0].requires_grad_(True).float().to(device=device, non_blocking=True)
                labels_data = train_data[1].long().to(device=device, non_blocking=True)
                #labels_data = nn.functional.one_hot(labels.long(),args.class_num).float().to(device=device, non_blocking=True)
                #labels_data = torch.zeros(args.batch_size, args.class_num).scatter_(1,train_data[1].long().resize(args.batch_size,1),1).to(device=device, non_blocking=True)
                predict_prob_out = yolo_classify(imgs_data)

                optimizer.zero_grad()
                loss = loss_classify(predict_prob_out, labels_data)
                loss.backward()
                optimizer.step()
                loss = loss.item()

                top1_acc, top5_acc = compute_acc(predict_prob_out, labels_data)
                epoch_train_loss = epoch_train_loss + loss
                epoch_train_top1_acc = epoch_train_top1_acc + top1_acc
                epoch_train_top5_acc = epoch_train_top5_acc + top5_acc

                tbar.set_description("epoch-train:{} loss:{} top1-acc:{} top5-acc:{}".format(epoch_id, round(loss, 4), round(top1_acc, 4), round(top5_acc, 4), refresh=True))
                tbar.update(1)

        avg_train_sample_loss = epoch_train_loss / train_batch_num
        avg_train_sample_top1_acc = epoch_train_top1_acc / train_batch_num
        avg_train_sample_top5_acc = epoch_train_top5_acc / train_batch_num

        print("train-loss:{} top1-acc:{} top5-acc:{}".format(
            round(avg_train_sample_loss, 4),
            round(avg_train_sample_top1_acc, 4),
            round(avg_train_sample_top5_acc, 4)))

        with tqdm(total=val_batch_num) as tbar:
            yolo_classify.eval()
            with torch.no_grad():

                for batch_id, val_data in enumerate(val_loader):
                    imgs_data = val_data[0].float().to(device=device, non_blocking=True)
                    labels_data = val_data[1].long().to(device=device, non_blocking=True)
                    #labels_data = nn.functional.one_hot(labels.long(),args.class_num).float().to(device=device, non_blocking=True)
                    predict_prob_out = yolo_classify(imgs_data)

                    loss = loss_classify(predict_prob_out, labels_data)
                    top1_acc, top5_acc = compute_acc(predict_prob_out, labels_data)

                    loss = loss.item()
                    epoch_val_loss = epoch_val_loss + loss
                    epoch_val_top1_acc = epoch_val_top1_acc + top1_acc
                    epoch_val_top5_acc = epoch_val_top5_acc + top5_acc

                    tbar.set_description("epoch-val:{} loss:{} top1-acc:{} top5-acc:{}".format(epoch_id, round(loss, 4), round(top1_acc, 4), round(top5_acc, 4), refresh=True))
                    tbar.update(1)

            avg_val_sample_loss = epoch_val_loss / val_batch_num
            avg_val_sample_top1_acc = epoch_val_top1_acc / val_batch_num
            avg_val_sample_top5_acc = epoch_val_top5_acc / val_batch_num

        print("val-loss:{} top1-acc:{} top5-acc:{}".format(
            round(avg_val_sample_loss, 4),
            round(avg_val_sample_top1_acc, 4),
            round(avg_val_sample_top5_acc, 4)))

        if epoch_val_top1_acc > max_acc:# best result
            max_acc = epoch_val_top1_acc
            model_weight = yolo_classify.state_dict()
            torch.save(model_weight, os.path.join(weights_dir_path,'YOLO_Classify_Best.pth'))


        if epoch_id % args.save_freq == 0:
            params_dict['optimizer'] = optimizer
            params_dict['model_weight'] = yolo_classify.state_dict()
            params_dict['max_acc'] = max_acc
            torch.save(params_dict, os.path.join(weights_dir_path,'YOLO_Classify_' + str(epoch_id) + '.pth'))
            params_dict = {}
            writer.close()
            writer = SummaryWriter(logdir=logs_dir_path, filename_suffix='[' + str(epoch_id) + '~' + str(epoch_id + args.save_freq) + ']')

        if args.grad_visualize:
            for i, (name, layer) in enumerate(yolo_classify.named_parameters()):##some bug
                print(name)
                if 'bn' not in name:
                    writer.add_histogram(name + '_grad', layer, epoch_id)

        writer.add_scalar('Train/Loss_sample', avg_train_sample_loss, epoch_id)
        writer.add_scalar('Train/Batch_Acc_Top1', round(avg_train_sample_top1_acc, 4), epoch_id)
        writer.add_scalar('Train/Batch_Acc_Top5', round(avg_train_sample_top5_acc, 4), epoch_id)

        writer.add_scalar('Val/Loss_sample', avg_val_sample_loss, epoch_id)
        writer.add_scalar('Val/Batch_Acc_Top1', round(avg_val_sample_top1_acc, 4), epoch_id)
        writer.add_scalar('Val/Batch_Acc_Top5', round(avg_val_sample_top5_acc, 4), epoch_id)

    writer.close()
