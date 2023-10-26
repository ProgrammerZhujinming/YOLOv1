import os
import torch
import torch.nn as nn
import torchvision

class Convolution(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding, bn=True):
        super(Convolution, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding,bias=not bn),
            nn.BatchNorm2d(output_dim) if bn else nn.Identity(),
            nn.LeakyReLU(inplace=True, negative_slope=0.02),
        )

    def forward(self, x):
        return self.conv(x)

    def weight_visualize(self, writer, epoch_id):
        for i, (name, layer) in enumerate(self.named_parameters()):##some bug
            if 'bn' not in name:
                if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                    writer.add_histogram(name + '_weight', layer, epoch_id)
                else:
                    layer.weight_visualize(writer)

class BackBone(nn.Module):
    def __init__(self, input_dim):
        super(BackBone, self).__init__()

        self.backbone = nn.Sequential(
            Convolution(input_dim, 64, 7, 2, 3),
            nn.MaxPool2d(2, 2),

            Convolution(64, 192, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            Convolution(192, 128, 1, 1, 0),
            Convolution(128, 256, 3, 1, 1),
            Convolution(256, 256, 1, 1, 0),
            Convolution(256, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            Convolution(512, 256, 1, 1, 0),
            Convolution(256, 512, 3, 1, 1),
            Convolution(512, 256, 1, 1, 0),
            Convolution(256, 512, 3, 1, 1),
            Convolution(512, 256, 1, 1, 0),
            Convolution(256, 512, 3, 1, 1),
            Convolution(512, 256, 1, 1, 0),
            Convolution(256, 512, 3, 1, 1),
            Convolution(512, 512, 1, 1, 0),
            Convolution(512, 1024, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            Convolution(1024, 512, 1, 1, 0),
            Convolution(512, 1024, 3, 1, 1),
            Convolution(1024, 512, 1, 1, 0),
            Convolution(512, 1024, 3, 1, 1),
        )

    def forward(self, x):
        x = self.backbone(x)
        return x
    
    def weight_visualize(self, writer, epoch_id):
        for i, (name, layer) in enumerate(self.named_parameters()):##some bug
            if 'bn' not in name:
                if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
                    writer.add_histogram(name + '_weight', layer, epoch_id)
                else:
                    layer.weight_visualize(writer)

class YOLO_CLASSIFY(nn.Module):
    def __init__(self, input_dim, class_num):
        super(YOLO_CLASSIFY, self).__init__()
        self.backbone = BackBone(input_dim)

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.class_predict = nn.Sequential(
            nn.Linear(1024, class_num),
        )

        #self.backbone = torchvision.models.resnet18(pretrained=True)
        #self.backbone.fc = nn.Linear(512, class_num)

    def forward(self, x, train=True):
        x = self.backbone(x)
        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1, end_dim=3)
        x = self.class_predict(x) if train else nn.Softmax(self.class_predict(x))
        return x 

class YOLOv1(nn.Module):
    def __init__(self, B=2, class_num=20):
        super(YOLOv1, self).__init__()
        self.B = B
        self.class_num = class_num

        self.backbone = BackBone(input_dim=3)

        self.det_conv = nn.Sequential(
            Convolution(1024, 1024, 3, 1, 1, bn=False),
            Convolution(1024, 1024, 3, 2, 1,  bn=False),
            Convolution(1024, 1024, 3, 1, 1, bn=False),
            Convolution(1024, 1024, 3, 1, 1,  bn=False),
        )

        self.predict = nn.Sequential(
            nn.Linear(7 * 7 * 1024, 4096),
            #torch.nn.Dropout(p=0.5, inplace=True),
            nn.LeakyReLU(inplace=True, negative_slope=0.02),
            nn.Linear(4096, 7 * 7 * (self.B * 5 + class_num)),
        )

        self.xy_coord = []
        for y_coord in range(7):
            for x_coord in range(7):
                self.xy_coord.append([x_coord, y_coord])
        self.xy_coord = torch.Tensor(self.xy_coord)

        self.sigmoid = nn.Sigmoid()
        self.grid_size = 64
        self.img_size = 448

    def forward(self, x, train=True):
        x = self.backbone(x)
        x = self.det_conv(x)
        x = torch.flatten(x, start_dim=1, end_dim=3)
        x = self.predict(x)
        x = x.view(-1, 7, 7, (self.B * 5 + self.class_num))

        box_predict = self.sigmoid(x[:,:,:, 0 : self.B * 5])
        class_predict = x[:,:,:, self.B * 5 :] if train else self.softmax(x[:,:,:, self.B * 5 :])
        predict = torch.cat([box_predict, class_predict], dim=3)

        return predict 

    def detect(self, x, conf_threshold=0.7):#return boxes

        batch_size, _, _, _ = x.shape

        xy_coord = self.xy_coord.clone().unsqueeze(0)
        xy_coord = xy_coord.repeat(batch_size, 1, 1)
        xy_coord = xy_coord.view(-1, 2)

        predict_boxes = self.forward(x, train=False).view(-1, self.B * 5 + self.class_num)
        positive_predict_grid_mask = predict_boxes[:,4] > conf_threshold or predict_boxes[:,9] > conf_threshold
        positive_predict_grid_data = torch.select(predict_boxes, positive_predict_grid_mask)

        positive_first_predict_boxes = positive_predict_grid_data[:,0:5]
        positive_second_predict_boxes = positive_predict_grid_data[:,5:10]
        positive_class = positive_predict_grid_data[:,self.B * 5:]

        xy_coord = torch.masked_select(xy_coord, positive_predict_grid_mask)

        predict_boxes = torch.where(positive_first_predict_boxes[:, 4] > positive_second_predict_boxes[:, 9], positive_first_predict_boxes, positive_second_predict_boxes)
        
        predict_boxes[:,0] = (predict_boxes[:,0] + xy_coord[:,0]) * self.grid
        predict_boxes[:,1] = (predict_boxes[:,1] + xy_coord[:,1]) * self.grid
        predict_boxes[:,2] = predict_boxes[:,2] * self.img_size
        predict_boxes[:,3] = predict_boxes[:,3] * self.img_size

        predict_boxes = torch.cat(
            torch.max(torch.Tensor([0]), (predict_boxes[:,0] - predict_boxes[:,2]).unsqueeze(1) / 2),
            torch.max(torch.Tensor([0]), (predict_boxes[:,1] - predict_boxes[:,3]).unsqueeze(1) / 2),
            torch.min(torch.Tensor([self.img_size - 1]), (predict_boxes[:,0] + predict_boxes[:,2]).unsqueeze(1) / 2),
            torch.min(torch.Tensor([self.img_size - 1]), (predict_boxes[:,1] + predict_boxes[:,3]).unsqueeze(1) / 2),
            predict_boxes[:,4].unsqueeze(1),
            torch.argmax(positive_class, dim=1).unsqueeze(1),
        )

        return predict_boxes