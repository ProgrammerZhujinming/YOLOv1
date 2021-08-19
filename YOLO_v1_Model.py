import torch.nn as nn
import torch

class Convention(nn.Module):
    def __init__(self,in_channels,out_channels,conv_size,conv_stride,padding):
        super(Convention,self).__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, conv_size, conv_stride, padding),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.Conv(x)

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class YOLO_V1(nn.Module):

    def __init__(self,B=2,Classes_Num=20):
        super(YOLO_V1,self).__init__()
        self.B = B
        self.Classes_Num = Classes_Num

        self.Conv_448 = nn.Sequential(
            Convention(3, 64, 7, 2, 3),
            nn.MaxPool2d(2,2),
        )

        self.Conv_112 = nn.Sequential(
            Convention(64, 192, 3, 1, 1),
            nn.MaxPool2d(2, 2),
        )

        self.Conv_56 = nn.Sequential(
            Convention(192, 128, 1, 1, 0),
            Convention(128, 256, 3, 1, 1),
            Convention(256, 256, 1, 1, 0),
            Convention(256, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2),
        )

        self.Conv_28 = nn.Sequential(
            Convention(512, 256, 1, 1, 0),
            Convention(256, 512, 3, 1, 1),
            Convention(512, 256, 1, 1, 0),
            Convention(256, 512, 3, 1, 1),
            Convention(512, 256, 1, 1, 0),
            Convention(256, 512, 3, 1, 1),
            Convention(512, 256, 1, 1, 0),
            Convention(256, 512, 3, 1, 1),
            Convention(512,512,1,1,0),
            Convention(512,1024,3,1,1),
            nn.MaxPool2d(2, 2),
        )

        self.Conv_14 = nn.Sequential(
            Convention(1024,512,1,1,0),
            Convention(512,1024,3,1,1),
            Convention(1024, 512, 1, 1, 0),
            Convention(512, 1024, 3, 1, 1),
            Convention(1024, 1024, 3, 1, 1),
            Convention(1024, 1024, 3, 2, 1),
        )

        self.Conv_7 = nn.Sequential(
            Convention(1024,1024,3,1,1),
            Convention(1024, 1024, 3, 1, 1),
        )

        self.Fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7*7*1024,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096,7 * 7 * (B*5 + Classes_Num)),
            nn.Sigmoid()
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.Conv_448(x)
        x = self.Conv_112(x)
        x = self.Conv_56(x)
        x = self.Conv_28(x)
        x = self.Conv_14(x)
        x = self.Conv_7(x)
        # batch_size * channel * height * weight -> batch_size * height * weight * channel
        x = x.permute(0, 2, 3, 1)
        x = torch.flatten(x, start_dim=1, end_dim=3)
        x = self.Fc(x)
        x = x.view((-1,7,7,(self.B*5 + self.Classes_Num)))
        x = torch.cat((x[:,0:10],self.softmax(x[:,10:])),dim=1)
        return x

    # 定义权值初始化
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, Convention):
                m.weight_init()