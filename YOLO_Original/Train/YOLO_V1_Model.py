import torch
import torch.nn as nn
from YOLO_Original.PreTrain.YOLO_V1_PreTrain_Model import Convention

class YOLO_V1(nn.Module):

    def __init__(self,B=2,classes_num=20):
        super(YOLO_V1,self).__init__()
        self.B = B
        self.classes_num = classes_num

        self.Conv_Feature = nn.Sequential(
            Convention(3, 64, 7, 2, 3),
            nn.MaxPool2d(2, 2),

            Convention(64, 192, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            Convention(192, 128, 1, 1, 0),
            Convention(128, 256, 3, 1, 1),
            Convention(256, 256, 1, 1, 0),
            Convention(256, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            Convention(512, 256, 1, 1, 0),
            Convention(256, 512, 3, 1, 1),
            Convention(512, 256, 1, 1, 0),
            Convention(256, 512, 3, 1, 1),
            Convention(512, 256, 1, 1, 0),
            Convention(256, 512, 3, 1, 1),
            Convention(512, 256, 1, 1, 0),
            Convention(256, 512, 3, 1, 1),
            Convention(512, 512, 1, 1, 0),
            Convention(512, 1024, 3, 1, 1),
            nn.MaxPool2d(2, 2),
        )

        self.Conv_Semanteme = nn.Sequential(
            Convention(1024, 512, 1, 1, 0),
            Convention(512, 1024, 3, 1, 1),
            Convention(1024, 512, 1, 1, 0),
            Convention(512, 1024, 3, 1, 1),
        )

        self.Conv_Back = nn.Sequential(
            Convention(1024, 1024, 3, 1, 1, need_bn=False),
            Convention(1024, 1024, 3, 2, 1, need_bn=False),
            Convention(1024, 1024, 3, 1, 1, need_bn=False),
            Convention(1024, 1024, 3, 1, 1, need_bn=False),
        )

        self.Fc = nn.Sequential(
            nn.Linear(7*7*1024,4096),
            nn.LeakyReLU(inplace=True, negative_slope=1e-1),
            nn.Linear(4096,7 * 7 * (B*5 + classes_num)),
        )

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.Conv_Feature(x)
        x = self.Conv_Semanteme(x)
        x = self.Conv_Back(x)
        # batch_size * channel * height * weight -> batch_size * height * weight * channel
        x = x.permute(0, 2, 3, 1)
        x = torch.flatten(x, start_dim=1, end_dim=3)
        x = self.Fc(x)
        x = x.view(-1,7,7,(self.B*5 + self.classes_num))
        x[:,:,:, 0 : self.B * 5] = self.sigmoid(x[:,:,:, 0 : self.B * 5])
        x[:,:,:, self.B * 5 : ] = self.softmax(x[:,:,:, self.B * 5 : ])
        return x

    # 定义权值初始化
    def initialize_weights(self, net_param_dict):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, Convention):
                m.weight_init()

        self_param_dict = self.state_dict()
        for name, layer in self.named_parameters():
            if name in net_param_dict:
                self_param_dict[name] = net_param_dict[name]
        self.load_state_dict(self_param_dict)

