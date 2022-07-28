import torch
import torchvision
import torch.nn as nn
from  YOLO_Original.PreTrain.YOLO_V1_PreTrain_Model import Convention

# 使用ResNet-50作为主干网络
class YOLO(nn.Module):
    def __init__(self, B, class_num):
        super(YOLO, self).__init__()
        self.B = B
        self.class_num = class_num

        self.conv = nn.Sequential(
            Convention(512, 1024, 3, 1, 1, need_bn=False),
            Convention(1024, 1024, 3, 2, 1, need_bn=False),
            Convention(1024, 1024, 3, 1, 1, need_bn=False),
            Convention(1024, 1024, 3, 1, 1, need_bn=False)
        )

        self.predict = nn.Sequential(
            nn.Linear(7*7*1024,4096),
            nn.LeakyReLU(inplace=True),
            nn.Linear(4096,7 * 7 * (B * 5 + class_num)),
        )

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()


    def forward(self, x):
        x = self.backbone(x)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        x = torch.flatten(x, start_dim=1, end_dim=3)
        x = self.predict(x)
        x = x.view((-1, 7, 7, (self.B * 5 + self.class_num)))
        x[:, :, :, 0: self.B * 5] = self.sigmoid(x[:, :, :, 0: self.B * 5])
        x[:, :, :, self.B * 5:] = self.softmax(x[:, :, :, self.B * 5:])
        return x

    def initialize_weights(self, pre_weight_dict, backbone_name = "resnet50"):
        for m in self.modules():
            if isinstance(m, Convention):
                m.weight_init()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

        if backbone_name == "resnet18":
            net = torchvision.models.resnet18()
        if backbone_name == "resnet34":
            net = torchvision.models.resnet34()
        elif backbone_name == "resnet50":
            net = torchvision.models.resnet50()
        elif  backbone_name == "resnet101":
            net = torchvision.models.resnet101()
        elif  backbone_name == "resnet152":
            net = torchvision.models.resnet152()

        net.load_state_dict(pre_weight_dict)
        self.backbone = nn.Sequential(*list(net.children())[:-2])