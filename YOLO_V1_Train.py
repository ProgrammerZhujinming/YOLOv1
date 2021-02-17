import torch
from YOLO_V1_DataSet import YoloV1DataSet
from tensorboardX import SummaryWriter

dataSet = YoloV1DataSet(imgs_dir="./VOC2007/Train/JPEGImages",annotations_dir="./VOC2007/Train/Annotations",ClassesFile="./VOC2007/Train/class.data")
from torch.utils.data import DataLoader
dataLoader = DataLoader(dataSet,batch_size=32,shuffle=True,num_workers=4)

from YOLO_v1_Model import YOLO_V1
Yolo = YOLO_V1().cuda(device=1)

from YOLO_V1_LossFunction import  Yolov1_Loss
loss_function = Yolov1_Loss().cuda(device=1)

import torch.optim as optim
optimizer = optim.SGD(Yolo.parameters(),lr=5e-4,momentum=0.9)

writer = SummaryWriter('log')

for epoch in range(2000*dataSet.Classes):
    loss_sum = 0
    for batch_index, batch_train in enumerate(dataLoader):
        train_data = torch.Tensor(batch_train[0]).float().cuda(device=1)
        train_data.requires_grad = True
        label_data = torch.Tensor(batch_train[1]).float().cuda(device=1)
        loss = loss_function(bounding_boxes=Yolo(train_data),ground_truth=label_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss
        print("batch_index : {} ; batch_loss : {}".format(batch_index,loss.item()))
        writer.add_image('image',train_data[0],batch_index)
        writer.add_graph(Yolo,(train_data,))
    if epoch != 0 and epoch % 100 == 0:
        torch.save(Yolo.state_dict(), './YOLO_V1_' + str(epoch) + '.pth')
    print("epoch : {} ; loss : {}".format(epoch,{loss_sum/(epoch+1)}))
    writer.add_scalar('Train/Loss',loss_sum,epoch+1)

writer.close()