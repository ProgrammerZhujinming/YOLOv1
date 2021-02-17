import torch
from YOLO_V1_DataSet import YoloV1DataSet
dataSet = YoloV1DataSet()
from torch.utils.data import DataLoader
dataLoader = DataLoader(dataSet,batch_size=64,shuffle=True,num_workers=4)

from YOLO_v1_Model import YOLO_V1
Yolo = YOLO_V1()

#接續訓練的文件名
train_file = "YOLO_V1_41.pth"
Yolo.load_state_dict(torch.load(train_file))
Yolo.cuda()

from YOLO_V1_LossFunction import  Yolov1_Loss
loss_function = Yolov1_Loss().cuda()

import torch.optim as optim

optimizer = optim.SGD(Yolo.parameters(),lr=3e-4)

loss_sum = 0
epoch = int(train_file.split('_')[2].split('.')[0]) + 1
while epoch <= (2000*dataSet.Classes):
    batch_num = 0
    for batch_index, batch_train in enumerate(dataLoader):
        train_data = torch.Tensor(batch_train[0]).float().cuda()
        train_data.requires_grad = True
        label_data = torch.Tensor(batch_train[1]).float().cuda()
        loss = loss_function(bounding_boxes=Yolo(train_data),ground_truth=label_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss = loss / len(batch_train[0])
        loss_sum += loss
        batch_num += 1
        print("batch_index : {} ; batch_loss : {}".format(batch_index,loss.item()))
    epoch += 1
    if epoch % 100 == 0:
        torch.save(Yolo.state_dict(), './YOLO_V1_'+str(epoch)+'.pth')
    print("epoch : {} ; loss : {}".format(epoch,{loss_sum/(epoch+1)}))