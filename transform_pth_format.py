import torch
from YOLO_V1_Model import YOLO_V1

weight_file_name = "YOLO_V1_2400.pth"

model = YOLO_V1().cuda(device=1)
model.load_state_dict(torch.load(weight_file_name))
torch.save(model.state_dict(), weight_file_name, _use_new_zipfile_serialization=False)