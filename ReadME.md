## Introduction
It is exciting for me to tell you I have finished the YOLO with PyTorch by myself.

# Usage
```shell
#download YOLO
git clone git@github.com:ProgrammerZhujinming/YOLO.git

#prepare pretrain data(take caltech256 for example)
wget https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar
tar -xf 256_ObjectCategories.tar -C ./
mv 256_ObjectCategories Caltech256  #rename data folder
rm -r Caltech256/257.clutter/          #delete the 257.clutter because it's not a sepical category
python YOLO/src/scripts/ClassificationSplit.py --ratio 0.6 --ori_data_path Caltech256/ --tar_data_path YOLO/data/Caltech256

#prepare detection data(take voc07 for example)
wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
tar -x VOCtrainval_06-Nov-2007.tar
python YOLO/src/scripts/VOCDetectionSplit.py --ori_data_path ./VOCdevkit/VOC2007 --tar_data_path YOLO/data/VOC2007

#base dir make
cd 
mkdir checkpoints
cd checkpoints
mkdir pretrain
mkdir train
cd ..

#if your GPU memory is enough, you can ingore the parameter - seed, and use cudnn to accrelate your training(cudnn may custome some extra GPU memory)
cd src
#backbone train
python pretrain_main.py --classify_dataset_path ../data/Caltech256/ --grad_visualize True --seed 2023
#detection train
python train_main.py --seed 2029 --class_num 20 --detection_dataset_path ../data/VOC/ --coord_loss_mode "ciou"
```
