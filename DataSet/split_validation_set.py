import os
import shutil
import random
import time

train_imgs_path = "./VOC2007+2012/Train/JPEGImages"
train_annotations_path = "./VOC2007+2012/Train/Annotations"

val_imgs_path = "./VOC2007+2012/Val/JPEGImages"
val_annotations_path = "./VOC2007+2012/Val/Annotations"

imgs_name = os.listdir(train_imgs_path)

random.seed(time.time())
random.shuffle(imgs_name)

val_set_len =round(0.1 * len(imgs_name))
img_index = 0

while img_index <= val_set_len:
    img_name = imgs_name[img_index]
    annotation_name = img_name.replace(".jpg", ".xml")
    img_path = os.path.join(train_imgs_path, img_name)
    annotation_path = os.path.join(train_annotations_path, annotation_name)

    shutil.move(img_path, val_imgs_path)
    shutil.move(annotation_path, val_annotations_path)

    img_index = img_index + 1