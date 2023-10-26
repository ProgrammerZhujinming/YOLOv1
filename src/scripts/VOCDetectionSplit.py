import os
import cv2
import torch
import shutil
import argparse
import xml.etree.ElementTree as ET

class_dict = {
    "aeroplane": 1,
    "bicycle": 2,
    "bird": 3,
    "boat": 4,
    "bottle": 5,
    "bus": 6,
    "car": 7,
    "cat": 8,
    "chair": 9,
    "cow": 10,
    "diningtable": 11,
    "dog": 12,
    "horse": 13,
    "motorbike": 14,
    "person": 15,
    "pottedplant": 16,
    "sheep": 17,
    "sofa": 18,
    "train": 19,
    "tvmonitor": 20
}

def saveData(seg_path, imgs_path, annos_path, data_dict = {"data":[]}, tar_imgs_path=""):
    with open(seg_path, 'r') as img_names:
        
        for img_name in img_names:
            img_name = img_name.strip()
            img_path = os.path.join(imgs_path, "{}.jpg".format(img_name))

            anno_path = os.path.join(annos_path, "{}.xml".format(img_name))

            shutil.copy(img_path, tar_imgs_path)
            #img_data = cv2.imread(img_path)
            tree = ET.parse(anno_path)
            anno_xml = tree.getroot()

            objects_xml = anno_xml.findall("object")
            coords = []

            for object_xml in objects_xml:
                bnd_xml = object_xml.find("bndbox")
                class_name = object_xml.find("name").text
                difficult = object_xml.find("difficult").text
                #if difficult == "1":
                    #print("filter difficult case!")
                    #continue
                if class_name not in class_dict:
                    print("error class")
                    continue
                xmin = round((float)(bnd_xml.find("xmin").text))
                ymin = round((float)(bnd_xml.find("ymin").text))
                xmax = round((float)(bnd_xml.find("xmax").text))
                ymax = round((float)(bnd_xml.find("ymax").text))
                class_id = class_dict[class_name] - 1
                coords.append([xmin, ymin, xmax, ymax, class_id])
            data_dict['data'].append({"img_path": os.path.join(tar_imgs_path, "{}.jpg".format(img_name)), "coords": coords})

def saveDataByRatio(imgs_path, annos_path, train_data_dict = {"data":[]}, train_imgs_path="", val_data_dict = {"data":[]}, val_imgs_path="", split_ratio=0.9):
    img_names = os.listdir(imgs_path)
    train_img_num = int(len(img_names) * split_ratio)

    for img_idx, img_name in enumerate(img_names):
        img_name = img_name.split(".")[0]
        img_path = os.path.join(imgs_path, "{}.jpg".format(img_name))

        anno_path = os.path.join(annos_path, "{}.xml".format(img_name))
        tree = ET.parse(anno_path)
        anno_xml = tree.getroot()
        objects_xml = anno_xml.findall("object")
        coords = []

        for object_xml in objects_xml:
            bnd_xml = object_xml.find("bndbox")
            class_name = object_xml.find("name").text
            difficult = object_xml.find("difficult").text
            #if difficult == "1":
                #print("filter difficult case!")
                #continue
            if class_name not in class_dict:
                print("error class")
                continue
            xmin = round((float)(bnd_xml.find("xmin").text))
            ymin = round((float)(bnd_xml.find("ymin").text))
            xmax = round((float)(bnd_xml.find("xmax").text))
            ymax = round((float)(bnd_xml.find("ymax").text))
            class_id = class_dict[class_name] - 1
            coords.append([xmin, ymin, xmax, ymax, class_id])

        if img_idx < train_img_num:
            shutil.copy(img_path, train_imgs_path)
        #img_data = cv2.imread(img_path)
            train_data_dict['data'].append({"img_path": os.path.join(train_imgs_path, "{}.jpg".format(img_name)), "coords": coords})
        else:
            shutil.copy(img_path, val_imgs_path)
        #img_data = cv2.imread(img_path)
            val_data_dict['data'].append({"img_path": os.path.join(val_imgs_path, "{}.jpg".format(img_name)), "coords": coords})
             

if __name__ == '__main__':
    # 1.deal data parameters
    parser = argparse.ArgumentParser(description="data deal config")
    parser.add_argument('--ori_data_path', type=str, help="detection data path", default="/media/zhujinming/软件/Dataset/VOCdevkit")
    parser.add_argument('--tar_data_path', type=str, help="the target data path", default="YOLO/data/VOC")
    parser.add_argument('--split_ratio', type=float, help="the ratio which is used to split the voc dataset", default=0.9)

    args = parser.parse_args()
    
    train_path = os.path.join(args.tar_data_path, 'train')
    val_path = os.path.join(args.tar_data_path, 'val')

    train_imgs_path = os.path.join(train_path, 'imgs')
    train_labels_path = os.path.join(train_path, 'labels')

    val_imgs_path = os.path.join(val_path, 'imgs')
    val_labels_path = os.path.join(val_path, 'labels')

    try:
        original_umask = os.umask(0)
        if not os.path.exists(train_imgs_path):
            os.makedirs(train_imgs_path, mode=0o777)
    except Exception:
            print("make dir {} fail! Please check or make manually!".format(train_imgs_path))
    finally:
        os.umask(original_umask)

    try:
        original_umask = os.umask(0)
        if not os.path.exists(train_labels_path):
            os.makedirs(train_labels_path, mode=0o777)
    except Exception:
            print("make dir {} fail! Please check or make manually!".format(train_labels_path))
    finally:
        os.umask(original_umask)

    try:
        original_umask = os.umask(0)
        if not os.path.exists(val_imgs_path):
            os.makedirs(val_imgs_path, mode=0o777)
    except Exception:
            print("make dir {} fail! Please check or make manually!".format(val_imgs_path))
    finally:
        os.umask(original_umask)

    try:
        original_umask = os.umask(0)
        if not os.path.exists(val_labels_path):
            os.makedirs(val_labels_path, mode=0o777)
    except Exception:
            print("make dir {} fail! Please check or make manually!".format(val_labels_path))
    finally:
        os.umask(original_umask)

    train_data_dict = {"data":[]}
    val_data_dict = {"data":[]}

    voc_categories = ["VOC2007", "VOC2012"]

    for voc_category in voc_categories:
        
        ori_data_path = os.path.join(args.ori_data_path, voc_category)

        if not os.path.exists(ori_data_path):
            continue

        annos_path = os.path.join(ori_data_path, 'Annotations')
        imgs_path = os.path.join(ori_data_path, 'JPEGImages')
        #train_seg_path = os.path.join(ori_data_path, 'ImageSets', 'Main', 'train.txt')
        #val_seg_path = os.path.join(ori_data_path, 'ImageSets', 'Main', 'val.txt')

        #saveData(train_seg_path, imgs_path, annos_path, train_data_dict, train_imgs_path)
        #saveData(val_seg_path, imgs_path, annos_path, val_data_dict, val_imgs_path)

        saveDataByRatio(imgs_path, annos_path, train_data_dict, train_imgs_path, val_data_dict, val_imgs_path, args.split_ratio)

    torch.save(train_data_dict, os.path.join(train_labels_path, "data.pth"))
    torch.save(val_data_dict, os.path.join(val_labels_path, "data.pth"))
'''
    category_names_list = os.listdir(args.ori_data_path)
    for category_name in category_names_list:
        category_path = os.path.join(args.ori_data_path, category_name)
        train_category_path = os.path.join(train_path, category_name)
        val_category_path = os.path.join(val_path, category_name)

        try:
            original_umask = os.umask(0)
            if not os.path.exists(train_category_path):
                os.makedirs(train_category_path, mode=0o777)
            if not os.path.exists(val_category_path):
                os.makedirs(val_category_path, mode=0o777)
        finally:
            os.umask(original_umask)

        img_names_list = os.listdir(category_path)
        img_nums = int(len(img_names_list) * args.ratio)
        train_img_names_list = img_names_list[:img_nums]
        val_img_names_list = img_names_list[img_nums:]

        for train_img_name in train_img_names_list:
            ori_train_img_path = os.path.join(category_path, train_img_name)
            tar_train_img_path = os.path.join(train_category_path, train_img_name)
            shutil.move(ori_train_img_path, tar_train_img_path)

        for val_img_name in val_img_names_list:
            ori_val_img_path = os.path.join(category_path, val_img_name)
            tar_val_img_path = os.path.join(val_category_path, val_img_name)
            shutil.move(ori_val_img_path, tar_val_img_path)
        category_path = os.path.join(args.ori_data_path, category_name)
        train_category_path = os.path.join(train_path, category_name)
        val_category_path = os.path.join(val_path, category_name)

        try:
            original_umask = os.umask(0)
            if not os.path.exists(train_category_path):
                os.makedirs(train_category_path, mode=0o777)
            if not os.path.exists(val_category_path):
                os.makedirs(val_category_path, mode=0o777)
        finally:
            os.umask(original_umask)

        img_names_list = os.listdir(category_path)
        img_nums = int(len(img_names_list) * args.ratio)
        train_img_names_list = img_names_list[:img_nums]
        val_img_names_list = img_names_list[img_nums:]

        for train_img_name in train_img_names_list:
            ori_train_img_path = os.path.join(category_path, train_img_name)
            tar_train_img_path = os.path.join(train_category_path, train_img_name)
            shutil.move(ori_train_img_path, tar_train_img_path)

        for val_img_name in val_img_names_list:
            ori_val_img_path = os.path.join(category_path, val_img_name)
            tar_val_img_path = os.path.join(val_category_path, val_img_name)
            shutil.move(ori_val_img_path, tar_val_img_path)
'''