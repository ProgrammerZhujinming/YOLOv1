import os
import json
import torch
import shutil
import argparse

if __name__ == '__main__':
    # 1.deal data parameters
    parser = argparse.ArgumentParser(description="data deal config")
    parser.add_argument('--ori_data_path', type=str, help="detection data path", default="/media/zhujinming/软件/Dataset/zip/COCO2017")
    parser.add_argument('--tar_data_path', type=str, help="the target data path", default="/media/zhujinming/软件/YOLO/data/COCO")
    args = parser.parse_args()

    train_path = os.path.join(args.tar_data_path, 'train')
    val_path = os.path.join(args.tar_data_path, 'val')

    train_imgs_path = os.path.join(train_path, 'imgs')
    train_labels_path = os.path.join(train_path, 'labels')

    val_imgs_path = os.path.join(val_path, 'imgs')
    val_labels_path = os.path.join(val_path, 'labels')

    category_info_path = os.path.join(args.tar_data_path, "class_info.pth")

    set_name_list = ["train", "val"]
    for set_name in set_name_list:
        json_file = os.path.join(args.ori_data_path, "annotations", "instances_{}2017.json".format(set_name))
        tar_imgs_path = os.path.join(args.tar_data_path, "{}".format(set_name), "imgs")
        tar_labels_path = os.path.join(args.tar_data_path, "{}".format(set_name), "labels")

        try:
            original_umask = os.umask(0)
            if not os.path.exists(tar_imgs_path):
                os.makedirs(tar_imgs_path, mode=0o777)
        except Exception:
                print("make dir {} fail! Please check or make manually!".format(tar_imgs_path))
        finally:
            os.umask(original_umask)

        try:
            original_umask = os.umask(0)
            if not os.path.exists(tar_labels_path):
                os.makedirs(tar_labels_path, mode=0o777)
        except Exception:
                print("make dir {} fail! Please check or make manually!".format(tar_labels_path))
        finally:
            os.umask(original_umask)

        with open(json_file, 'r') as json_content:
            json_dict = json.load(json_content)
            images_dict_list = json_dict['images']
            labels_dict = {}  # image_id : image_info_dict

            for image_dict in images_dict_list:
                file_name = image_dict['file_name'] # 'COCO_val2014_000000391895.jpg',
                #height = image_dict['height']
                #width = image_dict['width']
                image_id = image_dict['id']
                #labels_dict[image_id] = {'file_name':file_name, 'height': height, 'width':width, 'bboxs':[]}
                labels_dict[image_id] = {'file_name': file_name, 'bboxs': []}

            annotations_dict_list = json_dict['annotations']
            for annotation_dict in annotations_dict_list:
                image_id = annotation_dict['image_id']
                bbox = annotation_dict['bbox']
                bbox[2] = bbox[2] + bbox[0]
                bbox[3] = bbox[3] + bbox[1]
                category_id = annotation_dict['category_id']
                bbox.append(category_id)
                labels_dict[image_id]['bboxs'].append(bbox)

            category_index = 0
            category_id_index_map = {}
            category_info_dict = {}

            categorys_dict_list = json_dict['categories']
            for category_dict in categorys_dict_list:
                supercategory = category_dict['supercategory']
                category_id = category_dict['id']
                category_name = category_dict['name']
                category_id_index_map[category_id] = category_index
                category_info_dict[category_index] = {"name":category_name, "supercategory":supercategory}
                category_index = category_index + 1

            img_info_dict = {"data":[]}
            for image_id in labels_dict.keys():
                image_dict = labels_dict[image_id]

                for coord_id in range(len(image_dict["bboxs"])):
                    image_dict["bboxs"][coord_id][4] = category_id_index_map[image_dict["bboxs"][coord_id][4]]

                coords = image_dict['bboxs']
                file_name = image_dict['file_name']  # 'COCO_val2014_000000391895.jpg',
                #height = image_dict['height']
                #width = image_dict['width']

                ori_img_path = os.path.join(args.ori_data_path, "{}2017".format(set_name), file_name)
                tar_img_path = os.path.join(args.tar_data_path, "{}".format(set_name), "imgs", file_name)

                img_info_dict["data"].append({"img_path":ori_img_path, "coords":coords})

                shutil.copy(ori_img_path, tar_img_path)

            torch.save(img_info_dict, os.path.join(args.tar_data_path, "{}".format(set_name), "labels", "data.pth"))

        torch.save(category_info_dict, category_info_path)

        '''
        file_path = os.path.join(tar_labels_path, file_name.replace(".jpg", ".txt"))
        with open(file_path, "w") as w_file:
            #w_file.write(str(width) + " " + str(height) + "\n")
            for bbox in bboxs:
                w_file.write(str(bbox[0]) + " " + str(bbox[1]) + " " + str(bbox[2]) + " " + str(bbox[3]) + " " + str(category_id_index_map[bbox[4]]) + "\n")
        '''
