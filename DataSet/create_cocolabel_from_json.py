import os
import json
json_file = "/home/jin-ming/CV_DataSet/COCO2017/annotations/instances_val2017.json" #在这里填写你的json标注文件路径
target_category_file = "./class_dict.txt"
target_labels_path = "./Val/Labels" #这里填写你的提取信息要写入到哪里

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

    with open(target_category_file, 'w') as category_file:
        categorys_dict_list = json_dict['categories']
        for category_dict in categorys_dict_list:
            supercategory = category_dict['supercategory']
            category_id = category_dict['id']
            category_name = category_dict['name']
            category_id_index_map[category_id] = category_index
            category_file.write(str(category_index) + " " + category_name + " " + supercategory + "\n")
            category_index = category_index + 1

    for image_id in labels_dict.keys():
        image_dict = labels_dict[image_id]
        file_name = image_dict['file_name']  # 'COCO_val2014_000000391895.jpg',
        #height = image_dict['height']
        #width = image_dict['width']
        bboxs = image_dict['bboxs']

        file_path = os.path.join(target_labels_path, file_name.replace(".jpg", ".txt"))
        with open(file_path, "w") as w_file:
            #w_file.write(str(width) + " " + str(height) + "\n")
            for bbox in bboxs:
                w_file.write(str(bbox[0]) + " " + str(bbox[1]) + " " + str(bbox[2]) + " " + str(bbox[3]) + " " + str(category_id_index_map[bbox[4]]) + "\n")

