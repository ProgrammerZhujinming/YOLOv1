import os
import argparse
import shutil
# we assume that all data is organized as datasetName/category/xxx.jpg

if __name__ == '__main__':
    # 1.deal data parameters
    parser = argparse.ArgumentParser(description="data deal config")
    parser.add_argument('--ratio', type=float, help="ratio which used to split the classification data into train and val", default=0.9)
    parser.add_argument('--ori_data_path', type=str, help="classification data path", default="")
    parser.add_argument('--tar_data_path', type=str, help="the target data path", default="YOLO/data")
    args = parser.parse_args()
    
    train_path = os.path.join(args.tar_data_path, 'train')
    val_path = os.path.join(args.tar_data_path, 'val')

    category_names_list = os.listdir(args.ori_data_path)
    for category_name in category_names_list:
        category_path = os.path.join(args.ori_data_path, category_name)
        train_category_path = os.path.join(train_path, category_name)
        val_category_path = os.path.join(val_path, category_name)

        try:
            original_umask = os.umask(0)
            if not os.path.exists(train_category_path):
                os.makedirs(train_category_path, mode=0o777)
        except Exception:
            print("make dir {} fail! Please check or make manually!".format(train_category_path))
        finally:
            os.umask(original_umask)

        try:
            original_umask = os.umask(0)
            if not os.path.exists(val_category_path):
                os.makedirs(val_category_path, mode=0o777)
        except Exception:
            print("make dir {} fail! Please check or make manually!".format(val_category_path))
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