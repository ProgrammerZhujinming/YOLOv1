import os

files_dir = "./VOC2007/Train/ImageSets/Main"
files_name = os.listdir(files_dir)

classes_name = set()

for file_name in files_name:
    file_name = file_name.split('_')[0]
    classes_name.add(file_name)

class_file_dir = "./VOC2007/Train/class.data"
with open(class_file_dir,'w') as f:
    for class_name in classes_name:
        f.write(class_name + '\n')