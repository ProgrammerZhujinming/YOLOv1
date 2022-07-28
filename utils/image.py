# 按照指定图像大小调整尺寸
import cv2
import numpy as np

def resize_image_without_annotation(image, height, width):
    top, bottom, left, right = (0, 0, 0, 0)
    # 获取图像尺寸
    h, w, _ = image.shape
    # 求宽高的缩放比例
    scale_h = height / h
    scale_w = width / w
    # 对于长宽不相等的图片，找到最长的一边
    longest_edge = max(h, w)

    # 计算短边需要增加多上像素宽度使其与长边等长
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top

    else:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left

    # RGB颜色
    BLACK = [0, 0, 0]

    # 给图像增加边界，使得图片长、宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)
    constant = cv2.resize(constant, (width, height), cv2.INTER_LINEAR)
    return constant

def resize_image(image, height, width, annotation_xml, class_dict):
    top, bottom, left, right = (0, 0, 0, 0)
    # 获取图像尺寸
    h, w, _ = image.shape
    # 求宽高的缩放比例
    scale_h = height / h
    scale_w = width / w
    # 对于长宽不相等的图片，找到最长的一边
    longest_edge = max(h, w)

    objects_xml = annotation_xml.findall("object")
    coords = []
    # 计算短边需要增加多上像素宽度使其与长边等长
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
        scale_factor = scale_w
    else:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
        scale_factor = scale_h
    for object_xml in objects_xml:
        bnd_xml = object_xml.find("bndbox")
        class_name = object_xml.find("name").text
        if class_name not in class_dict:  # 不属于我们规定的类
            continue
        xmin = round(((float)(bnd_xml.find("xmin").text) + left) * scale_factor)
        ymin = round(((float)(bnd_xml.find("ymin").text) + top) * scale_factor)
        xmax = round(((float)(bnd_xml.find("xmax").text) + left) * scale_factor)
        ymax = round(((float)(bnd_xml.find("ymax").text) + top) * scale_factor)

        if xmax - xmin < 5 or ymax - ymin < 5:
            continue

        #coords.append([xmin, ymin, xmax, ymax, class_dict[class_name]])
        coords.append([xmin / width, ymin / height, xmax / width, ymax / height, class_dict[class_name]])
    # RGB颜色
    BLACK = [0, 0, 0]

    # 给图像增加边界，使得图片长、宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)
    constant = cv2.resize(constant, (width, height), cv2.INTER_LINEAR)
    return constant, coords

def resize_image_with_coords(image, height, width, temp_coords):
    top, bottom, left, right = (0, 0, 0, 0)
    # 获取图像尺寸
    h, w, _ = image.shape
    # 求宽高的缩放比例
    scale_h = height / h
    scale_w = width / w
    # 对于长宽不相等的图片，找到最长的一边
    longest_edge = max(h, w)

    # 计算短边需要增加多上像素宽度使其与长边等长
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
        scale_factor = scale_w
    else:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
        scale_factor = scale_h

    coords = [] # 是否会原地改变
    for coord in temp_coords:
        xmin = round((coord[0] + left) * scale_factor)
        ymin = round((coord[1] + top) * scale_factor)
        xmax = round((coord[2] + left) * scale_factor)
        ymax = round((coord[3] + top) * scale_factor)

        if xmax - xmin < 5 or ymax - ymin < 5:
            continue

        #coords.append([xmin, ymin, xmax, ymax, coord[4]])
        coords.append([xmin / width, ymin / height, xmax / width, ymax / height, coord[4]])
    # RGB颜色
    BLACK = [0, 0, 0]

    # 给图像增加边界，使得图片长、宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)
    constant = cv2.resize(constant, (width, height), cv2.INTER_LINEAR)
    return constant, coords

def transplant(image, annotation_xml, trans_factor=1.2, area_threshold=0.5):#平移 (图像, annotation_xml, 目标宽, 目标高, 平移的距离比例)
    # ------1.opencv right translation------
    coords = []
    height, width, _ = image.shape
    trans_width = (int)(width * (trans_factor - 1))
    trans_height = (int)(height * (trans_factor - 1))
    trans_martix = np.float32([[1, 0, trans_width], [0, 1, trans_height]])
    image = cv2.warpAffine(image, trans_martix, (width, height))

    objects_xml = annotation_xml.findall("object")
    for object_xml in objects_xml:
        # 获取目标的名字
        bnd_xml = object_xml.find("bndbox")
        xmin_org = (int)((float)(bnd_xml.find("xmin").text))
        ymin_org = (int)((float)(bnd_xml.find("ymin").text))
        xmax_org = (int)((float)(bnd_xml.find("xmax").text))
        ymax_org = (int)((float)(bnd_xml.find("ymax").text))

        if xmin_org + trans_width >= width or ymin_org + trans_height >= height:  # 框平移后出界 舍去
            continue

        xmin_new = xmin_org + trans_width
        xmax_new = min(xmax_org + trans_width, width)
        ymin_new = ymin_org + trans_height
        ymax_new = min(ymax_org + trans_height, height)

        area_org = (xmax_org - xmin_org) * (ymax_org - ymin_org)
        area_new = (xmax_new - xmin_new) * (ymax_new - ymin_new)

        if area_org * area_threshold > area_new or xmin_new >= xmax_new or ymin_new >= ymax_new:  # 要把旧的节点删除
            annotation_xml.remove(object_xml)
            continue

        bnd_xml.find("xmin").text = str(xmin_new)
        bnd_xml.find("xmax").text = str(xmax_new)
        bnd_xml.find("ymin").text = str(ymin_new)
        bnd_xml.find("ymax").text = str(ymax_new)
        coords.append([xmin_new, ymin_new, xmax_new, ymax_new])
    return image, annotation_xml

def transplant_with_coords(image, temp_coords, trans_factor=1.2, area_threshold=0.5):#平移 (图像, annotation_xml, 目标宽, 目标高, 平移的距离比例)
    # ------1.opencv right translation------
    coords = []
    height, width, _ = image.shape
    trans_width = (int)(width * (trans_factor - 1))
    trans_height = (int)(height * (trans_factor - 1))
    trans_martix = np.float32([[1, 0, trans_width], [0, 1, trans_height]])
    image = cv2.warpAffine(image, trans_martix, (width, height))

    for coord in temp_coords:
        xmin_org = coord[0]
        ymin_org = coord[1]
        xmax_org = coord[2]
        ymax_org = coord[3]

        if xmin_org + trans_width >= width or ymin_org + trans_height >= height:   # 框平移后出界 舍去
            continue

        xmin_new = xmin_org + trans_width
        xmax_new = min(xmax_org + trans_width, width)
        ymin_new = ymin_org + trans_height
        ymax_new = min(ymax_org + trans_height, height)

        area_org = (xmax_org - xmin_org) * (ymax_org - ymin_org)
        area_new = (xmax_new - xmin_new) * (ymax_new - ymin_new)

        if area_org * area_threshold > area_new or xmin_new >= xmax_new or ymin_new >= ymax_new:
            continue

        coords.append([xmin_new, ymin_new, xmax_new, ymax_new, coord[4]])
    return image, coords

def center_crop(image, annotation_xml, scale_factor=1.2, area_threshold=0.5): #缩放加中心裁剪 此处是长宽同比例缩放 直接调用cv2.resize即可
    coords = []
    height, width, _ = image.shape
    trans_width = (int)(width * (scale_factor - 1))
    trans_height = (int)(height * (scale_factor - 1))
    boundary_xmin = (int)(trans_width / 2)
    boundary_xmax = width + boundary_xmin
    boundary_ymin = (int)(trans_height / 2)
    boundary_ymax = height + boundary_ymin
    image = cv2.resize(src=image, dsize=(width + trans_width, height + trans_height),interpolation=cv2.INTER_LINEAR)
    image = image[boundary_ymin: boundary_ymax, boundary_xmin: boundary_xmax]
    objects_xml = annotation_xml.findall("object")
    for object_xml in objects_xml:
        # 获取目标的名字
        bnd_xml = object_xml.find("bndbox")
        xmin_org = (int)((float)(bnd_xml.find("xmin").text))
        ymin_org = (int)((float)(bnd_xml.find("ymin").text))
        xmax_org = (int)((float)(bnd_xml.find("xmax").text))
        ymax_org = (int)((float)(bnd_xml.find("ymax").text))

        xmin_new = max(boundary_xmin, (int)(xmin_org * scale_factor))
        xmax_new = min(boundary_xmax, (int)(xmax_org * scale_factor))
        ymin_new = max(boundary_ymin, (int)(ymin_org * scale_factor))
        ymax_new = min(boundary_ymax, (int)(ymax_org * scale_factor))

        area_org = (xmax_org - xmin_org) * (ymax_org - ymin_org)
        area_new = (xmax_new - xmin_new) * (ymax_new - ymin_new)

        if xmin_new >= xmax_new or ymin_new >= ymax_new or area_org * area_threshold > area_new:
            annotation_xml.remove(object_xml)
            continue

        xmin_new = xmin_new - boundary_xmin
        xmax_new = xmax_new - boundary_xmin
        ymin_new = ymin_new - boundary_ymin
        ymax_new = ymax_new - boundary_ymin

        bnd_xml.find("xmin").text = str(xmin_new)
        bnd_xml.find("xmax").text = str(xmax_new)
        bnd_xml.find("ymin").text = str(ymin_new)
        bnd_xml.find("ymax").text = str(ymax_new)
        coords.append([xmin_new, ymin_new, xmax_new, ymax_new])
    return image, annotation_xml

def center_crop_with_coords(image, temp_coords, scale_factor=1.2, area_threshold=0.5): #缩放加中心裁剪 此处是长宽同比例缩放 直接调用cv2.resize即可
    coords = []
    height, width, _ = image.shape
    trans_width = (int)(width * (scale_factor - 1))
    trans_height = (int)(height * (scale_factor - 1))
    boundary_xmin = (int)(trans_width / 2)
    boundary_xmax = width + boundary_xmin
    boundary_ymin = (int)(trans_height / 2)
    boundary_ymax = height + boundary_ymin
    image = cv2.resize(src=image, dsize=(width + trans_width, height + trans_height),interpolation=cv2.INTER_LINEAR)
    image = image[boundary_ymin: boundary_ymax, boundary_xmin: boundary_xmax]
    for coord in temp_coords:
        xmin_org = coord[0]
        ymin_org = coord[1]
        xmax_org = coord[2]
        ymax_org = coord[3]

        area_org = (xmax_org - xmin_org) * (ymax_org - ymin_org)

        xmin_new = max(boundary_xmin, (int)(xmin_org * scale_factor))
        xmax_new = min(boundary_xmax, (int)(xmax_org * scale_factor))
        ymin_new = max(boundary_ymin, (int)(ymin_org * scale_factor))
        ymax_new = min(boundary_ymax, (int)(ymax_org * scale_factor))

        area_new = (xmax_new - xmin_new) * (ymax_new - ymin_new)

        if xmin_new >= xmax_new or ymin_new >= ymax_new or area_org * area_threshold > area_new:
            continue

        xmin_new = xmin_new - boundary_xmin
        xmax_new = xmax_new - boundary_xmin
        ymin_new = ymin_new - boundary_ymin
        ymax_new = ymax_new - boundary_ymin

        coords.append([xmin_new, ymin_new, xmax_new, ymax_new, coord[4]])
    return image, coords

def brightness(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(img)
    cv2.merge([np.uint8(H), np.uint8(S * 1.5), np.uint8(V)], dst=img)
    cv2.cvtColor(src=img, dst=img, code=cv2.COLOR_HSV2BGR)
    return img

def saturation(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(img)
    cv2.merge([np.uint8(H), np.uint8(S * 1.5), np.uint8(V)], dst=img)
    cv2.cvtColor(src=img, dst=img, code=cv2.COLOR_HSV2BGR)
    return img

def exposure(img,gamma):
    gamma_table=[np.power(x/255.0,gamma)*255.0 for x in range(256)]#建立映射表
    gamma_table=np.round(np.array(gamma_table)).astype(np.uint8)#颜色值为整数
    return cv2.LUT(img,gamma_table)#图片颜色查表。另外可以根据光强（颜色）均匀化原则设计自适应算法。

def resize_image_with_test(image, height, width, coords):
    top, bottom, left, right = (0, 0, 0, 0)
    # 获取图像尺寸
    h, w, _ = image.shape
    # 求宽高的缩放比例
    scale_h = height / h
    scale_w = width / w
    # 对于长宽不相等的图片，找到最长的一边
    longest_edge = max(h, w)

    # 计算短边需要增加多上像素宽度使其与长边等长
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
        scale_factor = scale_w
    else:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
        scale_factor = scale_h

    for coord_index in range(len(coords)):
        coord = coords[coord_index]
        xmin = round((coord[0] + left) * scale_factor)
        ymin = round((coord[1] + top) * scale_factor)
        xmax = round((coord[2] + left) * scale_factor)
        ymax = round((coord[3] + top) * scale_factor)

        if xmax - xmin < 5 or ymax - ymin < 5:
            continue

        #coords.append([xmin, ymin, xmax, ymax, coord[4]])
        coords[coord_index] = [xmin, ymin, xmax, ymax, coord[4]]
    # RGB颜色
    BLACK = [0, 0, 0]

    # 给图像增加边界，使得图片长、宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)
    constant = cv2.resize(constant, (width, height), cv2.INTER_LINEAR)
    return constant

'''
import xml.etree.ElementTree as ET
img_path = "../DataSet/VOC2007+2012/Train/JPEGImages/000039.jpg"
annotation_path = "../DataSet/VOC2007+2012/Train/Annotations/000008.xml"
tree = ET.parse(annotation_path)
annotation_xml = tree.getroot()
img = cv2.imread(img_path)
cv2.imshow("original", img)
img_transplant = transplant(img, annotation_xml)
cv2.imshow("transplant", img_transplant[0])
img_centercrop = center_crop(img, annotation_xml)
cv2.imshow("centercrop", img_centercrop[0])
img_brightness = brightness(img)
cv2.imshow("brightness", img_brightness)
img_saturation = saturation(img)
cv2.imshow("saturation", img_saturation)
img_exposure = exposure(img, 0.5)
cv2.imshow("exposure", img_exposure)
cv2.waitKey()


img_path = "../DataSet/COCO2017/Train/JPEGImages/000000397973.jpg"
img = cv2.imread(img_path)
h, w, _ = img.shape
txt_path = "../DataSet/COCO2017/Train/Labels/000000397973.txt"
coords = []
cv2.imshow("original", img)
cv2.waitKey(1000)
with open(txt_path, 'r') as coord_file:
    for line in coord_file:
        class_id, center_x, center_y, width, height = line.split(' ')
        class_id = (int)(class_id)
        if class_id >= 80:
            continue
        center_x = round((float)(center_x) * w)
        center_y = round((float)(center_y) * h)
        width = round((float)(width) * w)
        height = round((float)(height) * h)
        print("w:{} h:{}".format(width, height))
        coord = [max(0, round(center_x - width / 2)), max(0, round(center_y - height / 2)),
                 min(w, round(center_x + width - width / 2)), min(h, round(center_y + height - height / 2)), class_id]
        coords.append(coord)
        img = cv2.rectangle(img, (coord[0], coord[1]),(coord[2], coord[3]), (0, 255, 0), 1)
        img = cv2.putText(img, "{}".format(coord[4]), (coord[0], coord[1]), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 0, 255) ,1)
        cv2.imshow("original", img)
        cv2.waitKey(1000)

print(coords)

img, coords = transplant_with_coords(img, coords)
print(coords)
img, coords = resize_image_with_coords(img, 608, 608, coords)
print(coords)
'''
#box:0.9999999999999964 0.0 anchor:350 330 coord:[0.4831730769230769, 0.5336538461538461, 0.4855769230769231, 0.5336538461538461, 14] txt_path:../DataSet/COCO2017/Train/Labels/000000397973.txt seed:2