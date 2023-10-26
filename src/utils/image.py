# 按照指定图像大小调整尺寸
import cv2
import random
import numpy as np

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

id_to_name_dict = {}

for name, id in class_dict.items():
    id_to_name_dict[id - 1] = name

def show_image_normal(img, output_coords, win_name):
    #img = constant.numpy()
    #print(img.dtype)
    for coord in output_coords:
        print("coord:{}".format(coord))
        img = cv2.rectangle(img, (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3])), (0, 0, 255), 1)
        #img = cv2.putText(img, id_to_name_dict[coord[4]], (int(coord[0]*448).item(),int(coord[1]*448)+10).item(), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        img = cv2.putText(img, id_to_name_dict[coord[4]], (int(coord[0]),int(coord[1])+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow(win_name, img)
    cv2.waitKey(0)

def show_image(constant, output_coords):

    img = constant[0].numpy()
    for coord in output_coords:
        coord = coord.cpu()
        print("show:{}".format(coord))
        img = cv2.rectangle(img, (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3])), (0, 255, 0), 1)
        img = cv2.putText(img, id_to_name_dict[coord[4].item()], (int(coord[0].item()),int(coord[1].item())+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow("main", img)
    cv2.waitKey(0)

def resize_image(image, height, width, input_coords=[]):
    top, bottom, left, right = (0, 0, 0, 0)
    # 获取图像尺寸
    h, w, _ = image.shape
    # 求宽高的缩放比例
    scale_h = height / h
    scale_w = width / w
    # 对于长宽不相等的图片，找到最长的一边
    longest_edge = max(h, w)

    # 计算短边需要增加多上像素宽度使其与长边等长
    if h < longest_edge:#短边是高
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
        scale_factor = scale_w#缩放因子为宽度变化系数
    else:
        dw = longest_edge - w#短边是宽
        left = dw // 2
        right = dw - left
        scale_factor = scale_h#缩放因子为高度变化系数

    # RGB颜色
    BLACK = [0, 0, 0]

    output_coords = []
    for coord in input_coords:
        xmin = round((coord[0] + left) * scale_factor)
        ymin = round((coord[1] + top) * scale_factor)
        xmax = round((coord[2] + left) * scale_factor)
        ymax = round((coord[3] + top) * scale_factor)

        #过滤过小的目标
        if xmax - xmin < 5 or ymax - ymin < 5:
            #print("xmin:{} ymin:{} xmax:{} ymax:{} coord:{}".format(xmin, ymin, xmax, ymax, coord))
            continue
        
        #print("res:{}".format([xmin, ymin, xmax, ymax, coord[4]]))
        #coords.append([xmin, ymin, xmax, ymax, coord[4]])
        output_coords.append([xmin / width, ymin / height, xmax / width, ymax / height, coord[4]])

    # 给图像增加边界，使得图片长、宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)
    constant = cv2.resize(constant, (width, height), cv2.INTER_LINEAR)
        
    return constant, output_coords


#噪声

#椒盐噪声
def AddSaltPepperNoise(img, max_rate=0.3):
    
    rate = random.uniform(0, max_rate)

    height, width = img.shape[0:2]
    noiseCount = int(rate * height * width / 2)
    # add salt noise
    X = np.random.randint(width, size=(noiseCount,))
    Y = np.random.randint(height, size=(noiseCount,))
    img[Y, X] = 255
    # add black peper noise
    X = np.random.randint(width, size=(noiseCount,))
    Y = np.random.randint(height, size=(noiseCount,))
    img[Y, X] = 0
    return img

def AddGaussNoise(img, sigma=0.3):
    mean = 0
    sigma = random.uniform(0, sigma)
    # 获取图片的高度和宽度
    height, width, channels = img.shape[0:3]
    gauss = np.random.normal(mean, sigma, (height, width, channels))
    img = np.uint8(img + gauss)
    img[img < 0] = 0
    img[img > 255] = 255
    return img

#滤波

def MeanBlur(img, kernel_size=3):
    img = cv2.blur(img, (kernel_size, kernel_size))
    img[img < 0] = 0
    img[img > 255] = 255
    return img

def GaussianBulr(img, kernel_size=3, sigma_x=0.2,sigma_y=0.2):
    sigma_x = random.uniform(0, sigma_x)
    sigma_y = random.uniform(0, sigma_y)
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma_x, sigma_y)
    img[img < 0] = 0
    img[img > 255] = 255
    return img

def MedianBlur(img, kernel_size=3):
    img = cv2.medianBlur(img, kernel_size)
    img[img < 0] = 0
    img[img > 255] = 255
    return img

def BilateralBlur(img, d=5, sigmaColor=20, sigmaSpace=50):
    sigmaColor = random.uniform(0, sigmaColor)
    sigmaSpace = random.uniform(0, sigmaSpace)
    img = cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)
    img[img < 0] = 0
    img[img > 255] = 255
    return img

##亮度变化
def SqrtImg(img):#有问题
    img = np.float32(img)
    img = cv2.sqrt(img)
    return img

def EqualizeHist(img):
    img = cv2.equalizeHist(img)
    img[img < 0] = 0
    img[img > 255] = 255
    return img

def ClaheImg(img): #自适应直方图均衡
    B, G, R = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5,5))
    clahe_B = clahe.apply(B)
    clahe_G = clahe.apply(G)
    clahe_R = clahe.apply(R)
    img = cv2.merge((clahe_B, clahe_G, clahe_R))
    img[img < 0] = 0
    img[img > 255] = 255
    return img

def DetailEnhance(img, sigma_s=10, sigma_r=1):
    sigma_s = random.uniform(0, sigma_s)
    sigma_r = random.uniform(0, sigma_r)
    img = cv2.detailEnhance(img, sigma_s, sigma_r)
    img[img < 0] = 0
    img[img > 255] = 255
    return img

def illuminationChange(img):
    img_zero = np.zeros(img.shape, dtype=np.uint8)
    img=cv2.illuminationChange(img,mask=img_zero,alpha=0.2,beta=0.4)
    img[img < 0] = 0
    img[img > 255] = 255
    return img


def X_Flip(image, input_coords):#水平翻转

    image = cv2.flip(image, 1)
    height, width, _ = image.shape

    output_coords = []
    for coord in input_coords:
        o_xmin = coord[0]
        ymin = coord[1]
        o_xmax = coord[2]
        ymax = coord[3]
        xmin = width - o_xmax
        xmax = width - o_xmin
        output_coords.append([xmin, ymin, xmax, ymax, coord[4]])

    return image, output_coords

def transplant(image, input_coords, max_trans_factor=1.3, area_threshold=0.5):#平移 (图像, annotation_xml, 目标宽, 目标高, 平移的距离比例)
    trans_factor = random.uniform(2 - max_trans_factor, max_trans_factor)

    # ------1.opencv right translation------
    height, width, _ = image.shape
    trans_width = (int)(width * (trans_factor - 1))
    trans_height = (int)(height * (trans_factor - 1))
    trans_martix = np.float32([[1, 0, trans_width], [0, 1, trans_height]])
    image = cv2.warpAffine(image, trans_martix, (width, height))

    #print(temp_coords)
    output_coords=[]
    for coord in input_coords:
        xmin_org = coord[0]
        ymin_org = coord[1]
        xmax_org = coord[2]
        ymax_org = coord[3]

        #print("coord:{} xmin_org + trans_width:{} width:{} ymin_org + trans_height:{} height:{} bool:{}".format(coord, xmin_org + trans_width, width, ymin_org + trans_height, height, xmin_org + trans_width >= width or ymin_org + trans_height >= height))

        if xmin_org + trans_width >= width or ymin_org + trans_height >= height:   # 框平移后出界 舍去
            continue

        xmin_new = min(xmin_org + trans_width, width - 1)
        xmax_new = min(xmax_org + trans_width, width - 1)
        ymin_new = min(ymin_org + trans_height, height - 1)
        ymax_new = min(ymax_org + trans_height, height - 1)

        #print("new coord:{}".format([xmin_new, xmax_new, ymin_new, ymax_new]))

        area_org = (xmax_org - xmin_org) * (ymax_org - ymin_org)
        area_new = (xmax_new - xmin_new) * (ymax_new - ymin_new)

        #print("area_org:{} area_new:{} area_threshold:{}".format(area_org, area_new, area_threshold))

        if area_org * area_threshold > area_new or xmin_new >= xmax_new or ymin_new >= ymax_new:
            continue
        #if (xmax_new - xmin_new < 20 and ymax_new - ymin_new < 20) and (area_org * area_threshold > area_new or xmin_new >= xmax_new or ymin_new >= ymax_new):
            #continue

        output_coords.append([xmin_new, ymin_new, xmax_new, ymax_new, coord[4]])
    #print(coords)
    return image, output_coords

def center_crop(image, input_coords, max_scale_factor=1.3, area_threshold=0.5): #缩放加中心裁剪 此处是长宽同比例缩放 直接调用cv2.resize即可
    
    #cv2.imshow("ori", image)
    #cv2.waitKey()

    scale_factor = random.uniform(1, max_scale_factor)

    height, width, _ = image.shape
    trans_width = (int)(width * (scale_factor - 1))
    trans_height = (int)(height * (scale_factor - 1))
    boundary_xmin = trans_width // 2
    boundary_xmax = width + boundary_xmin
    boundary_ymin = trans_height // 2
    boundary_ymax = height + boundary_ymin
    image = cv2.resize(src=image, dsize=(width + trans_width, height + trans_height),interpolation=cv2.INTER_LINEAR)
    image = image[boundary_ymin: boundary_ymax, boundary_xmin: boundary_xmax]

    coords = []
    for coord in input_coords:
        xmin_org = coord[0]
        ymin_org = coord[1]
        xmax_org = coord[2]
        ymax_org = coord[3]

        #area_org = (xmax_org - xmin_org) * (ymax_org - ymin_org)

        xmin_new = max(boundary_xmin, (int)(xmin_org * scale_factor))
        xmax_new = min(boundary_xmax - 1, (int)(xmax_org * scale_factor))
        ymin_new = max(boundary_ymin, (int)(ymin_org * scale_factor))
        ymax_new = min(boundary_ymax - 1, (int)(ymax_org * scale_factor))

        #area_new = (xmax_new - xmin_new) * (ymax_new - ymin_new)

        #if xmin_new >= xmax_new or ymin_new >= ymax_new or area_org * area_threshold > area_new:
            #continue

        if xmin_new >= xmax_new or ymin_new >= ymax_new:
            continue
            #print("ori-center:{}".format([xmin_org, ymin_org, xmax_org, ymax_org]))
            #print("boundary:{}".format([boundary_xmin, boundary_ymin, boundary_xmax, boundary_ymax]))
            #print("new-center:{}".format([xmin_new, ymin_new, xmax_new, ymax_new]))
            #print("c1:{} c2:{} c3:{} c4:{}".format(xmin_new >= boundary_xmax, ymin_new >= boundary_ymax, xmax_new <= boundary_xmin, ymax_new <= boundary_ymin))

        xmin_new = xmin_new - boundary_xmin
        xmax_new = xmax_new - boundary_xmin
        ymin_new = ymin_new - boundary_ymin
        ymax_new = ymax_new - boundary_ymin

        image = cv2.rectangle(image, (xmin_new, ymin_new),(xmax_new, ymax_new), (0, 255, 0), 1)

        coords.append([xmin_new, ymin_new, xmax_new, ymax_new, coord[4]])

    #cv2.imshow("crop", image)
    #cv2.waitKey()

    return image, coords


def bbox_rotate(bboxes, M, img_shape):
    """Flip bboxes horizontally.
    Args:
        bbox(list): [left, right, up, down]
        img_shape(tuple): (height, width)
    """
    for box_id in range(len(bboxes)):
        bbox = bboxes[box_id]
        #print(bbox)
        a = M[:, :2]  ##a.shape (2,2)
        b = M[:, 2:]  ###b.shape(2,1)
        b = np.reshape(b, newshape=(1, 2))
        a = np.transpose(a)

        [left, up, right, down, cls_id] = bbox
        corner_point = np.array([[left, up], [right, up], [left, down], [right, down]])
        corner_point = np.dot(corner_point, a) + b
        min_left = max(int(np.min(corner_point[:, 0])), 0)
        max_right = min(int(np.max(corner_point[:, 0])), img_shape[1])
        min_up = max(int(np.min(corner_point[:, 1])), 0)
        max_down = min(int(np.max(corner_point[:, 1])), img_shape[0])
        bboxes[box_id] = [min_left, min_up, max_right, max_down, cls_id]

    return bboxes

def RotateImage(img, coords=[], max_degree=60):
    height, width, _ = img.shape

    angle = random.uniform(-max_degree, max_degree)
    rotate_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    img = cv2.warpAffine(img, rotate_matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    coords = bbox_rotate(coords, rotate_matrix, (height, width))

    return img, coords

def EqualizeHistImage(img):
   
    B, G, R = cv2.split(img)

    B = cv2.equalizeHist(B)
    G = cv2.equalizeHist(G)
    R = cv2.equalizeHist(R)

    cv2.merge([np.uint8(B), np.uint8(G), np.uint8(R)], dst=img)
    
    img[img > 255] = 255
    return img

def AugBrightness_HSV(img, aug_ratio=1.3):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    aug_ratio = random.uniform(1, aug_ratio)
    H, S, V = cv2.split(img)
    cv2.merge([np.uint8(H), np.uint8(S), np.uint8(V * aug_ratio)], dst=img)
    cv2.cvtColor(src=img, dst=img, code=cv2.COLOR_HSV2BGR)
    img[img > 255] = 255
    img[img < 0] = 0
    return img

def AugBrightness_RGB(img, brightness=1.3):
    [averB, averG, averR] = np.array(cv2.mean(img))[:-1] / 3
    k = np.ones((img.shape))
    k[:, :, 0] *= averB
    k[:, :, 1] *= averG
    k[:, :, 2] *= averR
    brightness = random.uniform(1, brightness)
    img = (img + (brightness - 1) * k)
    img[img > 255] = 255
    img[img < 0] = 0
    img = img.astype(np.uint8)

    return img

def change_contrast(img, coefficent=1.3):

    imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m = cv2.mean(img)[0]
    coefficent = random.uniform(1, coefficent)
    graynew = m + coefficent * (imggray - m)
    img1 = np.zeros(img.shape, np.float32)
    k = np.divide(graynew, imggray, out=np.zeros_like(graynew), where=imggray != 0)
    img1[:, :, 0] = img[:, :, 0] * k
    img1[:, :, 1] = img[:, :, 1] * k
    img1[:, :, 2] = img[:, :, 2] * k
    img1[img1 > 255] = 255
    img1[img1 < 0] = 0
    img1 = img1.astype(np.uint8)

    return img1


def gamma_transfer(img, gamma_min=0, gamma_max=1):

    #cv2.imshow("ori", img)
    #cv2.waitKey()

    gamma = random.uniform(gamma_min, gamma_max)
    # 对图像进行伽马变换
    gamma_correction = np.power(img / 255.0, gamma)
    img = (gamma_correction * 255).astype(np.uint8)
    img[img > 255] = 255
    img[img < 0] = 0

    #cv2.imshow("gamma", img)
    #cv2.waitKey()

    return img

'''
def gaussian_blur(img, mean=1, std=2):
    #cv2.imshow("ori", img)
    #cv2.waitKey()

    mean = random.uniform(-mean, mean)
    std = random.uniform(-std, std)
    img = cv2.GaussianBlur(img, (3,3), mean, std)
    img[img > 255] = 255
    img[img < 0] = 0

    #cv2.imshow("blur", img)
    #cv2.waitKey()

    return img

def gaussian_noise(img, mean=0, std=0.01, noise_ratio=0.3):
    #cv2.imshow("ori", img)
    #cv2.waitKey()
    noise_ratio = random.uniform(0, noise_ratio)
    size = img.shape
    # 对图像归一化处理
    img = img / 255
    gauss = np.random.normal(mean, std**0.05, size)

    pixel_num = img.size
    noise_num = int(pixel_num * noise_ratio)

    mask_map = np.zeros(pixel_num)
    mask_map[:noise_num] = 1

    #print("sum:{} num:{}".format(mask_map.sum(), pixel_num))

    np.random.shuffle(mask_map)

    mask_map.resize(size)

    img = img + gauss * mask_map
    img[img > 255] = 255
    img[img < 0] = 0

    #cv2.imshow("noise", img)
    #cv2.waitKey()

    return img
'''

def saturation(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(img)
    cv2.merge([np.uint8(H), np.uint8(S * 1.05), np.uint8(V)], dst=img)
    cv2.cvtColor(src=img, dst=img, code=cv2.COLOR_HSV2BGR)
    return img

def exposure(img,gamma=1.3):
    gamma = random.uniform(1, gamma)
    gamma_table=[np.power(x/255.0,gamma)*255.0 for x in range(256)]#建立映射表
    gamma_table=np.round(np.array(gamma_table)).astype(np.uint8)#颜色值为整数
    img = cv2.LUT(img,gamma_table)
    img[img > 255] = 255
    img[img < 0] = 0
    return img#图片颜色查表。另外可以根据光强（颜色）均匀化原则设计自适应算法。
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