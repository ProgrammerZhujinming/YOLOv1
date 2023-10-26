# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Bharath Hariharan
# --------------------------------------------------------

import xml.etree.ElementTree as ET
import os
import pickle
import numpy as np

# 读取annotation里面的label数据
def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)

    return objects

# 计算AP，参考前面介绍
def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

# 主函数，读取预测和真实数据，计算Recall, Precision, AP
def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) 需要计算的类别的txt文件路径.
    annopath: Path to annotations
        annopath.format(imagename) label的xml文件所在的路径
    imagesetfile: 测试txt文件，里面是每个测试图片的地址，每行一个地址
    classname: 需要计算的类别
    cachedir: 缓存标注的目录
    [ovthresh]: IOU重叠度 (default = 0.5)
    [use_07_metric]: 是否使用VOC07的11点AP计算(default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt 加载ground truth。
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    #所有文件名字。
    imagenames = [os.path.basename(x.strip()).split('.jpg')[0] for x in lines]

    #如果cachefile文件不存在，则写入
    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath.format(imagename))
            if i % 100 == 0: # 进度条
                print( 'Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
        # save
        print( 'Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            #写入cPickle文件里面。写入的是一个字典，左侧为xml文件名，右侧为文件里面个各个参数。
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # 对每张图片的xml获取函数指定类的bbox等
    class_recs = {}  # 保存的是 Ground Truth的数据
    npos = 0
    for imagename in imagenames:
        # 获取Ground Truth每个文件中某种类别的物体
        R = [obj for obj in recs[imagename] if obj['name'] == classname]

        bbox = np.array([x['bbox'] for x in R])
        #  different基本都为0/False.
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R) #list中形参len(R)个False。
        npos = npos + sum(~difficult) #自增，~difficult取反,统计样本个数

        # 记录Ground Truth的内容
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets 读取某类别预测输出
    detfile = detpath.format(classname)

    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0].split('.')[0] for x in splitlines]  # 图片ID

    confidence = np.array([float(x[1]) for x in splitlines]) # IOU值
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines]) # bounding box数值

    # 对confidence的index根据值大小进行降序排列。
    sorted_ind = np.argsort(-confidence) 
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :] #重排bbox，由大概率到小概率。
    image_ids = [image_ids[x] for x in sorted_ind] # 图片重排，由大概率到小概率。

    # go down dets and mark TPs and FPs
    nd = len(image_ids) 

    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]  #ann

        '''
        1. 如果预测输出的是(x_min, y_min, x_max, y_max)，那么不需要下面的top,left,bottom, right转换
        2. 如果预测输出的是(x_center, y_center, h, w),那么就需要转换

        3. 计算只能使用[left, top, right, bottom],对应lable的[x_min, y_min, x_max, y_max]
        '''
        bb = BB[d, :].astype(float)

        # 转化为(x_min, y_min, x_max, y_max)
        top = int(bb[1]-bb[3]/2)
        left = int(bb[0]-bb[2]/2)
        bottom = int(bb[1]+bb[3]/2)
        right = int(bb[0]+bb[2]/2)
        bb = [left, top, right, bottom]

        ovmax = -np.inf  # 负数最大值
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps) # 最大重叠
            jmax = np.argmax(overlaps) # 最大重合率对应的gt
        # 计算tp 和 fp个数
        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                # 该gt被置为已检测到，下一次若还有另一个检测结果与之重合率满足阈值，则不能认为多检测到一个目标
                if not R['det'][jmax]: 
                    tp[d] = 1.
                    R['det'][jmax] = 1 #标记为已检测
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.
        print("**************")

    # compute precision recall
    fp = np.cumsum(fp)  # np.cumsum() 按位累加
    tp = np.cumsum(tp)
    rec = tp / float(npos)

    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    # np.finfo(np.float64).eps 为大于0的无穷小
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps) 
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


import torch
import cv2
def eval_model(coord_info_path, model, device="cuda"):
    coord_info_dict = torch.load(coord_info_path, torch.device("cpu"))["data"]
    for img_path in coord_info_dict:
        coords = coord_info_dict[img_path]

        img_data = cv2

        predict_boxes = model.detect(img_path)

