import os
import cv2
import numpy as np
import tensorflow as tf
import random
import colorsys # 转化灰度图之类的东西

def upsample(image):
    org_shape = image.shape
    return tf.image.resize(image, size=[org_shape[1]*2, org_shape[2]*2], method='nearest')

def box_iou(boxA, boxB):
    """
    Box is the decoded conv out, with chs +>box_xy, box_wh
    交集面积初一并集面积。。= inter/（SA + SB - inter）
    intersection 就是中间香蕉的方行啦, 这里方便里看的话，我假设所有方形满足如下命名
    ⬇➡   坐标轴 你自己画一下，就图像和矩阵的关系你搞图像的你比我清楚双减到D双加到B
       D\----\C
        \    \
        \    \
       A\____\B
    """
    SA = boxA[..., 2] * boxA[..., 3]
    SB = boxB[..., 2] * boxB[..., 3]
    A_posDB = tf.concat([boxA[..., :2] - boxA[..., 2:]*.5, boxA[..., :2] + boxA[..., 2:]*.5],
                        axis=-1) # 最后一个维度，前两个是D后两个是B
    B_posDB = tf.concat([boxB[..., :2] - boxB[..., 2:]*.5, boxB[..., :2] + boxB[..., 2:]*.5],
                        axis=-1)
    # 对于交集的B点，那肯定是xy都取小啊， 对于D点就是取大呗
    pos_D = tf.maximum(A_posDB[..., :2], B_posDB[..., :2])
    pos_B = tf.minimum(A_posDB[..., 2:], B_posDB[..., 2:])
    intersec = tf.nn.relu(pos_B - pos_D) # 防止在图片边上的时候，框长超出图片了。。
    Sintersec = intersec[..., 2] * intersec[..., 1]
    IOU = Sintersec / (SA + SB - Sintersec)
    return IOU

def box_giou(boxA, boxB):
    """
    GIOU 是说在图片R中选择出一个像素集合R_它包含了所有的此处计算IOU的两个框，假设A集合为A， B集合为B
    此处公式为 box_iou(boxA, boxB) - count(C - AUB)/count(C)
    转化成面积写法即为：intersec(boxA, boxB)/(SA + SB - intersec(boxA, boxB)) - (SC - SA - SB + intersec(boxA, boxB))/SC
    几何问题而已，这样做目的是亚索框与框之间的距离，保证选择的框尽量不散开
    """
    A_posDB = tf.concat([boxA[..., :2] - boxA[..., 2:] * .5, boxA[..., :2] + boxA[..., 2:] * .5],
                        axis=-1)  # 先找到顶点
    B_posDB = tf.concat([boxB[..., :2] - boxB[..., 2:] * .5, boxB[..., :2] + boxB[..., 2:] * .5],
                        axis=-1)
    A_posDB= tf.concat(
        [tf.minimum(A_posDB[..., :2], A_posDB[..., 2:]), tf.maximum(A_posDB[..., :2], A_posDB[..., 2:])], axis=-1)
    # 另一种数值保护的写法
    # 就是说B顶点大于D顶点的xy坐标,这里为了算C所以必须用这种保护，不然C=0就锤子了，就算是套反了也别爷顶
    B_posDB = tf.concat(
        [tf.minimum(B_posDB[..., :2], B_posDB[..., 2:]), tf.maximum(B_posDB[..., :2], B_posDB[..., 2:])], axis=-1)
    SA = (A_posDB[..., 3] - A_posDB[..., 1]) * (A_posDB[..., 2] - A_posDB[..., 0])
    SB = (B_posDB[..., 3] - B_posDB[..., 1]) * (B_posDB[..., 2] - B_posDB[..., 0])

    # 交集
    pos_D = tf.maximum(A_posDB[..., 2:], B_posDB[..., 2:])
    pos_B = tf.minimum(A_posDB[..., :2], B_posDB[..., :2])
    intersec = tf.nn.relu(pos_B - pos_D)
    Sintersec = intersec[..., 0] * intersec[..., 1]
    IOU = Sintersec / (SA + SB - Sintersec)

    # 最小外接方形包
    c_pos_D = tf.minimum(A_posDB[..., 2:], B_posDB[..., 2:])
    c_pos_B = tf.maximum(A_posDB[..., :2], B_posDB[..., :2])
    c = tf.maximum(c_pos_B - c_pos_D, 0)
    SC = c[..., 0] * c[..., 1]
    GIOU = IOU - (SC - SA - SB + Sintersec)/SC
    return GIOU

def get_class_name(fp=""):
    if fp=="":
        names = ["CTC", "Not CTC but cell"]
        return names
    else:
        res = []
        with open(fp, 'r') as data:
            for i , name in enumerate(data):
                res[i] = name
        return res

def image_preprocess(image, target_sz, gt_boxes=None):
    # rgb image as input as keep the org 比例 填充数值128, 同时根据比例移动框的位置以及形状
    h, w, c = image.shape
    th, tw, c = target_sz
    scale = min(int(th/h), int(tw/w))
    th_s, tw_s = h*scale, w*scale
    image = tf.image.resize(image, size=(th_s, tw_s))
    image_paded = np.full(shape=[th, tw, 3], fill_value=128.0) # 画布
    dh, dw = (th - th_s)//2, (tw - tw_s)//2
    image_paded[dh:dh+th_s, dw:dw+tw_s] = image
    image_paded = image / 255.

    if gt_boxes is None:
        return image_paded
    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] + dh
        return image_paded, gt_boxes

def draw_bbox(image, bboxes, fp="", show_label=True):
    """
    这个也是超的。画画无力
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """
    if fp=="":
        classes = get_class_name()
    else:
        classes = get_class_name(fp)
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    # 框的颜色，随机选择
    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.5
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick//2)[0]
            cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  # filled

            cv2.putText(image, bbox_mess, (c1[0], c1[1]-2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick//2, lineType=cv2.LINE_AA)

    return image

def nms(boxes, iou_threhold, sigma=.3, method="soft-nms"):
    """
    输入的盒子数据格式为 number， 5 5=> xywh, score ,class
    迭代步骤， 拿到所有盒子，判断当前类别盒子的树木是否只有一个：
    对于任意类别，对评分进行排序：
      选出最大评分的盒子 b_box，并放入输出器best_boxes
      算b_box盒子与别的同类盒子的IOU大小，IOU超过阈值的删除，剩下的继续迭代
    判断当前类别盒子数目是否只有一个。。。。。。。。循环
    """
    classes_in_image= list(set(boxes[:, 5])) #xywh,score
    # 转化为class 的集合，用数组的IDX作为目标类别的索引,这个索引只是菊部的
    best_box = [] # 输出器
    for class_idx in classes_in_image:
        class_box = boxes[boxes[:, 5] == class_idx]
        max_score_box_idx = np.argmax(class_box[:, 4])
        b_box= class_box[max_score_box_idx]
        best_box.append(b_box)
        class_boxes = class_box[np.arange(len(class_box))!=max_score_box_idx]# 把大的扣了
        # 计算最佳盒子和别的盒子的IOU,这里要统一维度，因为b_box只有一个
        iou = box_iou(class_box[:, :4], b_box[np.newaxis, :4]) # xywh哈
        weight = np.ones([len(iou),], dtype=np.float)
        # 每个盒子的权重硬的就是直接把别人的自信度弄成0， 软的就是通过
        # exp（-（iou**2/sigma））这样平滑一下，IOU越大越小嘛，就反正权重越低下次迭代越不会选你
        # 西格玛是指数陡峭度的控制因子，越大越平缓，等于0的时候等效于硬的，当然数学上不可实现
        assert method in ["nms", "soft"], "别乱写字啊"
        if method == "nms":
            iou_mask = iou > iou_threhold
            weight[iou_mask] = 0
        if method == "soft":
            weight = np.exp(-(1.*iou**2/sigma))

        class_box[:, 4] *= weight
        class_box[:, 4] = class_box[:, 4][class_box[:, 4] > 0.1]# 太捞的就不要了哈
    return best_box

def postprocess_boxes(pred_bbox, org_img_shape, input_size, score_threshold):
    """
    输入 n X len（whxy 自信 类别）
    将预测的盒子转化之后抛弃异常值，然后顺便把自信很低的盒子删了
    """
    num_box = len(pred_bbox)
    valid_scale = [0, np.inf]
    pred_bbox = np.asarray(pred_bbox)
    pred_xywh = pred_bbox[:, :4]
    pred_conf = pred_bbox[:, 4:5]
    pred_prob = pred_bbox[:, 5:]

    # 把 （xywh）->(D->cord, B->cord)
    pred_cord = np.concatenate([pred_xywh[:, :2] - .5*pred_xywh[:, 2:],
                                pred_xywh[:, :2] + .5*pred_xywh[:, 2:]], axis=-1)
    # 转化到原始图片的坐标里 是图片预处理的逆过程
    o_h, o_w = org_img_shape
    ratio = min(input_size / o_w, input_size / o_h) # 最小的缩放比例
    dw = (input_size - ratio * o_w) / 2 # 还差多少的像素格子就是便宜量
    dh = (input_size - ratio * o_h) / 2
    pred_cord[:, 0::2] = (pred_cord[:, 0::2] - dw)/ratio # 呕数索引表示w的坐标即第一个维度
    pred_cord[:, 1::2] = (pred_cord[:, 1::2] - dh)/ratio

    # 删除在图片外边的框
    pred_cord = np.concatenate([np.maximum(pred_cord[:, :2], [0, 0]), np.minimum(pred_cord[:, 2], [o_w-1, o_h-1])], axis=-1) # D 在0，0之外
    # 删除D比B大的框注意这里已经转化为 DB的坐标了
    # x轴
    mask_x = (pred_cord[:, 0] > pred_cord[:, 2]).astype(int)
    mask_y = (pred_cord[:, 1] > pred_cord[:, 3]).astype(int)
    mask = mask_x + mask_y
    pred_cord[mask, :] = 0
    # 删除无效的盒子
    # 计算面积，sqrt(x2 - x1 * y2 - y1)
    box_scale = np.sqrt(np.multiply.reduce(pred_cord[:, 2:4] - pred_cord[:, 0:2], axis=-1)) # multiply.reduce = tf.reduce_multiply
    mask_scale = np.logical_and((valid_scale[0] < box_scale), (box_scale < valid_scale[1])) # 大于0 小于无穷
    # 删除掉垃圾盒子（评分低）
    classes = np.argmax(pred_prob, axis=-1) # 类别编号
    scores = pred_conf * pred_prob[np.arange(num_box), classes] # 这是类别logits * 盒子的自信度
    score_mask = (scores > score_threshold)
    mask = (score_mask.astype(int) * mask_scale.astype(int)).astype(bool)
    pred_cord, scores, classes = pred_cord[mask], scores[mask], classes[mask]
    return tf.concat([pred_cord, scores[..., np.newaxis], classes[..., np.newaxis]], axis=-1)


if __name__ == '__main__':
    test_image = tf.ones([3, 256, 256, 3])
    print(upsample(test_image).shape)