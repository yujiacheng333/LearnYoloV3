import tensorflow as tf
import numpy as np
from darknet_ops import Conv
from darknet_ops import darknet53
from Utils import upsample
from Utils import box_iou, box_giou
class_num = 100
class yolo3(tf.keras.Model):
    """
    这里娱乐嘛， 我到时候发个inception net 给你你从中间随便找几个特征图大小一致的层就好使了
    """
    def __init__(self):
        self.class_num = class_num # 自己改把我选的是imagenet 的与训练模型，你要选自己的
        super(yolo3, self).__init__()

        self.duck = darknet53()
        self.conv1 = Conv((1, 512))
        self.conv2 = Conv((3, 1024))
        self.conv3 = Conv((1, 512))
        self.conv4 = Conv((3, 1024))
        self.conv5 = Conv((1, 512))
        # block 1 end
        self.conv6 = Conv((1, 256))
        # 这里加了Upsample 并拼接
        self.conv7 = Conv((1, 256)) # pointwiseconv 整合通道信息呗
        self.conv8 = Conv((3, 512))
        self.conv9 = Conv((1, 256))
        self.conv10 = Conv((3, 512))
        self.conv11 = Conv((1, 256))
        # block 2 end
        self.conv12 = Conv((1, 128))
        # 这里加了Upsample并拼接
        self.conv13 = Conv((1, 128))
        self.conv14 = Conv((3, 256))
        self.conv15 = Conv((1, 128))
        self.conv16 = Conv((3, 256))
        self.conv17 = Conv((1, 128)) # 就通过通道的震荡，甩掉不要的垃圾。。。。开个玩笑
        # block 3 end

        self._1_obj_pred_conv = Conv((3, 1024)) # 作为最粗粒度别的分之
        self._1_box_conv = Conv((1, 3*(self.class_num + 5)), activate=False, bn=False)
        self._2_obj_pred_conv = Conv((3, 512))  # 作为最粗粒度别的分之
        self._2_box_conv = Conv((1, 3 * (self.class_num + 5)), activate=False, bn=False)
        self._3_obj_pred_conv = Conv((3, 256))  # 作为最粗粒度别的分之
        self._3_box_conv = Conv((1, 3 * (self.class_num + 5)), activate=False, bn=False)
        # 每个cell生成3个盒子大小为，这里的话类别越少越容易算，毕竟这算法featuremap和通道太大了，，
        # （长宽+中心xy+先验框逻辑回归概率加类别）然后加mao点呗，然后另外两个分支算法差不多的就不写了注释了

        # 粒度变化，就是upsample 没啥意思，我看教程用的tf.image 的上采样，理论上可以用Conv2DT
    def call(self, input, training):
        """"(1, 31, 31, 256)
        (1, 15, 15, 512)
        (1, 7, 7, 1024)
        """
        _3, _2, _1 = self.duck(input, training)
        _1 = self.conv1(_1, training)
        _1 = self.conv2(_1, training)
        _1 = self.conv3(_1, training)
        _1 = self.conv4(_1, training)
        _1 = self.conv5(_1, training)

        _1_branch = self._1_obj_pred_conv(_1, training)
        _1_box_pred = self._1_box_conv(_1_branch, training)
        _1 = self.conv6(_1, training)

        _2 = tf.concat([upsample(_1), _2], axis=-1)
        _2 = self.conv7(_2, training)
        _2 = self.conv8(_2, training)
        _2 = self.conv9(_2, training)
        _2 = self.conv10(_2, training)
        _2 = self.conv11(_2, training)

        _2_branch = self._2_obj_pred_conv(_2, training)
        _2_box_pred = self._2_box_conv(_2_branch, training)
        _2 = self.conv12(_2, training)

        _3 = tf.concat([_3, upsample(_2)], axis=-1)
        _3 = self.conv13(_3, training)
        _3 = self.conv14(_3, training)
        _3 = self.conv15(_3, training)
        _3 = self.conv16(_3, training)
        _3 = self.conv17(_3, training)

        _3_branch = self._3_obj_pred_conv(_3, training)
        _3_box_pred = self._3_box_conv(_3, training)
        return _1_box_pred, _2_box_pred, _3_box_pred

class AUX_func_yolo(object):
    def __init__(self):
        self.default_anchors = "1.25,1.625, 2.0,3.75, 4.125,2.875, 1.875,3.8125, 3.875,2.8125, 3.6875,7.4375, 3.625,2.8125, 4.875,6.1875, 11.65625,10.1875"
        self.default_anchors = np.asarray(self.default_anchors.split(",")).astype(np.float32).reshape([3, 3, 2]) # 三层三个尺度，第一个指向conv输出数组，第二指向通道域中的3，然后2是边长
        self.class_num = class_num
        self.IOU_LOSS_THRESH = .45
        self.stride = [8, 16, 32] # 这是一个cell当原始图片多长的意思最小的是13嘛原始是412，412+padding次数（5次啦）/13～=32总之就是个大概，反正这只是个中心，有偏差无所为的
    def get_box(self, conv_out, type):
        """
        decode the output of conv
        :param conv_out: 卷机的输出值 这里用论文给出的通道协议，0,1对应相对与cellxy偏移量
        2,3 是盒子的长和高， 4是这个盒子对不对的预测概率， 5之后是这个东西是个啥玩意的onehot编码
        这里就很几把睿智，为啥要连在一起算啊。。。分组卷积不好使？？？？
        :param type: 第几个特征图，用来选锚点,最小的是第三个，最大的是第一个
        :return: Pxywh=>(x+cell*stride), (y+cell*stride), (w*anchor)*stride, (h*anchor)*stride
        """
        bs, sz_1, sz_2, ch = conv_out.shape
        assert sz_1 == sz_2, "麻烦resize成一样的。。。。"
        assert 3*(5 + self.class_num) == ch, "你类别输错了"
        conv_out = tf.reshape(conv_out, [bs, sz_1, sz_2, 3, 5 + self.class_num])
        dx_dy = conv_out[:, :, :, :, 0:2]
        h_w = conv_out[:, :, :, :, 2:4]
        Prob = conv_out[:, :, :, :, 4]
        leibie = conv_out[:, :, :, :, 5:]

        # 开始生成cell目标是这样的
        # 00 01 02 03
        # 10 11 12 13
        # 20 21 22 23
        # 30 31 32 33 （nxnx2）这里行和列是等价的，随缘了。。。
        x = np.arange(0, sz_2)
        y = np.arange(0, sz_1)
        x = tf.broadcast_to(tf.expand_dims(x, axis=-1), [sz_1, sz_1])
        y = tf.broadcast_to(tf.expand_dims(y, axis=0), [sz_1, sz_1])
        xy_grid = tf.concat([tf.expand_dims(x, axis=-1), tf.expand_dims(y, axis=-1)], axis=-1)
        xy_grid = tf.cast(xy_grid, tf.float32)[tf.newaxis, :, :, tf.newaxis, :]
        xy_grid = tf.broadcast_to(xy_grid, [bs, sz_1, sz_1, 3, 2])# 就是把格子拓展成和卷积输出一样的个数，占用你的内存～～没办法哈，循环tensorflow没法传梯度

        box_xy = (tf.sigmoid(dx_dy) + xy_grid)*self.stride[type]
        box_hw = (tf.exp(h_w)*self.default_anchors[type])*self.stride[type] # 都是相对啦

        Prob = tf.expand_dims(tf.sigmoid(Prob), axis=-1)
        leibie = tf.sigmoid(leibie)  # 这里可以不急着sigmoid，可以用tf.sigmoid_cross_entropy我这边提示器坏了，，反正可以用这个函数
        return tf.concat([box_xy, box_hw, Prob, leibie], axis=-1)
    def get_loss(self, pred, conv, label, bboxes, i=0):
        """
        不是很好写。。我直接抄了一份，顺便讲解一下把
        :param pred: decode之后的输出结果
        :param conv: 卷积的输出结果
        :param label:
        :param bboxes:
        :param i:
        :return:
        """
        bs, size, _, _ = conv.shape
        input_size = tf.cast(self.stride[i] * size, tf.float32)# 还原图片的形状
        conv = tf.reshape(conv, (bs, size, size, 3, 5 + self.class_num))

        conv_raw_conf = conv[..., 4:5] # 有东西没 这里4：5用于保持维度， 但只取出4
        conv_prb = conv[..., 5:] # 类别

        pred_xywh = pred[..., 0:4] # xywh,这是卷积结果经过decode之后的结果
        pred_conf = pred[..., 4:5] #

        label_xywh = label[..., 0:4]
        response_box = label[..., 4:5]
        label_prob = label[..., 5:]

        giou = tf.expand_dims(box_giou(pred_xywh, label_xywh), axis=-1)
        box_loss_scale = 2.0 - 1.0 * label_xywh[:, :, :, :, 2:3] * label_xywh[:, :, :, :, 3:4] / (input_size ** 2)
        giou_loss = response_box * box_loss_scale * (1 - giou) # 这里没看懂

        iou = box_iou(pred_xywh[:, :, :, :, np.newaxis, :], bboxes[:, np.newaxis, np.newaxis, :, :])
        max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)
        respond_bgd = (1.0 - response_box) * tf.cast(max_iou < self.IOU_LOSS_THRESH, tf.float32)
        # 预测边框是否能够与标签边框进行匹配，1，IOU足够大2， label对这个边框足够自信
        conf_focal = tf.pow(response_box - pred_conf, 2)
        conf_loss = conf_focal * (response_box * tf.nn.sigmoid_cross_entropy_with_logits(labels=response_box, logits=conv_raw_conf)
                                  +
                                  respond_bgd * tf.nn.sigmoid_cross_entropy_with_logits(labels=response_box, logits=conv_raw_conf))
        # 上述操作都说是在进行类别均衡
        prob_loss = response_box * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_prb)
        giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
        conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
        prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))
        return giou_loss, conf_loss, prob_loss

if __name__ == '__main__':
    test_input = tf.ones([1, 7, 7, 3*(100+5)])
    # a = yolo3()
    # a(test_input, True)
    a = AUX_func_yolo().get_box(test_input ,1)
