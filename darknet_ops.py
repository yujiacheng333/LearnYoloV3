import tensorflow as tf
import numpy as np

class Bn(tf.keras.layers.BatchNormalization):
    """
    这里区分了冻结状态以及预测状态，预测状态不会更新aerfa和beta，冻结状态不会更改方差与均值
    """
    def __init__(self):
        super(Bn, self).__init__()
    def call(self, x, training):
        if not training:
            training = tf.constant(False)
        training = tf.logical_and(training, self.trainable) # 这里与是否可以训练，以及本层是否冻结
        return super(Bn, self).call(x, training)

class Conv(tf.keras.Model):
    def __init__(self, filters_shape, down_sample=False, activate=True, bn=True):
        """
        标准的convbnrelu 无需多提
        :param filters_shape:
        :param down_sample:
        :param activate:
        :param bn:
        """
        super(Conv, self).__init__()
        self.down_sample = down_sample
        if down_sample:
            self.pad = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))
            self.padding = "valid"
            self.stride = 2
        else:
            self.pad = None
            self.padding = "same"
            self.stride = 1

        self.conv = tf.keras.layers.Conv2D(filters=filters_shape[-1],
                                           kernel_size=filters_shape[0],
                                           strides=self.stride,
                                           use_bias=not bn,
                                           kernel_regularizer=tf.keras.regularizers.l2(5e-4),
                                           padding=self.padding)

        if bn:
            self.bn = Bn()
        else:
            self.bn = None
        if activate:
            self.ac = tf.keras.layers.LeakyReLU(.1)
        else:
            self.ac = tf.keras.activations.linear
    def call(self, x, training):
        if not self.down_sample:
            if not self.bn is None:
                return self.ac(self.bn(self.conv(x), training))
            else:
                return self.ac(self.conv(x))
        else:
            if not self.bn is None:
                return self.ac(self.bn(self.conv(self.pad(x)), training))
            else:
                return self.ac(self.conv(x))

class Res_block(tf.keras.Model):
    """
    直的残差块没啥意思
    """
    def __init__(self, filter_num1, filter_num2):
        super(Res_block, self).__init__()
        self.conv1 = Conv(filters_shape=(1, filter_num1))
        self.conv2 = Conv(filters_shape=(3, filter_num2))

    def call(self, x, training):
        return self.conv2(self.conv1(x, training), training) + x

def upsample(input):
    # 这个是上采样哈，不是反卷机
    return tf.image.resize(input, (input.shape[1] * 2, input.shape[2] * 2), method='nearest')


class darknet53(tf.keras.Model):
    """堆就完事了"""
    def __init__(self):
        super(darknet53, self).__init__()
        self.res_list = []
        self.co_No = []
        self.res_list.append(Conv((3, 32)))
        self.res_list.append(Conv((3, 64), down_sample=True))
        for i in range(1):
            self.res_list.append(Res_block(filter_num1=32, filter_num2=64))

        self.res_list.append(Conv((3, 128), down_sample=True))
        for i in range(2):
            self.res_list.append(Res_block(filter_num1=64, filter_num2=128))

        self.res_list.append(Conv((3, 256), down_sample=True))
        for i in range(8):
            self.res_list.append(Res_block(filter_num1=128, filter_num2=256))

        self.co_No.append(len(self.res_list))
        self.res_list.append(Conv((3, 512), down_sample=True))
        for i in range(8):
            self.res_list.append(Res_block(filter_num1=256, filter_num2=512))

        self.co_No.append(len(self.res_list))
        self.res_list.append(Conv((3, 1024), down_sample=True))
        for i in range(4):
            self.res_list.append(Res_block(filter_num1=512, filter_num2=1024))

        self.co_No.append(len(self.res_list))
        self.co_No = np.asarray(self.co_No) - 1
    def call(self, x, training):
        collector = []
        for i, layer in enumerate(self.res_list):
            x = layer(x, training)
            if i in self.co_No:
                collector.append(x)
        return collector



if __name__ == '__main__':
    test_input = tf.ones([32, 416, 416, 1])
    test_la = tf.ones([32, 7, 7, 1024])
    # test_input = tf.data.Dataset.from_tensor_slices(test_input)
    # test_input.repeat(10)
    a = darknet53()
    a.build(test_input.shape)
    out = a(test_input, True)
    dataset = tf.data.Dataset.from_tensor_slices((test_input, test_la)).batch(1)
    a.summary()
    import os
    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # 创建一个检查点回调
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    model = a
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=[tf.keras.metrics.categorical_accuracy])

    model.fit(dataset)
    """
    Total params: 40,620,064
Trainable params: 40,584,352
Non-trainable params: 35,712
    """
