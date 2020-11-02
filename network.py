from pretreat import *
import paddle
from weightInit import bn_initializer, conv_initializer
import paddle.fluid as fluid
import paddle
import paddle.nn.functional as F


class Discriminator(paddle.nn.Layer):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_1 = paddle.nn.Conv2D(3, 64, 4, 2, 1, bias_attr=False,
                                       weight_attr=fluid.ParamAttr(name="d_conv_weight_1_",
                                                                   initializer=conv_initializer, trainable=True))
        self.conv_2 = paddle.nn.Conv2D(64, 128, 4, 2, 1, bias_attr=False,
                                       weight_attr=fluid.ParamAttr(name="d_conv_weight_2_",
                                                                   initializer=conv_initializer, trainable=True))
        self.bn_2 = paddle.nn.BatchNorm2D(128, bias_attr=None,
                                          weight_attr=fluid.ParamAttr(name="d_2_bn_weight_", initializer=bn_initializer,
                                                                      trainable=True), momentum=0.8)
        self.conv_3 = paddle.nn.Conv2D(128, 256, 4, 2, 1, bias_attr=False,
                                       weight_attr=fluid.ParamAttr(name="d_conv_weight_3_",
                                                                   initializer=conv_initializer, trainable=True))
        self.bn_3 = paddle.nn.BatchNorm2D(256, bias_attr=None,
                                          weight_attr=fluid.ParamAttr(name="d_3_bn_weight_", initializer=bn_initializer,
                                                                      trainable=True), momentum=0.8)
        self.conv_4 = paddle.nn.Conv2D(256, 512, 4, 2, 1, bias_attr=False,
                                       weight_attr=fluid.ParamAttr(name="d_conv_weight_4_",
                                                                   initializer=conv_initializer, trainable=True))
        self.bn_4 = paddle.nn.BatchNorm2D(512, bias_attr=None,
                                          weight_attr=fluid.ParamAttr(name="d_4_bn_weight_", initializer=bn_initializer,
                                                                      trainable=True), momentum=0.8)
        self.conv_5 = paddle.nn.Conv2D(512, 1, 4, 1, 0, bias_attr=False,
                                       weight_attr=fluid.ParamAttr(name="d_conv_weight_5_",
                                                                   initializer=conv_initializer, trainable=True))

    def forward(self, x):
        x = self.conv_1(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.conv_4(x)
        x = self.bn_4(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.conv_5(x)
        x = F.sigmoid(x)
        return x


class Generator(paddle.nn.Layer):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv_1 = paddle.nn.Conv2DTranspose(100, 512, 4, 1, 0, bias_attr=False,
                                                weight_attr=fluid.ParamAttr(name="g_dconv_weight_1_",
                                                                            initializer=conv_initializer,
                                                                            trainable=True))
        self.bn_1 = paddle.nn.BatchNorm2D(512, bias_attr=None,
                                          weight_attr=fluid.ParamAttr(name="g_1_bn_weight_", initializer=bn_initializer,
                                                                      trainable=True), momentum=0.8)
        self.conv_2 = paddle.nn.Conv2DTranspose(512, 256, 4, 2, 1, bias_attr=False,
                                                weight_attr=fluid.ParamAttr(name="g_dconv_weight_2_",
                                                                            initializer=conv_initializer,
                                                                            trainable=True))
        self.bn_2 = paddle.nn.BatchNorm2D(256, bias_attr=None,
                                          weight_attr=fluid.ParamAttr(name="g_2_bn_weight_", initializer=bn_initializer,
                                                                      trainable=True), momentum=0.8)
        self.conv_3 = paddle.nn.Conv2DTranspose(256, 128, 4, 2, 1, bias_attr=False,
                                                weight_attr=fluid.ParamAttr(name="g_dconv_weight_3_",
                                                                            initializer=conv_initializer,
                                                                            trainable=True))
        self.bn_3 = paddle.nn.BatchNorm2D(128, bias_attr=None,
                                          weight_attr=fluid.ParamAttr(name="g_3_bn_weight_", initializer=bn_initializer,
                                                                      trainable=True), momentum=0.8)
        self.conv_4 = paddle.nn.Conv2DTranspose(128, 64, 4, 2, 1, bias_attr=False,
                                                weight_attr=fluid.ParamAttr(name="g_dconv_weight_4_",
                                                                            initializer=conv_initializer,
                                                                            trainable=True))
        self.bn_4 = paddle.nn.BatchNorm2D(64, bias_attr=None,
                                          weight_attr=fluid.ParamAttr(name="g_4_bn_weight_", initializer=bn_initializer,
                                                                      trainable=True), momentum=0.8)
        self.conv_5 = paddle.nn.Conv2DTranspose(64, 3, 4, 2, 1, bias_attr=False,
                                                weight_attr=fluid.ParamAttr(name="g_dconv_weight_5_",
                                                                            initializer=conv_initializer,
                                                                            trainable=True))
        self.tanh = paddle.nn.Tanh()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = F.relu(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = F.relu(x)
        x = self.conv_3(x)
        x = self.bn_3(x)
        x = F.relu(x)
        x = self.conv_4(x)
        x = self.bn_4(x)
        x = F.relu(x)
        x = self.conv_5(x)
        x = self.tanh(x)
        return x


###损失函数
loss = paddle.nn.BCELoss()
