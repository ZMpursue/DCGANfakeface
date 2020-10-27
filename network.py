from pretreat import *
import paddle.fluid as fluid
from weightInit import bn_initializer, conv_initializer

use_cudnn = True
use_gpu = True
n = 0

###BatchNorm层
def bn(x, name=None, act=None,momentum=0.5):
    return fluid.layers.batch_norm(
        x,
        param_attr=fluid.ParamAttr(
            name=name+"_bn_weight_1_",
            initializer=bn_initializer,
            trainable=True),
        # 指定权重参数属性的对象
        bias_attr=None,
        # 指定偏置的属性的对象
        moving_mean_name=name + '3',
        # moving_mean的名称
        moving_variance_name=name + '4',
        # moving_variance的名称
        name=name,
        act=act,
        momentum=momentum,
    )


###卷积层
def conv(x, num_filters,name=None, act=None):
    return fluid.nets.simple_img_conv_pool(
        input=x,
        filter_size=5,
        num_filters=num_filters,
        pool_size=2,
        # 池化窗口大小
        pool_stride=2,
        # 池化滑动步长
        param_attr=fluid.ParamAttr(
            name=name+"_conv_weight_1_",
            initializer=conv_initializer,
            trainable=True),
        bias_attr=False,
        use_cudnn=use_cudnn,
        act=act
    )

###全连接层
def fc(x, num_filters, name=None, act=None):
    return fluid.layers.fc(
        input=x,
        size=num_filters,
        act=act,
        param_attr=name + 'w',
        bias_attr=name + 'b'
    )

###转置卷积层
def deconv(x, num_filters, name=None, filter_size=5, stride=2, dilation=1, padding=2, output_size=None, act=None):
    return fluid.layers.conv2d_transpose(
        input=x,
        param_attr=fluid.ParamAttr(
            name=name+"_conv_weight_2_",
            initializer=conv_initializer,
            trainable=True),
        bias_attr=False,
        num_filters=num_filters,
        # 滤波器数量
        output_size=output_size,
        # 输出图片大小
        filter_size=filter_size,
        # 滤波器大小
        stride=stride,
        # 步长
        dilation=dilation,
        # 膨胀比例大小
        padding=padding,
        use_cudnn=use_cudnn,
        # 是否使用cudnn内核
        act=act
        # 激活函数
    )

#卷积BatchNorm组
def conv_bn_layer(input,
                  ch_out,
                  filter_size,
                  stride,
                  padding,
                  act=None,
                  groups=64,
                  name=None):
    tmp = fluid.layers.conv2d(
        input=input,
        filter_size=filter_size,
        num_filters=ch_out,
        stride=stride,
        padding=padding,
        act=None,
        bias_attr=False,
        param_attr=fluid.ParamAttr(
            name=name+"_conv_weight_3_",
            initializer=conv_initializer,
            trainable=True),
    )
    return fluid.layers.batch_norm(
        input=tmp,
        act=act,
        param_attr=fluid.ParamAttr(
            name=name+"_bn_weight_2_",
            initializer=bn_initializer,
            trainable=True),
        # 指定权重参数属性的对象
        bias_attr=None,
        # 指定偏置的属性的对象
        moving_mean_name=name + '_bn_3',
        # moving_mean的名称
        moving_variance_name=name + '_bn_4',
        # moving_variance的名称
        name=name + '_bn_',
        momentum=0.5,
    )

def conv_layer(input,
                  ch_out,
                  filter_size,
                  stride,
                  padding,
                  act=None,
                  name=None):
    return fluid.layers.conv2d(
        input=input,
        filter_size=filter_size,
        num_filters=ch_out,
        stride=stride,
        padding=padding,
        act=None,
        bias_attr=False,
        param_attr=fluid.ParamAttr(
            name=name+"_conv_weight_4_",
            initializer=conv_initializer,
            trainable=True),
    )

###判别器
def D(x):
    # (128 + 2 * 1 - 4) / 2 + 1 = 64
    x = conv_layer(x, 128, 4, 2, 1, act=None, name='d_conv_1')
    x = fluid.layers.leaky_relu(x,alpha=0.2,name='d_leaky_relu_1')

    # (64 + 2 * 1 - 4) / 2 + 1 = 32
    x = conv_bn_layer(x, 256, 4, 2, 1, act=None, name='d_conv_bn_2')
    x = fluid.layers.leaky_relu(x,alpha=0.2,name='d_leaky_relu_2')

    # (32 + 2 * 1 - 4) / 2 + 1 = 16
    x = conv_bn_layer(x, 385, 4, 2, 1, act=None, name='d_conv_bn_3')
    x = fluid.layers.leaky_relu(x,alpha=0.2,name='d_leaky_relu_3')

    # (16 + 2 * 1 - 4) / 2 + 1 = 8
    x = conv_bn_layer(x, 768, 4, 2, 1, act=None, name='d_conv_bn_4')
    x = fluid.layers.leaky_relu(x,alpha=0.2,name='d_leaky_relu_4')

    # (8 + 2 * 1 - 4) / 2 + 1 = 4
    x = conv_bn_layer(x, 1024, 4, 2, 1, act=None, name='d_conv_bn_5')
    x = fluid.layers.leaky_relu(x,alpha=0.2,name='d_leaky_relu_5')

    # (4 + 2 * 1 - 4) / 2 + 1 = 2
    x = conv_bn_layer(x, 512, 4, 2, 1, act=None, name='d_conv_bn_6')
    x = fluid.layers.leaky_relu(x, alpha=0.2, name='d_leaky_relu_6')

    x = fluid.layers.reshape(x,shape=[-1, 2048])
    x = fc(x, 2, name='d_fc1')
    return x



###生成器
def G(x):
    x = fluid.layers.reshape(x, shape=[-1, 100, 1, 1])

    # 2 * (1 - 1) - 2 * 0  + 4 = 4
    x = deconv(x, num_filters=2048, filter_size=4, stride=1, padding=0, name='g_deconv_0')
    x = bn(x, name='g_bn_1', act='relu', momentum=0.5)

    # 2 * (4 - 1) - 2 * 1  + 4 = 8
    x = deconv(x, num_filters=1024, filter_size=4, stride=2, padding=1, name='g_deconv_1')
    x = bn(x, name='g_bn_2', act='relu', momentum=0.5)

    # 2 * (8 - 1) - 2 * 1  + 4 = 16
    x = deconv(x, num_filters=1024 , filter_size=4, stride=2, padding=1, name='g_deconv_2')
    x = bn(x, name='g_bn_3', act='relu', momentum=0.5)

    # 2 * (16 - 1) - 2 * 1  + 4 = 32
    x = deconv(x, num_filters=768, filter_size=4, stride=2, padding=1, name='g_deconv_3')
    x = bn(x, name='g_bn_4', act='relu', momentum=0.5)

    # 1 * (16 - 1) - 2 * 1  + 3 = 32
    x = deconv(x, num_filters=512, filter_size=3, stride=1, padding=1, name='g_deconv_4')
    x = bn(x, name='g_bn_5', act='relu', momentum=0.5)

    # 2 * (32 - 1) - 2 * 1  + 4 = 64
    x = deconv(x, num_filters=512, filter_size=4, stride=2, padding=1, name='g_deconv_5')
    x = bn(x, name='g_bn_6', act='relu',momentum=0.5)

    # 2 * (64 - 1) - 2 * 1  + 4 = 128
    x = deconv(x, num_filters=3, filter_size=4, stride=2, padding=1, name='g_deconv_6', act='tanh')
    return x


###损失函数
def loss(x, label):
    return fluid.layers.mean(fluid.layers.softmax_with_cross_entropy(logits=x, label=label))