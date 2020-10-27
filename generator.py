import warnings
from pretreat import *
import paddle.fluid as fluid
from weightInit import bn_initializer, conv_initializer
from network import *
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')
import IPython.display as display

'''定义超参数'''
img_dim = 128
output = "Output/"
G_DIMENSION = 100
#生成图片数量
number = 20
use_gpu = True
dg_program = fluid.Program()
d_program = fluid.Program()


###定义判别器program
# program_guard()接口配合with语句将with block中的算子和变量添加指定的全局主程序（main_program)和启动程序（start_progrom)
with fluid.program_guard(d_program):
    # 输入图片大小为128*128
    img = fluid.layers.data(name='img', shape=[None,3,img_dim,img_dim], dtype='float32')
    # 标签shape=1
    label = fluid.layers.data(name='label', shape=[None,1], dtype='int64')
    d_logit = D(img)
    d_loss = loss(x=d_logit, label=label)

###定义生成器program
with fluid.program_guard(dg_program):
    noise = fluid.layers.data(name='noise', shape=[None,G_DIMENSION], dtype='float32')
    # 噪声数据作为输入得到生成照片
    g_img = G(x=noise)
    g_program = dg_program.clone()
    g_program_test = dg_program.clone(for_test=True)

    # 判断生成图片为真实样本的概率
    dg_logit = D(g_img)

    # 计算生成图片被判别为真实样本的loss
    dg_loss = loss(
        x=dg_logit,
        label=fluid.layers.fill_constant_batch_size_like(input=noise, dtype='int64', shape=[-1,1], value=1)
    )

if use_gpu:
    exe = fluid.Executor(fluid.CUDAPlace(0))
else:
    exe = fluid.Executor(fluid.CPUPlace())
start_program = fluid.default_startup_program()
exe.run(start_program)
fluid.io.load_persistables(exe,'work/Model/D/',d_program)
fluid.io.load_persistables(exe,'work/Model/G/',dg_program)
try:
    const_n = []
    for m in range(number):
        noise = np.random.uniform(low=0.0, high=1.0,size=[G_DIMENSION]).astype('float32')
        const_n.append(noise)
    const_n = np.array(const_n).astype('float32')
    generated_image = exe.run(g_program, feed={'noise': const_n}, fetch_list=[g_img])[0]
    for j in range(number):
        image = generated_image[j].transpose()
        plt.figure(figsize=(4,4))
        plt.imshow(image)
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.savefig('work/Generate/generated_' + str(j + 1), bbox_inches='tight')
        plt.close()
except IOError:
    print(IOError)